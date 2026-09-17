"""Azure ML scorer for the standalone employee tiny Transformer."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, context_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.position_embedding = nn.Embedding(context_size, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(128)
        self.lm_head = nn.Linear(128, vocab_size)
        self.context_size = context_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 0, 0.0)
        hidden = self.encoder(hidden, mask=mask)
        return self.lm_head(self.norm(hidden))


def init() -> None:
    global model, token_to_id, id_to_token, unknown_id, device, context_size
    # Azure ML mounts the registered job output one level down (model/), so
    # search the tree rather than assume the depth.
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    checkpoint_path = next(
        os.path.join(root, name)
        for root, _, files in os.walk(model_dir)
        for name in files
        if name == "employee_tiny_transformer.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    id_to_token = checkpoint["vocab"]
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    unknown_id = token_to_id.get("\0", 0)
    context_size = int(checkpoint["context_size"])
    model = TinyTransformer(len(id_to_token), context_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cpu")
    model.to(device)
    model.eval()


def _payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        value = json.loads(raw)
    else:
        value = raw
    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object.")
    return value


def run(raw: Any) -> dict[str, Any]:
    payload = _payload(raw)
    question = str(payload.get("question", payload.get("instruction", ""))).strip()
    if not question:
        raise ValueError("Provide a non-empty 'question' or 'instruction'.")

    prompt = f"Instruction: {question}\nAnswer:"
    tokens = [token_to_id.get(character, unknown_id) for character in prompt]
    started = time.perf_counter()
    generated: list[int] = []
    max_new_tokens = min(max(1, int(payload.get("max_new_tokens", 160))), 512)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = tokens[-context_size:]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            next_id = int(model(input_ids)[0, -1].argmax().item())
            tokens.append(next_id)
            generated.append(next_id)

    text = "".join(id_to_token[index] for index in generated)
    return {
        "answer": text.split("\0", 1)[0].strip(),
        "model": "employee-from-scratch-tiny-transformer",
        "base_model": None,
        "fine_tuning": False,
        "domain": "synthetic_employee",
        "proof_of_concept": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
