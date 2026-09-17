"""Azure ML scorer for the from-scratch employee model.

Request:  {"question": "How many hours did Jonas Weber work?", "max_new_tokens": 64}
Response: {"answer": "...", "model": "...", "latency_ms": ...}

The checkpoint carries its own vocabulary and architecture config, so this
file only needs the model class and the tokenizer rules - both copied from
training/train.py and kept identical.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import torch
from torch import nn

TOKEN_RE = re.compile(r"\w+(?:[-.,:'/]\w+)*|[^\w\s]")
PAD, BOS, SEP, EOS, UNK = "<pad>", "<bos>", "<sep>", "<eos>", "<unk>"
SPECIALS = {PAD, BOS, SEP, EOS, UNK}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r"\s+([,.;:?!)])", r"\1", out)
    out = re.sub(r"([($])\s+", r"\1", out)
    return out


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, context: int, d_model: int, n_layers: int,
                 n_heads: int, dropout: float) -> None:
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = ids.shape[1]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=ids.device), diagonal=1)
        x = self.drop(self.tok(ids) + self.pos(torch.arange(n, device=ids.device)))
        x = self.blocks(x, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prefix: list[int], eos: int, max_new: int) -> list[int]:
        ids = list(prefix)
        for _ in range(max_new):
            window = torch.tensor([ids[-self.context:]], device=self.tok.weight.device)
            next_id = int(self(window)[0, -1].argmax())
            if next_id == eos:
                break
            ids.append(next_id)
        return ids[len(prefix):]


model: CausalTransformer
itos: list[str]
stoi: dict[str, int]


def init() -> None:
    global model, itos, stoi
    # Azure ML mounts the registered job output one level down (model/).
    root = os.environ["AZUREML_MODEL_DIR"]
    path = next(os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if f == "employee_model.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    itos = ckpt["vocab"]
    stoi = {t: i for i, t in enumerate(itos)}
    cfg = ckpt["config"]
    model = CausalTransformer(len(itos), cfg["context"], cfg["d_model"], cfg["layers"],
                              cfg["heads"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {path}: vocab={len(itos)} config={cfg}")


def run(raw: Any) -> dict[str, Any]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    question = str(payload.get("question", payload.get("instruction", ""))).strip()
    if not question:
        raise ValueError("Provide a non-empty 'question'.")
    max_new = int(payload.get("max_new_tokens", 64))

    t0 = time.time()
    prefix = [stoi[BOS]] + [stoi.get(t, stoi[UNK]) for t in tokenize(question)] + [stoi[SEP]]
    out = model.generate(prefix, stoi[EOS], max_new)
    answer = detokenize([itos[i] for i in out if itos[i] not in SPECIALS])
    return {
        "answer": answer,
        "model": "employee-from-scratch-causal-transformer",
        "base_model": None,
        "fine_tuning": False,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
