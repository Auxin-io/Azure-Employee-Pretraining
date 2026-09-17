"""Train a tiny causal Transformer from random weights on employee text only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, context_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.position_embedding = nn.Embedding(context_size, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True, activation="gelu",
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
            torch.full(
                (length, length), float("-inf"), device=tokens.device
            ),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 0, 0.0)
        hidden = self.encoder(hidden, mask=mask)
        return self.lm_head(self.norm(hidden))


def read_text(path: str) -> str:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Employee dataset is empty: {path}")
    return "\n".join(
        f"Instruction: {row['instruction']}\nAnswer: {row['output']}\n"
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    train_text = read_text(args.corpus_data)
    validation_text = read_text(args.validation_data)
    corpus = train_text + "\n" + validation_text
    alphabet = sorted(set(corpus) | {"\0"})
    token_to_id = {token: index for index, token in enumerate(alphabet)}
    encode = lambda text: [token_to_id[token] for token in text]
    train_tokens = torch.tensor(encode(train_text), dtype=torch.long)
    validation_tokens = torch.tensor(encode(validation_text), dtype=torch.long)
    context_size = min(args.context_size, max(8, len(train_tokens) - 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(len(alphabet), context_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for step in range(args.steps):
        starts = torch.randint(
            0, max(1, len(train_tokens) - context_size), (8,)
        )
        batch = torch.stack(
            [train_tokens[start:start + context_size + 1] for start in starts]
        ).to(device)
        logits = model(batch[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, len(alphabet)), batch[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step} train_loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        validation_window = validation_tokens[:context_size + 1]
        validation_input = validation_window[:-1].unsqueeze(0).to(device)
        validation_target = validation_window[1:].to(device)
        validation_logits = model(validation_input)[0]
        validation_loss = nn.functional.cross_entropy(
            validation_logits, validation_target
        ).item()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": alphabet,
            "context_size": context_size,
            "architecture": "tiny-transformer-from-scratch",
            "training_data": "synthetic employee dataset only",
        },
        output / "employee_tiny_transformer.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "validation_loss": validation_loss,
                "perplexity": math.exp(validation_loss),
                "steps": args.steps,
                "architecture": "tiny-transformer-from-scratch",
                "base_model": None,
                "fine_tuning": False,
                "dataset": "synthetic_employee_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"validation_loss={validation_loss:.4f}")


if __name__ == "__main__":
    main()
