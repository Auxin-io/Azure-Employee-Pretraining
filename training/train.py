"""Train a small causal Transformer from random weights on employee Q/A only.

No pretrained model, no tokenizer download: the vocabulary is built from the
training rows themselves. The task is closed-book recall over ten employee
documents - question in, answer sentence out - so the loss is taken on the
answer tokens only and the model learns the question -> answer mapping.

    python train.py --train-data train.jsonl --validation-data validation.jsonl \
        --test-data test.jsonl --output-dir out --steps 3000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from torch import nn

# ------------------------------------------------------------ tokenizer ---
# Words, numbers, ids and dates stay whole ("EMP-8373", "2026-05-04", "42.2",
# "1,240.50", "Jonas's"); everything else is a single-character token.
TOKEN_RE = re.compile(r"\w+(?:[-.,:'/]\w+)*|[^\w\s]")
PAD, BOS, SEP, EOS, UNK = "<pad>", "<bos>", "<sep>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, SEP, EOS, UNK]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r"\s+([,.;:?!)])", r"\1", out)
    out = re.sub(r"([($])\s+", r"\1", out)
    return out


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str]) -> "Vocab":
        seen = sorted({t for text in texts for t in tokenize(text)})
        return cls(SPECIALS + seen)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        return detokenize([self.itos[i] for i in ids if self.itos[i] not in SPECIALS])

    def __len__(self) -> int:
        return len(self.itos)


# ---------------------------------------------------------------- model ---
class CausalTransformer(nn.Module):
    """GPT-style decoder: token + position embeddings, pre-norm blocks, tied head."""

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


# ----------------------------------------------------------------- data ---
def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# Cheap surface noise so the model does not key on exact casing/punctuation.
def augment(question: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        question = question.lower()
    elif r < 0.35:
        question = question.rstrip("?.! ")
    elif r < 0.45:
        question = question.rstrip("?.! ") + "?"
    elif r < 0.55:
        question = rng.choice(["Hi, ", "Please: ", "Question: ", "Hey, "]) + question
    return question


def encode_pair(vocab: Vocab, question: str, answer: str, context: int) -> tuple[list[int], list[int]]:
    """Returns (ids, loss_mask): mask is 1 on answer tokens + <eos> only."""
    q = [vocab.stoi[BOS]] + vocab.encode(question) + [vocab.stoi[SEP]]
    a = vocab.encode(answer) + [vocab.stoi[EOS]]
    ids = (q + a)[:context + 1]
    mask = ([0] * len(q) + [1] * len(a))[:context + 1]
    return ids, mask


def make_batch(rows: list[dict], vocab: Vocab, context: int, rng: random.Random,
               device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad = vocab.stoi[PAD]
    seqs, masks = [], []
    for row in rng.sample(rows, batch_size):
        ids, mask = encode_pair(vocab, augment(row["instruction"], rng), row["output"], context)
        seqs.append(ids)
        masks.append(mask)
    n = max(len(s) for s in seqs)
    ids = torch.full((batch_size, n), pad, dtype=torch.long)
    lm = torch.zeros((batch_size, n), dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s)
        lm[i, :len(m)] = torch.tensor(m, dtype=torch.bool)
    return ids.to(device), lm.to(device), (ids == pad).to(device)


def exact_match(model: CausalTransformer, vocab: Vocab, rows: list[dict], device: torch.device,
                max_new: int = 64) -> tuple[float, list[dict]]:
    model.eval()
    hits, samples = 0, []
    for row in rows:
        prefix = [vocab.stoi[BOS]] + vocab.encode(row["instruction"]) + [vocab.stoi[SEP]]
        pred = vocab.decode(model.generate(prefix, vocab.stoi[EOS], max_new))
        gold = vocab.decode(vocab.encode(row["output"]))   # same detokenizer on both sides
        ok = pred == gold
        hits += ok
        if len(samples) < 5:
            samples.append({"question": row["instruction"], "prediction": pred, "gold": gold, "match": ok})
    model.train()
    return hits / max(1, len(rows)), samples


# ----------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--validation-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = read_rows(args.train_data)
    val = read_rows(args.validation_data)
    test = read_rows(args.test_data)
    vocab = Vocab.build([r["instruction"] for r in train] + [r["output"] for r in train])
    print(f"device={device} train={len(train)} val={len(val)} test={len(test)} vocab={len(vocab)}")

    model = CausalTransformer(len(vocab), args.context, args.d_model, args.layers, args.heads,
                              args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params:,}  (all trained from random init)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        ids, lm, pad = make_batch(train, vocab, args.context, rng, device, args.batch_size)
        logits = model(ids[:, :-1], pad[:, :-1])
        target = ids[:, 1:].clone()
        target[~lm[:, 1:]] = -100                       # loss on answer tokens only
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                           ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"step={step} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
        if step % 1000 == 0:
            acc, _ = exact_match(model, vocab, val, device)
            print(f"  validation exact match={acc:.3f}", flush=True)

    val_acc, val_samples = exact_match(model, vocab, val, device)
    test_acc, test_samples = exact_match(model, vocab, test, device)
    print(f"final validation exact match={val_acc:.3f}  test (unseen phrasing) exact match={test_acc:.3f}")
    for s in test_samples:
        print(f"  [{'ok' if s['match'] else 'XX'}] {s['question']}\n       -> {s['prediction']}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.itos,
        "config": {"context": args.context, "d_model": args.d_model, "layers": args.layers,
                   "heads": args.heads, "dropout": 0.0},
        "architecture": "causal-transformer-from-scratch",
    }, out / "employee_model.pt")
    (out / "metrics.json").write_text(json.dumps({
        "parameters": n_params, "steps": args.steps, "final_train_loss": loss.item(),
        "validation_exact_match": val_acc, "test_exact_match": test_acc,
        "train_rows": len(train), "vocab_size": len(vocab),
        "base_model": None, "fine_tuning": False, "dataset": "closed_book_employee",
        "samples": test_samples, "train_seconds": round(time.time() - t0),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
