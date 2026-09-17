"""Azure ML QLoRA training entry point for the synthetic employee dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from prompt_format import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL file found under {source}")
        source = files[0]
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Employee dataset is empty: {source}")
    required = {"instruction", "output"}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"{source} is missing fields: {sorted(missing)}")
    return rows


def tokenize_fn(tokenizer, max_length: int):
    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        prompt = tokenizer.apply_chat_template(
            build_messages(row["instruction"], row.get("input", "")),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(
            row["output"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        input_ids = (prompt_ids + answer_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    return tokenize


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        torch.set_num_threads(os.cpu_count() or 1)
        print("CUDA is unavailable; using CPU LoRA fallback.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float32, device_map={"": "cpu"}
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    tokenize = tokenize_fn(tokenizer, args.max_seq_length)
    train_ds = Dataset.from_list(train_rows).map(
        tokenize, remove_columns=list(train_rows[0])
    )
    validation_ds = Dataset.from_list(validation_rows).map(
        tokenize, remove_columns=list(validation_rows[0])
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/employee-checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            bf16=use_bf16,
            fp16=torch.cuda.is_available() and not use_bf16,
            gradient_checkpointing=True,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps({"dataset": "synthetic_employee", **metrics}, default=str))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
