"""
Qwen3-1.7B-Base を Alpaca データでフルパラメータ SFT するスクリプト。
DeepSpeed ZeRO-3 + マルチ GPU 対応。

起動例:
    deepspeed --num_gpus=4 train.py
    deepspeed --num_gpus=4 train.py --epochs 1 --max_length 512
"""
import argparse
import json
import os
from typing import Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── デフォルト設定 ────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-1.7B"
DATA_PATH    = "./alpaca_data.json"
OUTPUT_DIR   = "./output_qwen3_alpaca"
MAX_LENGTH   = 1024
IGNORE_INDEX = -100

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# ── Dataset ───────────────────────────────────────────────────
class AlpacaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.input_ids_list: list[torch.Tensor] = []
        self.labels_list:    list[torch.Tensor] = []

        skipped = 0
        for item in raw:
            instruction = item["instruction"]
            inp         = item.get("input", "")
            output      = item["output"]

            if not output.strip():
                skipped += 1
                continue

            prompt = (
                PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
                if inp
                else PROMPT_WITHOUT_INPUT.format(instruction=instruction)
            )
            full_text = prompt + output + tokenizer.eos_token

            full_ids   = tokenizer(full_text, add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt,    add_special_tokens=False).input_ids

            # max_length 超えはトランケート
            full_ids = full_ids[:max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]

            # 末尾パディング
            pad_len   = max_length - len(full_ids)
            input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels    = labels   + [IGNORE_INDEX]           * pad_len

            self.input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            self.labels_list.append(   torch.tensor(labels,    dtype=torch.long))

        if skipped:
            print(f"[warn] 空 output によりスキップされたサンプル数: {skipped}")
        print(f"[info] データセットサイズ: {len(self.input_ids_list)} 件")

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids_list[idx]
        return {
            "input_ids":      input_ids,
            "labels":         self.labels_list[idx],
            "attention_mask": (input_ids != 0).long(),
        }


# ── メイン ────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",  default=MODEL_NAME)
    p.add_argument("--data_path",   default=DATA_PATH)
    p.add_argument("--output_dir",  default=OUTPUT_DIR)
    p.add_argument("--max_length",  type=int, default=MAX_LENGTH)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=2,  help="per_device_train_batch_size")
    p.add_argument("--grad_accum",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--save_steps",  type=int, default=500)
    p.add_argument("--local_rank",  type=int, default=-1, help="DeepSpeed が自動設定")
    return p.parse_args()


def train() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    dataset = AlpacaDataset(args.data_path, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        deepspeed="./ds_config.json",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("[info] 学習を開始します...")
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[info] モデルを保存しました: {args.output_dir}")


if __name__ == "__main__":
    train()
