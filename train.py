"""
Qwen3-1.7B-Base を Alpaca データでフルパラメータ SFT するスクリプト。
DeepSpeed ZeRO-3 + マルチ GPU 対応。

推奨起動コマンド (RTX 4090 x2 = 24GB x 2):
    deepspeed --num_gpus=2 train.py \\
        --batch_size 2 --grad_accum 8 --max_length 512 --epochs 1

VRAM がさらに厳しい場合 (CPU offload 利用):
    deepspeed --num_gpus=2 train.py \\
        --deepspeed ./ds_config_offload.json \\
        --batch_size 1 --grad_accum 16 --max_length 512 --epochs 1
"""
import argparse
import json
import os
from typing import Dict, List

# ── OOM 対策: フラグメンテーション緩和を自動有効化 ────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── デフォルト設定 ────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-1.7B"
DATA_PATH    = "./alpaca_data.json"
OUTPUT_DIR   = "./output_qwen3_alpaca"
MAX_LENGTH   = 512          # 24GB x 2 で安全に動く長さ (1024 だと OOM しやすい)
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


# ── Dataset (動的パディング用に可変長で保持) ─────────────────
class AlpacaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.examples: List[Dict[str, List[int]]] = []
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

            full_ids = full_ids[:max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            # プロンプト部分は IGNORE_INDEX でマスク (レスポンスのみ学習)
            labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]

            self.examples.append({
                "input_ids": full_ids,
                "labels":    labels,
            })

        if skipped:
            print(f"[warn] 空 output によりスキップ: {skipped} 件")
        print(f"[info] データセットサイズ: {len(self.examples)} 件")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        return self.examples[idx]


# ── メイン ────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",  default=MODEL_NAME)
    p.add_argument("--data_path",   default=DATA_PATH)
    p.add_argument("--output_dir",  default=OUTPUT_DIR)
    p.add_argument("--deepspeed",   default="./ds_config.json",
                   help="DeepSpeed 設定ファイル (./ds_config_offload.json で CPU offload)")
    p.add_argument("--max_length",  type=int, default=MAX_LENGTH)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=2,  help="per_device_train_batch_size")
    p.add_argument("--grad_accum",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
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
        attn_implementation="sdpa",     # FlashAttn 風に省メモリ
    )
    model.config.use_cache = False      # gradient checkpointing と両立させる
    model.enable_input_require_grads()

    dataset = AlpacaDataset(args.data_path, tokenizer, args.max_length)

    # 動的パディング: バッチ内の最長に合わせるため固定 max_length より省メモリ
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,           # Tensor Core 効率化
        label_pad_token_id=IGNORE_INDEX,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("[info] 学習を開始します...")
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[info] モデルを保存しました: {args.output_dir}")


if __name__ == "__main__":
    train()
