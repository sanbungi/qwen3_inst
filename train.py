"""
Qwen3-1.7B-Base を llm-jp/magpie-sft-v1.0 (日本語) でフルパラメータ SFT。
DeepSpeed ZeRO-3 + マルチ GPU 対応。

推奨起動コマンド (RTX 4090 x2 = 24GB x 2):
    deepspeed --num_gpus=2 train.py \\
        --batch_size 2 --grad_accum 8 --max_length 1024 --epochs 1

VRAM が厳しい場合 (CPU offload):
    deepspeed --num_gpus=2 train.py \\
        --deepspeed ./ds_config_offload.json \\
        --batch_size 1 --grad_accum 16 --max_length 1024 --epochs 1
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

# ── Attention 実装の自動選択 (FlashAttention2 → sdpa) ─────────
def _select_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        # bf16/fp16 + Ampere 以上が必要
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                print(f"[info] FlashAttention2 を使用 (flash_attn={flash_attn.__version__})")
                return "flash_attention_2"
    except ImportError:
        pass
    print("[info] SDPA attention を使用 (FlashAttn 未利用)")
    return "sdpa"

ATTN_IMPL = _select_attn_impl()

# ── デフォルト設定 ────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-1.7B"
DATA_PATH    = "./magpie_data.json"
OUTPUT_DIR   = "./output_qwen3_magpie"
MAX_LENGTH   = 1024          # Magpie は応答が長め (Alpaca より長い設定)
IGNORE_INDEX = -100


# ── Dataset (Qwen チャットテンプレート + 動的パディング) ─────
class MagpieDataset(Dataset):
    """
    llm-jp/magpie-sft-v1.0 形式 (conversations: user/assistant 2 turn) を
    Qwen のチャットテンプレートでフォーマットし、assistant 部分のみ学習対象とする。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.examples: List[Dict[str, List[int]]] = []
        skipped_empty = 0
        skipped_format = 0

        for item in raw:
            convs = item.get("conversations", [])
            if len(convs) < 2 or convs[0]["role"] != "user" or convs[1]["role"] != "assistant":
                skipped_format += 1
                continue

            user_msg      = {"role": "user",      "content": convs[0]["content"]}
            assistant_msg = {"role": "assistant", "content": convs[1]["content"]}

            if not assistant_msg["content"].strip():
                skipped_empty += 1
                continue

            # プロンプト部分 (生成開始トークン込み)
            prompt_text = tokenizer.apply_chat_template(
                [user_msg],
                tokenize=False,
                add_generation_prompt=True,
            )
            # フル会話
            full_text = tokenizer.apply_chat_template(
                [user_msg, assistant_msg],
                tokenize=False,
                add_generation_prompt=False,
            )

            full_ids   = tokenizer(full_text,   add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

            full_ids   = full_ids[:max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            # ユーザ発話とテンプレートヘッダ部分は IGNORE_INDEX でマスク
            labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]

            self.examples.append({
                "input_ids": full_ids,
                "labels":    labels,
            })

        if skipped_empty:
            print(f"[warn] 空 assistant 応答スキップ: {skipped_empty} 件")
        if skipped_format:
            print(f"[warn] フォーマット不一致スキップ: {skipped_format} 件")
        print(f"[info] データセットサイズ: {len(self.examples)} 件")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        return self.examples[idx]


# ── メイン ────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",   default=MODEL_NAME)
    p.add_argument("--data_path",    default=DATA_PATH)
    p.add_argument("--output_dir",   default=OUTPUT_DIR)
    p.add_argument("--deepspeed",    default="./ds_config.json",
                   help="DeepSpeed 設定ファイル (./ds_config_offload.json で CPU offload)")
    p.add_argument("--max_length",   type=int, default=MAX_LENGTH)
    p.add_argument("--epochs",       type=int, default=1)
    p.add_argument("--batch_size",   type=int, default=2,  help="per_device_train_batch_size")
    p.add_argument("--grad_accum",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_steps",   type=int, default=500)
    p.add_argument("--local_rank",   type=int, default=-1, help="DeepSpeed が自動設定")
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
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"{args.model_name} のトークナイザに chat_template がありません。"
            " Qwen3 系を使ってください。"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    dataset = MagpieDataset(args.data_path, tokenizer, args.max_length)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
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
