"""Alpaca データセットを HuggingFace からダウンロードしてローカル JSON に保存する。"""
from datasets import load_dataset
import json, sys

SAVE_PATH = "alpaca_data.json"

print("tatsu-lab/alpaca をダウンロード中...")
ds = load_dataset("tatsu-lab/alpaca", split="train")
records = [dict(row) for row in ds]

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"保存完了: {SAVE_PATH}  ({len(records)} 件)")
