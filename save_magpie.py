"""llm-jp/magpie-sft-v1.0 を HuggingFace からダウンロードしてローカル JSON に保存。"""
from datasets import load_dataset
import json

SAVE_PATH = "magpie_data.json"

print("llm-jp/magpie-sft-v1.0 をダウンロード中...")
ds = load_dataset("llm-jp/magpie-sft-v1.0", split="train")

records = []
for row in ds:
    records.append({
        "id":            row["id"],
        "conversations": row["conversations"],
    })

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"保存完了: {SAVE_PATH}  ({len(records)} 件)")
