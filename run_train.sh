#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 学習起動スクリプト — setup.sh が NUM_GPUS を自動更新します
# 手動変更も可能です
# ============================================================

NUM_GPUS=4   # ← setup.sh が自動書き換え

CYAN='\033[0;36m'; NC='\033[0m'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 仮想環境確認
if [[ ! -f ".venv/bin/python" ]]; then
    echo "仮想環境が見つかりません。先に bash setup.sh を実行してください。" >&2
    exit 1
fi

# ── オプション ────────────────────────────────────────────────
# 追加引数は train.py にそのまま渡されます
# 例: bash run_train.sh --epochs 1 --max_length 512
EXTRA_ARGS="${*:-}"

echo -e "${CYAN}[run_train]${NC} GPU 数: ${NUM_GPUS}, 追加引数: ${EXTRA_ARGS:-なし}"
echo ""

# ── DeepSpeed で起動 ──────────────────────────────────────────
.venv/bin/deepspeed \
    --num_gpus="${NUM_GPUS}" \
    train.py \
    ${EXTRA_ARGS}
