#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen3-1.7B Alpaca SFT — Interactive Setup Script
# Usage: bash setup.sh
# ============================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  Qwen3-1.7B Alpaca フルパラメータ SFT — 環境セットアップ"
echo "============================================================"
echo ""

# ── 1. GPU 確認 ──────────────────────────────────────────────
info "GPU 環境を確認中..."
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi が見つかりません。GPU ドライバを確認してください。"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ok "検出された GPU 数: ${NUM_GPUS}"

# ── 2. GPU 数の確認 ──────────────────────────────────────────
read -rp "$(echo -e "${CYAN}使用する GPU 数 [${NUM_GPUS}]: ${NC}")" INPUT_GPUS
NUM_GPUS="${INPUT_GPUS:-$NUM_GPUS}"
ok "使用 GPU 数: ${NUM_GPUS}"

# ── 3. HuggingFace トークン ───────────────────────────────────
echo ""
if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN は環境変数から取得済みです。"
else
    read -rp "$(echo -e "${CYAN}HuggingFace トークン (Enter でスキップ): ${NC}")" HF_TOKEN
    if [[ -n "$HF_TOKEN" ]]; then
        export HF_TOKEN
        ok "HF_TOKEN を設定しました。"
    else
        warn "HF_TOKEN 未設定。モデルがゲートされている場合は失敗します。"
    fi
fi

# ── 4. uv のインストール ──────────────────────────────────────
echo ""
info "uv を確認中..."
if ! command -v uv &>/dev/null; then
    info "uv をインストール中..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    ok "uv インストール完了"
else
    ok "uv は既にインストール済み: $(uv --version)"
fi

# ── 5. Python 仮想環境の作成 ──────────────────────────────────
echo ""
info "Python 3.11 仮想環境を作成中..."
if [[ ! -d ".venv" ]]; then
    uv venv .venv --python 3.11
    ok ".venv 作成完了"
else
    ok ".venv は既に存在します。スキップします。"
fi

PYTHON=".venv/bin/python"

# ── 6. 依存パッケージのインストール ──────────────────────────
echo ""
info "パッケージをインストール中 (uv pip)..."
uv pip install --python .venv/bin/python \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

uv pip install --python .venv/bin/python \
    setuptools \
    wheel \
    transformers \
    accelerate \
    deepspeed \
    datasets \
    sentencepiece \
    protobuf \
    packaging \
    ninja
ok "パッケージインストール完了"

# ── 6.5 FlashAttention 2 のインストール (Ampere 以上で有効) ──
echo ""
info "GPU の compute capability を確認中..."
CUDA_CC=$($PYTHON -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo "0")
if [[ "$CUDA_CC" -ge 8 ]]; then
    info "compute capability ${CUDA_CC}.x → FlashAttention2 をインストール中..."
    if uv pip install --python .venv/bin/python flash-attn --no-build-isolation; then
        ok "FlashAttention2 インストール完了"
    else
        warn "FlashAttention2 のインストールに失敗しました (sdpa にフォールバック)"
    fi
else
    warn "GPU が Ampere 未満のため FlashAttention2 をスキップ (sdpa を使用)"
fi

# ── 7. DeepSpeed バージョン確認 ──────────────────────────────
echo ""
info "DeepSpeed バージョン確認..."
$PYTHON -c "import deepspeed; print('DeepSpeed:', deepspeed.__version__)"

# ── 8. Magpie データのダウンロード ────────────────────────────
echo ""
if [[ -f "magpie_data.json" ]]; then
    RECORD_COUNT=$(python3 -c "import json; d=json.load(open('magpie_data.json')); print(len(d))" 2>/dev/null || echo "?")
    warn "magpie_data.json が既に存在します (${RECORD_COUNT} 件)。スキップします。"
    read -rp "$(echo -e "${CYAN}再ダウンロードしますか? [y/N]: ${NC}")" REDOWNLOAD
    if [[ "${REDOWNLOAD:-N}" =~ ^[Yy]$ ]]; then
        info "Magpie データをダウンロード中..."
        $PYTHON save_magpie.py
    fi
else
    info "Magpie データをダウンロード中..."
    $PYTHON save_magpie.py
fi
ok "magpie_data.json 準備完了"

# ── 9. 完了メッセージ ────────────────────────────────────────
echo ""
echo "============================================================"
ok "セットアップ完了！"
echo ""
echo "  通常の学習コマンド (24GB x ${NUM_GPUS} GPU 推奨):"
echo -e "  ${CYAN}.venv/bin/deepspeed --num_gpus=${NUM_GPUS} train.py \\\\${NC}"
echo -e "  ${CYAN}    --batch_size 2 --grad_accum 8 --max_length 512 --epochs 1${NC}"
echo ""
echo "  VRAM が厳しい場合 (CPU offload, 遅いがメモリ節約):"
echo -e "  ${CYAN}.venv/bin/deepspeed --num_gpus=${NUM_GPUS} train.py \\\\${NC}"
echo -e "  ${CYAN}    --deepspeed ./ds_config_offload.json \\\\${NC}"
echo -e "  ${CYAN}    --batch_size 1 --grad_accum 16 --max_length 512 --epochs 1${NC}"
echo ""
echo "  ※ batch_size と grad_accum は ds_config*.json と一致させる必要があります"
echo "============================================================"
echo ""
