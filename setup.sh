#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CUSTOM_TOOLS_REPO="${CUSTOM_TOOLS_REPO:-https://github.com/Franri3008/Custom-Tools.git}"
CUSTOM_TOOLS_DIR="${CUSTOM_TOOLS_DIR:-$PROJECT_DIR/../Custom-Tools}"
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-E2B-it}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-gemma4}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"
VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm.log}"

say() { printf "\n\033[1;36m==>\033[0m %s\n" "$*"; }
die() { printf "\n\033[1;31merror:\033[0m %s\n" "$*" >&2; exit 1; }

say "[1/5] OS packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    git build-essential curl jq
else
  echo "(non-apt system — assuming python3.11 + git + curl are already present)"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found on PATH"

say "[2/5] Custom-Tools sibling checkout"
if [[ ! -d "$CUSTOM_TOOLS_DIR" ]]; then
  git clone --depth 1 "$CUSTOM_TOOLS_REPO" "$CUSTOM_TOOLS_DIR"
else
  echo "(found $CUSTOM_TOOLS_DIR — leaving as is)"
fi

say "[3/5] Python venv"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
pip install -q -U pip wheel

say "[4/5] Project + Custom-Tools (editable)"
pip install -q -r requirements-dev.txt
pip install -q -e "$CUSTOM_TOOLS_DIR"
pip install -q -e .

say "[5/5] vLLM nightly + transformers-from-source (required for gemma4)"
pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install -q --upgrade --force-reinstall --no-deps git+https://github.com/huggingface/transformers.git
pip install -q --upgrade 'huggingface_hub>=1.0'
pip install -q --no-deps 'compressed-tensors==0.14.0.1'

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "(logging into huggingface)"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
fi

say "done"
cat <<EOF

  Setup complete. The vLLM server is NOT started automatically — launch it
  yourself when you want it (see README "Running vLLM" for the copy-paste command).

  Next steps:

  1.  Edit .env (already copied from .env.example?) and fill in:
         DROPBOX_TOKEN, OPENAI_API_KEY, HF_TOKEN,
         LLM__BACKEND=vllm
         LLM__VLLM_BASE_URL=http://localhost:${VLLM_PORT}/v1
         LLM__VLLM_MODEL=${VLLM_SERVED_NAME}

  2.  Start vLLM in a separate tmux window — see README.

  3.  Run the pipeline:
         python main.py sync && python main.py run-all
EOF
