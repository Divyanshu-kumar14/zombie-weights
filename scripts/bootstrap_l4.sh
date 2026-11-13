#!/bin/bash
# Fallback bootstrap for running the L4 smoke test on a stock PyTorch image
# when the pre-baked dual-venv image is unavailable.  Creates the same two
# isolated environments in /tmp and execs the smoke script.
#
# Usage (inside an hf jobs container):
#   bash /code/scripts/bootstrap_l4.sh
set -euo pipefail

CORE=/tmp/venv_core
ALIGN=/tmp/venv_aligntune

if ! command -v uv >/dev/null 2>&1; then
  pip install -q --break-system-packages uv
fi

uv venv --system-site-packages --seed "$CORE"
"$CORE/bin/pip" install --no-cache-dir \
  "transformers>=5.15,<5.22" \
  "trl>=0.12,<1.15" \
  "peft>=0.18,<1" \
  "huggingface-hub>=1.5,<2" \
  "transformer-lens==3.8.0" \
  "bitsandbytes>=0.41" \
  "datasets>=3.0" \
  "accelerate>=1.4" \
  "sentencepiece" \
  "protobuf" \
  "scikit-learn" \
  "hf_transfer" \
  "curatorkit[hf,connectors]" \
  "circuitkit==0.2.3" \
  "safetune==0.1.18" \
  "auditkit==1.1.11"

uv venv --system-site-packages --seed "$ALIGN"
"$ALIGN/bin/pip" install --no-cache-dir \
  "torchvision==0.28.0" \
  "transformers>=5.15,<5.22" \
  "peft>=0.18,<1" \
  "datasets>=4.7.0" \
  "accelerate>=1.6" \
  "trl==1.7.1" \
  "torchao==0.18.0" \
  "aligntune==0.1.35"

export ALIGNTUNE_BIN="$ALIGN/bin/aligntune"
exec "$CORE/bin/python" /code/scripts/smoke_l4.py "$@"
