#!/bin/bash
# Minimal core-only bootstrap for the CircuitKIT discovery probe.
set -euo pipefail
CORE=/tmp/venv_core
if ! command -v uv >/dev/null 2>&1; then pip install -q --break-system-packages uv; fi
uv venv --system-site-packages --seed "$CORE"
"$CORE/bin/pip" install --no-cache-dir \
  "transformers>=5.15,<5.22" \
  "huggingface-hub>=1.5,<2" \
  "transformer-lens==3.8.0" \
  "datasets>=3.0" \
  "accelerate>=1.4" \
  "sentencepiece" \
  "protobuf" \
  "scikit-learn" \
  "hf_transfer" \
  "circuitkit==0.2.3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$CORE/bin/python" /code/scripts/discover_only.py "$@"
