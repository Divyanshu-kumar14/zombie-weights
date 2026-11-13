#!/bin/bash
set -euo pipefail
echo "python: $(python --version 2>&1)"
python -c "import torch, sys; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo "code mount:"
ls /code
echo "scripts:"
ls /code/scripts
echo "gpu:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
