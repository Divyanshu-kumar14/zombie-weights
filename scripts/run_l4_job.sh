#!/bin/bash
# Launch the Hour-1 L4 smoke test.
#
# Pre-baked image (preferred, requires the GHCR image to be public or HF Jobs
# to have registry credentials):
#   USE_PREBAKED=1 bash scripts/run_l4_job.sh
#
# Fallback: stock PyTorch image + on-job bootstrap (installs the same two
# environments in /tmp). Costs ~$0.05 extra L4 time.
set -euo pipefail

if [[ "${USE_PREBAKED:-0}" == "1" ]]; then
  IMAGE="ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest"
  CMD=(/opt/venv_core/bin/python /opt/zombie-weights/scripts/smoke_l4.py)
else
  IMAGE="pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime"
  CMD=(bash /code/scripts/bootstrap_l4.sh)
fi

hf jobs run \
  --flavor l4x1 \
  --timeout 3600 \
  --name "zw-smoke-$(date +%H%M)" \
  -s HF_TOKEN \
  -v hf://datasets/tony0014/zombie-weights-code:/code:ro \
  "$IMAGE" "${CMD[@]}"
