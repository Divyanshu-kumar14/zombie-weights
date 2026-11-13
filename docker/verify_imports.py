"""Import verification for the dual-venv L4 image.

Usage:
    /opt/venv_core/bin/python verify_imports.py core
    /opt/venv_aligntune/bin/python verify_imports.py aligntune

Exits non-zero on any failure so the Docker build fails loudly.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys


def _version(mod: str) -> str:
    m = importlib.import_module(mod)
    return getattr(m, "__version__", "unknown")


def check_core() -> dict:
    import circuitkit  # must be imported before transformer_lens to install Cohere/Gemma patches
    import torch
    import transformers
    import trl
    import peft
    import huggingface_hub
    import transformer_lens
    import curatorkit
    import safetune
    import auditkit

    if importlib.util.find_spec("torchao") is not None:
        raise RuntimeError("venv_core must NOT contain torchao")

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    else:
        device = "cpu"

    return {
        "mode": "core",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "transformer_lens": transformer_lens.__version__ if hasattr(transformer_lens, "__version__") else "n/a",
        "curatorkit": curatorkit.__version__,
        "circuitkit": circuitkit.__version__,
        "safetune": getattr(safetune, "__version__", "unknown"),
        "auditkit": getattr(auditkit, "__version__", "unknown"),
        "torchao_present": False,
    }


def check_aligntune() -> dict:
    import torch
    import torchao
    import trl
    import transformers
    import aligntune

    assert trl.__version__ == "1.7.1", f"trl must be 1.7.1, got {trl.__version__}"

    return {
        "mode": "aligntune",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "aligntune": getattr(aligntune, "__version__", "unknown"),
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "core"
    try:
        info = check_core() if mode == "core" else check_aligntune()
    except Exception as exc:  # noqa: BLE001 - build gate must surface everything
        print(f"IMPORT CHECK FAILED ({mode}): {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print("IMPORT CHECK OK: " + json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
