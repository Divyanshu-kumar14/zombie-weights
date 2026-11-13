"""Minimal CircuitKIT discovery probe for the L4 memory budget."""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_l4 import Telemetry  # noqa: E402

MODEL = os.environ.get("ZW_MODEL", "CohereLabs/tiny-aya-global")
REPO = os.environ.get("ZW_TELEMETRY", "tony0014/zombie-weights-telemetry")


def main() -> int:
    import torch
    import circuitkit as ck
    from circuitkit.quick import build_discovery_config
    from circuitkit.api import discover_circuit

    workdir = Path("/tmp/zw_discover")
    workdir.mkdir(parents=True, exist_ok=True)
    telemetry = Telemetry(REPO, workdir)

    results = {}
    for max_seq_len in (128, 64):
        try:
            torch.cuda.empty_cache()
            gc.collect()
            t0 = time.time()
            tl_model = ck.load_model(MODEL, dtype="bfloat16", device="cuda")
            cfg = build_discovery_config(
                tl_model,
                "wmdp",
                algorithm="eap-ig",
                level="node",
                n_examples=1,
                batch_size=1,
                ig_steps=1,
                max_seq_len=max_seq_len,
                configs=["wmdp-cyber"],
                scope="mlp",
                model_name=MODEL,
                output_path=str(workdir / f"circuit_{max_seq_len}.pt"),
            )
            tl_model.to("cpu")
            del tl_model
            gc.collect()
            torch.cuda.empty_cache()
            circuit = discover_circuit(cfg)
            scores = dict(getattr(circuit, "scores", {}) or {})
            if not scores and getattr(circuit, "circuit_scores", None) is not None:
                scores = dict(getattr(circuit.circuit_scores, "node_scores", {}) or {})
            per_layer = {}
            for node, score in scores.items():
                m = re.search(r"blocks\.(\d+)\.", str(node))
                if m:
                    layer = int(m.group(1))
                    per_layer[layer] = max(per_layer.get(layer, 0.0), abs(float(score)))
            ranked = [l for l, _ in sorted(per_layer.items(), key=lambda kv: -kv[1])]
            results[str(max_seq_len)] = {
                "ok": True,
                "n_nodes": len(scores),
                "layers": ranked[:8],
                "per_layer": {str(k): v for k, v in sorted(per_layer.items())},
                "elapsed_s": round(time.time() - t0, 1),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
            telemetry.push_json("phase3/discover_probe.json", results)
            print(json.dumps(results, indent=1), flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            results[str(max_seq_len)] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            telemetry.push_json("phase3/discover_probe.json", results)
            print(f"[discover] seq={max_seq_len} failed: {exc}", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
    return 1


if __name__ == "__main__":
    sys.exit(main())
