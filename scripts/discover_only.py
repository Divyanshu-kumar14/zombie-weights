"""High-confidence CircuitKIT discovery probe (A100 80 GB profile).

Fixes the N=1 attribution limitation: 32 examples x 5 IG steps at
max_seq_len=256 over MLP nodes. Batch size is raised to use the 80 GB A100
without changing the attribution math (EAP-IG averages per-example scores).

Artifact semantics (CircuitKIT 0.2.3):
  * ``discover_circuit`` returns the node-level pruning artifact (list[str]).
  * the true node importances are written to ``<stem>_scores.json`` under
    ``node_scores``.
  * if the side-car is missing, the artifact order is the descending-importance
    pruning order and is used as the fallback.

Emits ``phase3/discover_probe.json`` with the ranked top-6 layers.
"""
from __future__ import annotations

import argparse
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
N_EXAMPLES = int(os.environ.get("ZW_N_EXAMPLES", "32"))
IG_STEPS = int(os.environ.get("ZW_IG_STEPS", "5"))
MAX_SEQ_LEN = int(os.environ.get("ZW_MAX_SEQ_LEN", "256"))
BATCH_SIZE = int(os.environ.get("ZW_BATCH_SIZE", "8"))
TOP_K = int(os.environ.get("ZW_TOP_K", "6"))


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_strings(key)
            yield from _walk_strings(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _walk_strings(value)


def _layer_from_node(node: str) -> int | None:
    match = re.fullmatch(r"MLP[ ._-]*(\d+)", str(node).strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def rank_layers(artifact, scores: dict[str, float], top_k: int) -> dict:
    """Aggregate node importances into a per-layer ranking.

    MLP-scope discovery emits one node per layer (``MLP <n>``), so summing and
    maxing over a layer's nodes are equivalent today but summing is robust if
    a future run emits neuron-level nodes.
    """
    per_layer: dict[int, float] = {}
    for node, score in scores.items():
        layer = _layer_from_node(node)
        if layer is not None:
            per_layer[layer] = per_layer.get(layer, 0.0) + abs(float(score))

    artifact_order: list[int] = []
    for node in _walk_strings(artifact):
        layer = _layer_from_node(node)
        if layer is not None and layer not in artifact_order:
            artifact_order.append(layer)

    if per_layer:
        ranked = [layer for layer, _ in sorted(per_layer.items(), key=lambda kv: -kv[1])]
        source = "node_scores"
    else:
        ranked = artifact_order
        source = "artifact_order"
        per_layer = {layer: float(top_k - i) for i, layer in enumerate(artifact_order)}

    # Keep any artifact-only layers that somehow lack a score.
    for layer in artifact_order:
        if layer not in ranked:
            ranked.append(layer)

    return {
        "rank_source": source,
        "ranked_layers": ranked,
        "top_layers": ranked[:top_k],
        "top_layers_sorted": sorted(ranked[:top_k]),
        "per_layer_score": {str(k): round(v, 8) for k, v in sorted(per_layer.items())},
        "artifact_order": artifact_order,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    parser.add_argument("--ig-steps", type=int, default=IG_STEPS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    import torch
    import circuitkit as ck
    from circuitkit.quick import build_discovery_config
    from circuitkit.api import discover_circuit

    workdir = Path("/tmp/zw_discover")
    workdir.mkdir(parents=True, exist_ok=True)
    telemetry = Telemetry(REPO, workdir)
    artifact = workdir / "circuit_highconf.pt"
    sidecar = workdir / "circuit_highconf_scores.json"

    config = {
        "model": args.model,
        "algorithm": "eap-ig",
        "level": "node",
        "n_examples": args.n_examples,
        "ig_steps": args.ig_steps,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "scope": "mlp",
        "task": "wmdp",
        "configs": ["wmdp-cyber"],
    }
    print(f"[discover] config={json.dumps(config)}", flush=True)

    reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
    if reset_peak is not None:
        reset_peak()
    torch.cuda.empty_cache()
    gc.collect()
    t0 = time.time()

    result: dict = {"ok": False, "config": config, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        tl_model = ck.load_model(args.model, dtype="bfloat16", device="cuda")
        cfg = build_discovery_config(
            tl_model,
            "wmdp",
            algorithm="eap-ig",
            level="node",
            n_examples=args.n_examples,
            batch_size=args.batch_size,
            ig_steps=args.ig_steps,
            max_seq_len=args.max_seq_len,
            configs=["wmdp-cyber"],
            scope="mlp",
            model_name=args.model,
            output_path=str(artifact),
        )
        # Free the probe model before the engine loads its own copy (80 GB is
        # roomy, but this keeps peak VRAM attributable to attribution alone).
        tl_model.to("cpu")
        del tl_model
        gc.collect()
        torch.cuda.empty_cache()

        circuit = discover_circuit(cfg)

        scores: dict[str, float] = {}
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                scores = dict(payload.get("node_scores") or {})
            except Exception as exc:  # noqa: BLE001 - artifact order fallback
                print(f"[discover] sidecar parse failed: {exc}", flush=True)
        if not scores:
            node_scores = getattr(getattr(circuit, "circuit_scores", None), "node_scores", None)
            if node_scores:
                scores = dict(node_scores)

        ranking = rank_layers(circuit, scores, args.top_k)
        result.update(ranking)
        result.update(
            {
                "ok": True,
                "n_nodes": len(scores),
                "n_artifact_nodes": len(list(_walk_strings(circuit))),
                "artifact_repr": repr(circuit)[:2000],
                "elapsed_s": round(time.time() - t0, 1),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
        )
        if len(ranking["top_layers"]) < args.top_k:
            result["ok"] = False
            result["error"] = f"only {len(ranking['top_layers'])} layers ranked, expected {args.top_k}"
    except Exception as exc:  # noqa: BLE001 - telemetry must capture the failure
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[discover] failed: {exc}", flush=True)
        try:
            result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass

    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["elapsed_s"] = result.get("elapsed_s", round(time.time() - t0, 1))
    telemetry.push_json("phase3/discover_probe.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
