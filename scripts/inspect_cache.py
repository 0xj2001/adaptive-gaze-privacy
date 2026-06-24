"""
Inspect a tensor cache before using it for training.

Example:
    python scripts/inspect_cache.py --cache_dir data/cache_gazebase_ak250_64ms
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect cached GazeBase tensors")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--expected_persons", type=int, default=None)
    parser.add_argument("--max_zero_ratio", type=float, default=0.01)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    config = manifest.get("config", {})
    recordings = manifest.get("recordings", [])
    identity_field = config.get("identity_field", "subject_id")
    persons = {r.get("person_id") for r in recordings if "person_id" in r}
    identities = {r.get(identity_field) for r in recordings if identity_field in r}

    total_windows = 0
    zero_windows = 0
    global_sum = 0.0
    global_sq_sum = 0.0
    global_count = 0
    loaded_files = 0
    shape_examples = set()

    for rec in recordings:
        if args.max_files is not None and loaded_files >= args.max_files:
            break
        cache_file = rec.get("cache_file")
        if not cache_file:
            continue
        payload = torch.load(cache_dir / cache_file, map_location="cpu", weights_only=False)
        x = payload["x"] if isinstance(payload, dict) else payload
        loaded_files += 1
        shape_examples.add(tuple(x.shape))
        total_windows += int(x.shape[0])
        zero_windows += int((x.abs().sum(dim=(1, 2)) == 0).sum().item())
        global_sum += float(x.sum().item())
        global_sq_sum += float((x * x).sum().item())
        global_count += int(x.numel())

    mean = global_sum / max(1, global_count)
    variance = global_sq_sum / max(1, global_count) - mean * mean
    std = max(0.0, variance) ** 0.5
    zero_ratio = zero_windows / max(1, total_windows)

    report = {
        "cache_dir": str(cache_dir),
        "config": config,
        "manifest_summary": manifest.get("summary", {}),
        "recordings": len(recordings),
        "loaded_files": loaded_files,
        "identities": len(identities),
        "persons": len(persons),
        "total_windows_loaded": total_windows,
        "zero_windows_loaded": zero_windows,
        "zero_window_ratio_loaded": zero_ratio,
        "tensor_mean_loaded": mean,
        "tensor_std_loaded": std,
        "shape_examples": sorted(shape_examples),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = []
    if args.expected_persons is not None and len(persons) != args.expected_persons:
        failures.append(f"persons={len(persons)} expected={args.expected_persons}")
    if zero_ratio > args.max_zero_ratio:
        failures.append(f"zero_window_ratio={zero_ratio:.6f} > {args.max_zero_ratio}")
    if std <= 0:
        failures.append("tensor std is zero")
    expected_features = config.get("features")
    if expected_features is not None and expected_features != ["x", "y"]:
        print(f"Warning: features={expected_features}; main experiment expects ['x', 'y']")

    if failures:
        raise SystemExit("Cache inspection failed: " + "; ".join(failures))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
