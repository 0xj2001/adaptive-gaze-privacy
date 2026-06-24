"""
Build a persistent tensor cache for GazeBase CSV recordings.

Usage:
    python scripts/build_cache.py --config configs/default.yaml
    python scripts/build_cache.py --config configs/default.yaml --limit 20 --force
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.gazebase_loader import apply_round_protocol, discover_recordings, load_gazebase_csv
from src.data.preprocessing import preprocess_ak250_sin_xy, preprocess_gaze_sequence


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_name_for_recording(source_csv: str) -> str:
    digest = hashlib.sha1(source_csv.replace("\\", "/").encode("utf-8")).hexdigest()[:12]
    stem = Path(source_csv).stem
    return f"recordings/{stem}_{digest}.pt"


def build_recording_cache(args: tuple[dict, dict, str, bool]) -> dict:
    rec, cache_cfg, output_dir, force = args
    output_path = Path(output_dir)
    cache_file = cache_name_for_recording(rec["file"])
    cache_path = output_path / cache_file
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            meta = payload.get("meta", {})
            return {
                **rec,
                "cache_file": cache_file,
                "n_samples": int(meta.get("n_samples", payload["x"].shape[1])),
                "n_windows": int(meta.get("n_windows", payload["x"].shape[0])),
                "n_chunks": int(meta.get("n_chunks", meta.get("n_windows", payload["x"].shape[0]))),
                "zero_windows": int(meta.get(
                    "zero_windows",
                    (payload["x"].abs().sum(dim=(1, 2)) == 0).sum().item(),
                )),
                "status": "cached",
            }
        except Exception:
            pass

    try:
        df = load_gazebase_csv(rec["file"])
        n_samples = len(df)
        window_size = int(cache_cfg["window_size"])
        window_stride = int(cache_cfg["window_stride"])
        features = cache_cfg["features"]
        sampling_rate = int(cache_cfg["sampling_rate"])
        preprocess_mode = cache_cfg.get("preprocess_mode", "legacy_window_zscore")
        sampling_rate_in = int(cache_cfg.get("sampling_rate_in", sampling_rate))
        sampling_rate_out = int(cache_cfg.get("sampling_rate_out", sampling_rate))

        windows = []
        if preprocess_mode == "ak250_sin_xy":
            arr_full = preprocess_ak250_sin_xy(
                df,
                sampling_rate_in=sampling_rate_in,
                sampling_rate_out=sampling_rate_out,
            )
            n_windows = max(1, (len(arr_full) - window_size) // window_stride + 1)
            for window_idx in range(n_windows):
                start = window_idx * window_stride
                end = start + window_size
                arr = arr_full[start:end]
                x = torch.tensor(arr, dtype=torch.float32)
                if x.shape[0] < window_size:
                    pad = torch.zeros(window_size - x.shape[0], x.shape[1])
                    x = torch.cat([x, pad], dim=0)
                windows.append(x)
        elif preprocess_mode == "legacy_window_zscore":
            n_windows = max(1, (n_samples - window_size) // window_stride + 1)
            for window_idx in range(n_windows):
                start = window_idx * window_stride
                end = start + window_size
                window_df = df.iloc[start:end].copy()
                arr = preprocess_gaze_sequence(
                    window_df,
                    features,
                    sampling_rate=sampling_rate,
                )
                x = torch.tensor(arr, dtype=torch.float32)
                if x.shape[0] < window_size:
                    pad = torch.zeros(window_size - x.shape[0], x.shape[1])
                    x = torch.cat([x, pad], dim=0)
                windows.append(x)
        else:
            raise ValueError(f"Unknown preprocess_mode={preprocess_mode!r}")

        x_windows = torch.stack(windows, dim=0)
        zero_windows = int((x_windows.abs().sum(dim=(1, 2)) == 0).sum().item())
        payload = {
            "x": x_windows,
            "meta": {
                **rec,
                "n_samples": n_samples,
                "n_windows": n_windows,
                "n_chunks": n_windows,
                "zero_windows": zero_windows,
                "features": features,
                "window_size": window_size,
                "window_stride": window_stride,
                "sampling_rate": sampling_rate,
                "preprocess_mode": preprocess_mode,
                "identity_field": cache_cfg.get("identity_field", "subject_id"),
                "sampling_rate_in": sampling_rate_in,
                "sampling_rate_out": sampling_rate_out,
            },
        }
        torch.save(payload, cache_path)

        return {
            **rec,
            "cache_file": cache_file,
            "n_samples": n_samples,
            "n_windows": n_windows,
            "n_chunks": n_windows,
            "zero_windows": zero_windows,
            "status": "built",
        }
    except Exception as exc:
        return {
            **rec,
            "cache_file": cache_file,
            "n_samples": 0,
            "n_windows": 0,
            "status": "failed",
            "error": repr(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Build GazeBase tensor cache")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    output_dir = args.output_dir or data_cfg.get("cache_dir", "data/cache_gazebase")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(data_cfg["data_dir"], data_cfg["tasks"])
    protocol_summary = None
    if data_cfg.get("split_strategy") == "round_protocol":
        recordings, protocol_summary = apply_round_protocol(recordings)
    if args.limit is not None:
        recordings = recordings[:args.limit]
    if not recordings:
        raise SystemExit(f"No recordings found under {data_cfg['data_dir']}")

    cache_cfg = {
        "source_data_dir": data_cfg["data_dir"],
        "tasks": data_cfg["tasks"],
        "features": data_cfg["features"],
        "sampling_rate": data_cfg.get("sampling_rate", 1000),
        "sampling_rate_in": data_cfg.get("sampling_rate_in", data_cfg.get("sampling_rate", 1000)),
        "sampling_rate_out": data_cfg.get("sampling_rate_out", data_cfg.get("sampling_rate", 1000)),
        "window_size": data_cfg["window_size"],
        "window_stride": data_cfg["window_stride"],
        "preprocess_mode": data_cfg.get("preprocess_mode", "legacy_window_zscore"),
        "identity_field": data_cfg.get("identity_field", "subject_id"),
        "split_strategy": data_cfg.get("split_strategy", "recording"),
        "subwindow_size": data_cfg.get("subwindow_size"),
    }

    worker_args = [(rec, cache_cfg, str(output_path), args.force) for rec in recordings]
    results = []

    print(f"Building cache: {output_path}")
    print(f"Recordings: {len(worker_args)}")
    print(f"Workers: {args.num_workers}")

    if args.num_workers <= 1:
        for item in tqdm(worker_args, desc="Caching recordings"):
            results.append(build_recording_cache(item))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(build_recording_cache, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching recordings"):
                results.append(future.result())

    results = sorted(results, key=lambda r: r["file"])
    failed = [r for r in results if r["status"] == "failed"]
    ok_results = [r for r in results if r["status"] != "failed"]
    identity_field = cache_cfg["identity_field"]
    total_windows = sum(int(r["n_windows"]) for r in ok_results)
    zero_windows = sum(int(r.get("zero_windows", 0)) for r in ok_results)
    split_counts = {}
    for split_name in ["train", "val", "test"]:
        split_records = [r for r in ok_results if r.get("split_name") == split_name]
        if split_records:
            split_counts[f"{split_name}_recordings"] = len(split_records)
            split_counts[f"{split_name}_subjects"] = len({r[identity_field] for r in split_records})
            split_counts[f"{split_name}_windows"] = sum(int(r["n_windows"]) for r in split_records)
            split_counts[f"{split_name}_rounds"] = sorted({int(r["round_id"]) for r in split_records})
    manifest = {
        "version": 1,
        "config": cache_cfg,
        "recordings": ok_results,
        "failed": failed,
        "protocol_summary": protocol_summary,
        "summary": {
            "recordings": len(results),
            "cached_or_built": len(results) - len(failed),
            "failed": len(failed),
            "total_windows": total_windows,
            "zero_windows": zero_windows,
            "zero_window_ratio": zero_windows / max(1, total_windows),
            "identities": len({r[identity_field] for r in ok_results}),
            "raw_subject_ids": len({r["raw_subject_id"] for r in ok_results if "raw_subject_id" in r}),
            "persons": len({r["person_id"] for r in ok_results if "person_id" in r}),
            **split_counts,
        },
    }

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2))
    if failed:
        print("Some recordings failed; inspect manifest['failed'] before training.")


if __name__ == "__main__":
    main()
