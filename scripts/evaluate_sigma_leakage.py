"""Evaluate whether ANIAE sigma(z) leaks identity under the Round Protocol."""

import argparse
import csv
import json
import math
import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.round_protocol import load_manifest, load_recording_windows, records_by_split
from src.evaluation.biometric_metrics import _select_enrollment_probe, metrics_from_similarity
from src.models.autoencoder import ANIAE


METRIC_KEYS = ["rank1_ir", "rank5_ir", "eer", "random_rank1", "random_rank5", "mean_rank"]


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def load_aniae_checkpoint(config: dict, checkpoint_path: Path, device: torch.device) -> ANIAE:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ae_config = ckpt.get("config", config)
    model = ANIAE(ae_config).to(device)
    state = ckpt.get("aniae_state", ckpt.get("model_state"))
    if state is None:
        raise KeyError(f"No ANIAE state found in {checkpoint_path}")
    model.load_state_dict(state)
    model.eval()
    if hasattr(model, "set_eval_noise_mode"):
        model.set_eval_noise_mode("deterministic")
    return model


@torch.no_grad()
def embed_sigma_recording(
    model: ANIAE,
    cache_dir: Path,
    record: dict,
    device: torch.device,
    subwindow_size: int,
    batch_size: int,
) -> torch.Tensor:
    x = load_recording_windows(cache_dir, record)
    if x.ndim != 3:
        raise ValueError(f"Expected recording tensor with 3 dims, got {tuple(x.shape)}")
    sigma_chunks = []
    for start in range(0, x.size(0), batch_size):
        batch = x[start:start + batch_size].to(device)
        bsz, seq_len, n_features = batch.shape
        if seq_len % subwindow_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by subwindow_size={subwindow_size}")
        n_sub = seq_len // subwindow_size
        sub = batch.reshape(bsz, n_sub, subwindow_size, n_features).reshape(
            bsz * n_sub,
            subwindow_size,
            n_features,
        )
        out = model(sub)
        sigma_chunks.append(out["sigma"].detach().cpu())
    if not sigma_chunks:
        raise ValueError(f"No chunks found for recording {record.get('cache_file')}")
    sigma_vec = torch.cat(sigma_chunks, dim=0).mean(dim=0)
    return F.normalize(sigma_vec, p=2, dim=0, eps=1e-12)


@torch.no_grad()
def evaluate_sigma_leakage(
    model: ANIAE,
    cache_dir: Path,
    records: list[dict],
    device: torch.device,
    subwindow_size: int,
    batch_size: int,
    task: str,
    enrollment_session: int,
    probe_session: int,
) -> dict:
    enrollment_records, probe_records = _select_enrollment_probe(
        records,
        task=task,
        enrollment_session=enrollment_session,
        probe_session=probe_session,
    )
    people = sorted(set(enrollment_records) & set(probe_records))
    if not people:
        raise ValueError("No enrollment/probe pairs found for sigma leakage evaluation")

    enroll = torch.stack([
        embed_sigma_recording(model, cache_dir, enrollment_records[p], device, subwindow_size, batch_size)
        for p in people
    ])
    probe = torch.stack([
        embed_sigma_recording(model, cache_dir, probe_records[p], device, subwindow_size, batch_size)
        for p in people
    ])
    similarity = probe @ enroll.t()
    metrics = metrics_from_similarity(similarity)
    metrics["num_eval_subjects"] = len(people)
    return metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return float("nan"), float("nan")
    arr = np.asarray(clean, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(key: str, percent: bool = True) -> str:
        mean = summary[f"{key}_mean"]
        std = summary[f"{key}_std"]
        if math.isnan(mean):
            return "n/a"
        scale = 100.0 if percent else 1.0
        return f"${mean * scale:.2f} \\pm {std * scale:.2f}$"

    text = r"""\begin{table}[!t]
\centering
\caption{Sigma-only side-channel check for ANIAE. The enrollment/probe matcher uses recording-level averages of predicted $\sigma(z)$ rather than decoded gaze.}
\label{tab:sigma_leakage}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Representation & Rank-1 IR (\%) & Rank-5 IR (\%) & EER (\%) & Random Rank-1 (\%) \\
\midrule
ANIAE $\sigma(z)$ only & RANKONE & RANKFIVE & EERVAL & RANDOMONE \\
\bottomrule
\end{tabular}}
\end{table}
"""
    text = text.replace("RANKONE", fmt("rank1_ir"))
    text = text.replace("RANKFIVE", fmt("rank5_ir"))
    text = text.replace("EERVAL", fmt("eer"))
    text = text.replace("RANDOMONE", fmt("random_rank1"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def summarize_rows(rows: list[dict]) -> dict:
    summary = {"n_seeds": len({row["seed"] for row in rows})}
    for key in METRIC_KEYS:
        mean, std = mean_std([row.get(key, float("nan")) for row in rows])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ANIAE sigma-only identity leakage")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--checkpoint_template",
        type=str,
        default="experiments/round_protocol_results/proposed_aniae_seed{seed}/checkpoint_best.pt",
    )
    parser.add_argument("--output_dir", type=str, default="experiments/round_protocol_results/sigma_leakage")
    parser.add_argument("--latex_out", type=Path, default=None, help="Optional private manuscript LaTeX table path.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(config["data"]["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), config["data"]["tasks"])
    eval_records = splits[args.split]
    protocol = config["data"].get("round_protocol", {})
    task = protocol.get("enrollment_task", "RAN")
    enrollment_session = int(protocol.get("enrollment_session", 1))
    probe_session = int(protocol.get("probe_session", 2))
    subwindow_size = int(config["data"]["subwindow_size"])

    rows = []
    for seed in args.seeds:
        set_global_seed(seed)
        checkpoint_path = Path(args.checkpoint_template.format(seed=seed))
        if not checkpoint_path.exists():
            print(f"Skipping seed {seed}: missing {checkpoint_path}")
            continue
        model = load_aniae_checkpoint(config, checkpoint_path, device)
        metrics = evaluate_sigma_leakage(
            model,
            cache_dir,
            eval_records,
            device,
            subwindow_size,
            int(args.batch_size),
            task,
            enrollment_session,
            probe_session,
        )
        row = {"seed": seed, "checkpoint": str(checkpoint_path), **metrics}
        rows.append(row)
        print(json.dumps(json_safe(row), ensure_ascii=False))

    if not rows:
        raise SystemExit("No sigma leakage rows were produced")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    with open(output_dir / f"{args.split}_sigma_leakage_rows.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(rows), f, indent=2, ensure_ascii=False)
    with open(output_dir / f"{args.split}_sigma_leakage_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    write_csv(output_dir / f"{args.split}_sigma_leakage_rows.csv", rows)
    write_csv(output_dir / f"{args.split}_sigma_leakage_summary.csv", [summary])
    if args.latex_out is not None:
        write_latex_table(args.latex_out, summary)
    print(f"Sigma leakage summary written to {output_dir}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
