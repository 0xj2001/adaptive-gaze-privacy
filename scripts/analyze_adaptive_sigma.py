"""Analyze ANIAE adaptive latent-noise allocation by task and subwindow."""

import argparse
import csv
import json
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

_mpl_cache = project_root / "experiments" / "round_protocol_results" / ".matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache.resolve()))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.round_protocol import (  # noqa: E402
    RoundProtocolChunkDataset,
    load_manifest,
    make_loader,
    records_by_split,
)
from src.models.autoencoder import ANIAE  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_aniae(path: Path, config: dict, device: torch.device) -> ANIAE:
    if not path.exists():
        raise FileNotFoundError(f"Missing ANIAE checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ae_config = ckpt.get("config", config)
    model = ANIAE(ae_config).to(device)
    state = ckpt.get("aniae_state", ckpt.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model


def build_loader(config: dict, split: str, seed: int, args):
    data_cfg = config["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), data_cfg["tasks"])
    records = splits[split]
    label_map = {
        person_id: idx
        for idx, person_id in enumerate(sorted({int(r["person_id"]) for r in records}))
    }
    dataset = RoundProtocolChunkDataset(
        cache_dir,
        records,
        identity_field=data_cfg["identity_field"],
        label_map=label_map,
        tasks=data_cfg["tasks"],
        cache_lru_size=int(data_cfg.get("cache_lru_size", 64)),
    )
    loader = make_loader(
        dataset,
        batch_size=int(args.batch_size or config["task_training"]["batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        seed=seed,
    )
    return dataset, loader


@torch.no_grad()
def collect_sigma(model: ANIAE, loader, config: dict, device: torch.device, max_batches: int | None = None):
    subwindow_size = int(config["data"]["subwindow_size"])
    idx_to_task = {idx: task for idx, task in enumerate(config["data"]["tasks"])}
    values = defaultdict(list)
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc="SigmaEval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        bsz, seq_len, n_features = x.shape
        if seq_len % subwindow_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by subwindow_size={subwindow_size}")
        n_sub = seq_len // subwindow_size
        sub = x.reshape(bsz, n_sub, subwindow_size, n_features).reshape(
            bsz * n_sub, subwindow_size, n_features
        )
        out = model(sub)
        sigma = out["sigma"].reshape(bsz, n_sub, -1).detach().cpu().numpy()
        sigma_per_sub = sigma.mean(axis=2)
        task_labels_np = task_labels.detach().cpu().numpy()
        for row_idx, task_idx in enumerate(task_labels_np):
            task = idx_to_task[int(task_idx)]
            for sub_idx in range(n_sub):
                values[(task, sub_idx)].append(float(sigma_per_sub[row_idx, sub_idx]))
    return values


def summarize(seed_values: dict[int, dict[tuple[str, int], list[float]]], tasks: list[str], n_subwindows: int):
    rows = []
    pooled = defaultdict(list)
    for seed, values in seed_values.items():
        for task in tasks:
            for sub_idx in range(n_subwindows):
                vals = np.asarray(values.get((task, sub_idx), []), dtype=float)
                if vals.size == 0:
                    continue
                mean = float(vals.mean())
                std = float(vals.std(ddof=0))
                rows.append({
                    "seed": seed,
                    "task": task,
                    "subwindow_idx": sub_idx,
                    "sigma_mean": mean,
                    "sigma_std": std,
                    "sigma_cv": float(std / max(abs(mean), 1e-8)),
                    "n": int(vals.size),
                })
                pooled[(task, sub_idx)].extend(vals.tolist())
    pooled_rows = []
    for task in tasks:
        for sub_idx in range(n_subwindows):
            vals = np.asarray(pooled.get((task, sub_idx), []), dtype=float)
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            pooled_rows.append({
                "seed": "pooled",
                "task": task,
                "subwindow_idx": sub_idx,
                "sigma_mean": mean,
                "sigma_std": std,
                "sigma_cv": float(std / max(abs(mean), 1e-8)),
                "n": int(vals.size),
            })
    return rows, pooled_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_sigma(path: Path, pooled_rows: list[dict], tasks: list[str], n_subwindows: int, fixed_sigma: float) -> None:
    matrix = np.full((len(tasks), n_subwindows), np.nan)
    for row in pooled_rows:
        if row["task"] not in tasks:
            continue
        matrix[tasks.index(row["task"]), int(row["subwindow_idx"])] = float(row["sigma_mean"])
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("No finite sigma values available for plotting")
    adaptive_min = max(0.0, float(np.nanmin(finite)) * 0.95)
    adaptive_max = float(np.nanmax(finite)) * 1.05
    line_ymax = max(adaptive_max * 1.18, 0.08)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.8),
        dpi=220,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    heat_ax, line_ax = axes
    im = heat_ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=adaptive_min,
        vmax=adaptive_max,
    )
    heat_ax.set_yticks(range(len(tasks)), tasks)
    heat_ax.set_xlabel("Subwindow index")
    heat_ax.set_ylabel("Task")
    heat_ax.set_title("(a) Adaptive sigma by task and subwindow", loc="left", fontsize=9)
    cbar = fig.colorbar(im, ax=heat_ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean sigma", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    x = np.arange(n_subwindows)
    for task_idx, task in enumerate(tasks):
        line_ax.plot(x, matrix[task_idx], linewidth=1.2, label=task)
    if fixed_sigma <= line_ymax:
        line_ax.axhline(
            fixed_sigma,
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label=f"Fixed sigma={fixed_sigma:g}",
        )
    else:
        line_ax.axhline(
            line_ymax * 0.97,
            color="#444444",
            linestyle="--",
            linewidth=1.0,
            label=f"Fixed sigma={fixed_sigma:g} (above scale)",
        )
        line_ax.annotate(
            f"fixed sigma={fixed_sigma:g}",
            xy=(n_subwindows - 1, line_ymax * 0.97),
            xytext=(-4, -10),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color="#333333",
        )
    line_ax.set_ylim(0.0, line_ymax)
    line_ax.set_xlabel("Subwindow index")
    line_ax.set_ylabel("Mean sigma")
    line_ax.set_title("(b) Task-wise sigma profiles", loc="left", fontsize=9)
    line_ax.grid(True, alpha=0.25)
    line_ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_latex_snippet(path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/analyze_adaptive_sigma.py",
        "\\begin{figure}[!t]",
        "    \\centering",
        "    \\includegraphics[width=\\linewidth]{figures/adaptive_sigma_allocation.png}",
        "    \\caption{Adaptive latent-noise allocation. ANIAE predicts data-dependent noise magnitudes across tasks and 64 ms subwindows; the dashed line marks the fixed-noise autoencoder setting used as the matched baseline.}",
        "    \\label{fig:adaptive_sigma}",
        "\\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ANIAE adaptive sigma")
    parser.add_argument("--config", type=str, default="configs/round_protocol_ekyt_aniae.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/round_protocol_results/adaptive_sigma"))
    parser.add_argument("--figure_out", type=Path, default=Path("results/adaptive_sigma/adaptive_sigma_allocation.png"))
    parser.add_argument("--latex_out", type=Path, default=None, help="Optional private manuscript LaTeX snippet path.")
    parser.add_argument("--aniae_checkpoint_template", type=str, default="experiments/round_protocol_results/proposed_aniae_seed{seed}/checkpoint_best.pt")
    parser.add_argument("--fixed_sigma", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_subwindows = int(config["data"]["window_size"]) // int(config["data"]["subwindow_size"])
    seed_values = {}
    for seed in args.seeds:
        set_global_seed(seed)
        _, loader = build_loader(config, args.split, seed, args)
        path = Path(args.aniae_checkpoint_template.format(seed=seed))
        model = load_aniae(path, config, device)
        seed_values[seed] = collect_sigma(model, loader, config, device, args.max_batches)
        print(f"Collected sigma values for seed={seed} from {path}")

    rows, pooled_rows = summarize(seed_values, config["data"]["tasks"], n_subwindows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "adaptive_sigma_by_task.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"seed_rows": rows, "pooled_rows": pooled_rows}), f, indent=2, ensure_ascii=False)
    write_csv(args.output_dir / "adaptive_sigma_by_task.csv", rows)
    write_csv(args.output_dir / "adaptive_sigma_by_task_pooled.csv", pooled_rows)
    plot_sigma(args.figure_out, pooled_rows, config["data"]["tasks"], n_subwindows, args.fixed_sigma)
    if args.latex_out is not None:
        write_latex_snippet(args.latex_out)
    print(f"Adaptive sigma results written to {args.output_dir}")
    print(f"Figure written to {args.figure_out}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
