"""Train semantic task probes on protected gaze under the Round Protocol."""

import argparse
import csv
import json
import multiprocessing
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.round_protocol import (  # noqa: E402
    RoundProtocolChunkDataset,
    load_manifest,
    make_loader,
    records_by_split,
)
from src.models.autoencoder import ANIAE  # noqa: E402
from src.models.biometric import GazeTaskEvaluator  # noqa: E402


TASK_SETS = {
    "non_ran": ["FXS", "HSS", "TEX"],
    "four_class": ["FXS", "HSS", "RAN", "TEX"],
}

METHODS = ("no_protection", "fixed_ae_sigma1", "aniae")


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


def checkpoint_path(template: str, seed: int) -> Path:
    return Path(template.format(seed=seed))


def load_aniae_transform(path: Path, config: dict, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Missing ANIAE checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ae_config = ckpt.get("config", config)
    model = ANIAE(ae_config).to(device)
    state = ckpt.get("aniae_state", ckpt.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    subwindow_size = int(config["data"]["subwindow_size"])

    @torch.no_grad()
    def transform(x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_features = x.shape
        if seq_len % subwindow_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by subwindow_size={subwindow_size}")
        n_sub = seq_len // subwindow_size
        sub = x.reshape(bsz, n_sub, subwindow_size, n_features).reshape(
            bsz * n_sub, subwindow_size, n_features
        )
        out = model(sub)
        protected = out["x_hat"].reshape(bsz, n_sub, subwindow_size, n_features).reshape(
            bsz, seq_len, n_features
        )
        return torch.clamp(protected, -1.0, 1.0)

    return transform


def make_transform(method: str, config: dict, seed: int, device: torch.device, args):
    if method == "no_protection":
        return lambda x: x, None
    if method == "fixed_ae_sigma1":
        path = checkpoint_path(args.fixed_ae_checkpoint_template, seed)
        return load_aniae_transform(path, config, device), str(path)
    if method == "aniae":
        path = checkpoint_path(args.aniae_checkpoint_template, seed)
        return load_aniae_transform(path, config, device), str(path)
    raise ValueError(f"Unknown method={method}")


def make_probe_model(config: dict, num_tasks: int, device: torch.device) -> GazeTaskEvaluator:
    task_cfg = config.get("task_model", {})
    bio_cfg = config.get("biometric_model", {})
    return GazeTaskEvaluator(
        input_dim=len(config["data"]["features"]),
        num_tasks=num_tasks,
        hidden_dim=int(task_cfg.get("hidden_dim", 128)),
        base_channels=int(bio_cfg.get("base_channels", 64)),
        growth_rate=int(bio_cfg.get("growth_rate", 16)),
        block_layers=task_cfg.get("block_layers", [3, 3, 3]),
        dropout=float(task_cfg.get("dropout", 0.2)),
    ).to(device)


def build_loaders(config: dict, task_labels: list[str], seed: int, args):
    data_cfg = config["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), task_labels)
    if task_labels == TASK_SETS["non_ran"]:
        for split, records in splits.items():
            bad = sorted({record["task"] for record in records if record["task"] == "RAN"})
            if bad:
                raise RuntimeError(f"RAN records leaked into non-RAN {split}: {bad}")
    label_map = {
        person_id: idx
        for idx, person_id in enumerate(
            sorted({int(r["person_id"]) for records in splits.values() for r in records})
        )
    }
    datasets = {
        split: RoundProtocolChunkDataset(
            cache_dir,
            records,
            identity_field=data_cfg["identity_field"],
            label_map=label_map,
            tasks=task_labels,
            cache_lru_size=int(data_cfg.get("cache_lru_size", 64)),
        )
        for split, records in splits.items()
    }
    train_cfg = config["task_training"]
    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
            seed=seed,
        ),
        "val": make_loader(
            datasets["val"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(args.val_num_workers),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
            seed=seed,
        ),
        "test": make_loader(
            datasets["test"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(args.val_num_workers),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
            seed=seed,
        ),
    }
    return datasets, loaders


def run_epoch(model, loader, criterion, device, transform, num_tasks: int, optimizer=None, max_batches=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true = []
    y_pred = []
    desc = "ProbeTrain" if is_train else "ProbeEval"
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        task_labels = task_labels.to(device)
        with torch.no_grad():
            protected = transform(x).detach()
        logits = model(protected)
        loss = criterion(logits, task_labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += loss.item() * task_labels.size(0)
        y_true.extend(task_labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    labels = list(range(num_tasks))
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else float("nan"),
        "per_class_recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist() if y_true else [],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else [],
        "labels": labels,
    }


def train_probe(config: dict, loaders: dict, transform, task_labels: list[str], seed: int, device: torch.device, args):
    train_cfg = config["task_training"]
    model = make_probe_model(config, len(task_labels), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"] if args.max_epochs is None else min(train_cfg["epochs"], args.max_epochs))
    patience = int(args.patience if args.patience is not None else train_cfg.get("patience", 6))
    best_state = None
    best_val = -float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, transform, len(task_labels), optimizer, args.max_train_batches
        )
        val_metrics = run_epoch(
            model, loaders["val"], criterion, device, transform, len(task_labels), None, args.max_eval_batches
        )
        score = val_metrics["macro_f1"]
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })
        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_metrics = run_epoch(
        model, loaders["test"], criterion, device, transform, len(task_labels), None, args.max_eval_batches
    )
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "test": test_metrics,
        "history": history,
    }


def summarize_rows(rows: list[dict], task_labels: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = []
    for method, method_rows in grouped.items():
        values = {
            "accuracy": [r["metrics"]["test"]["accuracy"] for r in method_rows],
            "macro_f1": [r["metrics"]["test"]["macro_f1"] for r in method_rows],
            "retention_accuracy": [r["retention_accuracy"] for r in method_rows],
            "retention_macro_f1": [r["retention_macro_f1"] for r in method_rows],
        }
        item = {"method": method, "n": len(method_rows), "task_labels": task_labels}
        for key, vals in values.items():
            arr = np.asarray(vals, dtype=float)
            item[f"{key}_mean"] = float(arr.mean()) if arr.size else float("nan")
            item[f"{key}_std"] = float(arr.std(ddof=0)) if arr.size else float("nan")
        summary.append(item)
    order = {name: idx for idx, name in enumerate(METHODS)}
    return sorted(summary, key=lambda item: order.get(item["method"], 999))


def fmt_pm(mean: float, std: float, scale: float = 100.0) -> str:
    return f"${mean * scale:.2f} \\pm {std * scale:.2f}$"


def write_latex_table(path: Path, summaries_by_task_set: dict[str, list[dict]]) -> None:
    method_names = {
        "no_protection": "No Protection",
        "fixed_ae_sigma1": "Fixed-Noise AE",
        "aniae": "Proposed ANIAE",
    }
    lines = [
        "% Auto-generated by scripts/evaluate_semantic_utility_probes.py",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Protected semantic task-probe utility. The non-RAN probe excludes the RAN task used by enrollment/probe biometric evaluation.}",
        "\\label{tab:semantic_utility}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Task set & Method & Accuracy (\\%) & Macro-F1 (\\%) & Acc. Ret. (\\%) & F1 Ret. (\\%) \\\\",
        "\\midrule",
    ]
    first = True
    for task_set in ["non_ran", "four_class"]:
        summary = summaries_by_task_set.get(task_set, [])
        if not summary:
            continue
        if not first:
            lines.append("\\midrule")
        first = False
        label = "Non-RAN" if task_set == "non_ran" else "Four-class"
        for item in summary:
            lines.append(
                f"{label} & {method_names.get(item['method'], item['method'])} & "
                f"{fmt_pm(item['accuracy_mean'], item['accuracy_std'])} & "
                f"{fmt_pm(item['macro_f1_mean'], item['macro_f1_std'])} & "
                f"{fmt_pm(item['retention_accuracy_mean'], item['retention_accuracy_std'])} & "
                f"{fmt_pm(item['retention_macro_f1_mean'], item['retention_macro_f1_std'])} \\\\"
            )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def run_task_set(config: dict, task_set: str, device: torch.device, args) -> tuple[list[dict], list[dict]]:
    task_labels = TASK_SETS[task_set]
    rows = []
    for seed in args.seeds:
        set_global_seed(seed)
        datasets, loaders = build_loaders(config, task_labels, seed, args)
        print(
            f"task_set={task_set} seed={seed} "
            f"chunks train/val/test={len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}"
        )
        seed_rows = []
        raw_acc = None
        raw_f1 = None
        for method in METHODS:
            set_global_seed(seed)
            transform, ckpt = make_transform(method, config, seed, device, args)
            print(f"Training semantic probe: task_set={task_set} seed={seed} method={method}")
            metrics = train_probe(config, loaders, transform, task_labels, seed, device, args)
            acc = metrics["test"]["accuracy"]
            f1 = metrics["test"]["macro_f1"]
            if method == "no_protection":
                raw_acc = max(acc, 1e-8)
                raw_f1 = max(f1, 1e-8)
            row = {
                "method": method,
                "seed": seed,
                "task_set": task_set,
                "task_labels": task_labels,
                "checkpoint_path": ckpt,
                "metrics": metrics,
                "test_accuracy": acc,
                "test_macro_f1": f1,
                "retention_accuracy": acc / max(raw_acc or acc, 1e-8),
                "retention_macro_f1": f1 / max(raw_f1 or f1, 1e-8),
            }
            seed_rows.append(row)
        rows.extend(seed_rows)
    return rows, summarize_rows(rows, task_labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train protected semantic utility probes")
    parser.add_argument("--config", type=str, default="configs/round_protocol_ekyt_aniae.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--task_set", choices=["non_ran", "four_class", "both"], default="both")
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/round_protocol_results/semantic_utility"))
    parser.add_argument("--latex_out", type=Path, default=None, help="Optional private manuscript LaTeX table path.")
    parser.add_argument("--fixed_ae_checkpoint_template", type=str, default="experiments/round_protocol_results/fixed_noise_ae_sigma1_seed{seed}/checkpoint_best.pt")
    parser.add_argument("--aniae_checkpoint_template", type=str, default="experiments/round_protocol_results/proposed_aniae_seed{seed}/checkpoint_best.pt")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_num_workers", type=int, default=0)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_sets = ["non_ran", "four_class"] if args.task_set == "both" else [args.task_set]
    all_rows = []
    summaries_by_task_set = {}
    for task_set in task_sets:
        rows, summary = run_task_set(config, task_set, device, args)
        all_rows.extend(rows)
        summaries_by_task_set[task_set] = summary
        with open(args.output_dir / f"{task_set}_rows.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(rows), f, indent=2, ensure_ascii=False)
        with open(args.output_dir / f"{task_set}_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
        write_csv(args.output_dir / f"{task_set}_summary.csv", summary)
    with open(args.output_dir / "semantic_utility_rows.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(all_rows), f, indent=2, ensure_ascii=False)
    print(f"Semantic utility results written to {args.output_dir}")
    if args.latex_out is not None:
        write_latex_table(args.latex_out, summaries_by_task_set)
        print(f"LaTeX table written to {args.latex_out}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
