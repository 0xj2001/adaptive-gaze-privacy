"""Task-conditioned semantic retention benchmark for protected gaze.

This script adds a broader utility benchmark without retraining the protection
models. It evaluates three public components:

1. Cross-domain task transfer.
2. Temporal prediction compatibility at multiple horizons.
3. Oculomotor event-statistic retention.

The reported Semantic Retention Index (SRI) is the unweighted mean of these
three component scores. Each component is written separately so the composite
score remains auditable.
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import f1_score
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

_mpl_cache = project_root / "experiments" / "round_protocol_results" / ".matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache.resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

METHOD_DISPLAY = {
    "no_protection": "No Protection",
    "fixed_ae_sigma1": "Fixed-Noise AE",
    "aniae": "Proposed ANIAE",
    "gaussian_sigma0.05": "Gaussian Noise",
    "gaussian_sigma1": "Gaussian Noise",
    "dp_epsilon20": "Laplace DP",
}

METHOD_PARAMETER = {
    "no_protection": "--",
    "fixed_ae_sigma1": "sigma=1.0",
    "aniae": "adaptive",
    "gaussian_sigma0.05": "sigma=0.05",
    "gaussian_sigma1": "sigma=1.0",
    "dp_epsilon20": "epsilon=20",
}

DEFAULT_METHODS = ["no_protection", "fixed_ae_sigma1", "aniae"]

EVENT_FEATURES = [
    "fixation_rate",
    "saccade_rate",
    "scanpath_length",
    "dispersion",
    "velocity_mean",
    "velocity_q50",
    "velocity_q90",
    "velocity_q95",
    "acceleration_q90",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


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
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def bounded_score(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def mean_clean(values: list[float], default: float = float("nan")) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else default


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=1)) if clean.size > 1 else 0.0


def fmt_pm(mean: float, std: float, scale: float = 100.0, decimals: int = 2) -> str:
    if not math.isfinite(mean):
        return "n/a"
    return f"${mean * scale:.{decimals}f} \\pm {std * scale:.{decimals}f}$"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list))
                else row.get(key)
                for key in keys
            })


def checkpoint_path(template: str, seed: int) -> Path:
    return Path(template.format(seed=seed))


def load_aniae_transform(path: Path, config: dict, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Missing protection checkpoint: {path}")
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
    if method == "gaussian_sigma0.05":
        return lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, -1.0, 1.0), None
    if method == "gaussian_sigma1":
        return lambda x: torch.clamp(x + torch.randn_like(x), -1.0, 1.0), None
    if method == "dp_epsilon20":
        sensitivity = float(config.get("baselines", {}).get("dp_sensitivity", 2.0))
        scale = sensitivity / 20.0
        return lambda x: torch.clamp(
            x + torch.distributions.Laplace(0.0, scale).sample(x.shape).to(x.device),
            -1.0,
            1.0,
        ), None
    raise ValueError(f"Unknown method={method!r}")


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


class TemporalPredictor(nn.Module):
    """Small temporal convolutional predictor for future protected gaze."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 48, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        modules = []
        in_ch = input_dim
        for idx in range(layers):
            dilation = 2 ** idx
            modules.extend([
                nn.Conv1d(in_ch, hidden_dim, kernel_size=5, padding=2 * dilation, dilation=dilation),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = hidden_dim
        modules.append(nn.Conv1d(hidden_dim, input_dim, kernel_size=1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return out[:, : x.shape[1], :]


def build_loaders(config: dict, task_labels: list[str], seed: int, args):
    data_cfg = config["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), task_labels)
    if task_labels == TASK_SETS["non_ran"]:
        for split, records in splits.items():
            bad = [record for record in records if record["task"] == "RAN"]
            if bad:
                raise RuntimeError(f"RAN records leaked into non-RAN {split}")

    label_map = {
        person_id: idx
        for idx, person_id in enumerate(
            sorted({int(r["person_id"]) for records in splits.values() for r in records})
        )
    }
    datasets = {}
    loaders = {}
    for split, records in splits.items():
        datasets[split] = RoundProtocolChunkDataset(
            cache_dir,
            records,
            identity_field=data_cfg["identity_field"],
            label_map=label_map,
            tasks=task_labels,
            cache_lru_size=int(data_cfg.get("cache_lru_size", 64)),
        )
    batch_size = int(args.batch_size or config["task_training"]["batch_size"])
    for split, dataset in datasets.items():
        loaders[split] = make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=int(args.num_workers if split == "train" else args.val_num_workers),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
            seed=seed,
        )
    return datasets, loaders


def run_probe_epoch(
    model,
    loader,
    criterion,
    device,
    transform,
    num_tasks: int,
    optimizer=None,
    max_batches: int | None = None,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true = []
    y_pred = []
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc="TaskProbe", leave=False)):
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
        total_loss += loss.item() * task_labels.numel()
        y_true.extend(task_labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return {
        "loss": total_loss / max(1, len(y_true)),
        "accuracy": correct / max(1, len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(num_tasks)), average="macro", zero_division=0))
        if y_true else float("nan"),
        "n": len(y_true),
    }


def train_task_probe(config: dict, loaders: dict, transform, task_labels: list[str], seed: int, device: torch.device, args):
    set_global_seed(seed)
    model = make_probe_model(config, len(task_labels), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args.task_lr),
        weight_decay=float(config["task_training"].get("weight_decay", 0.0001)),
    )
    best_state = None
    best_val = -float("inf")
    patience_counter = 0
    for epoch in range(int(args.task_epochs)):
        train_metrics = run_probe_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            transform,
            len(task_labels),
            optimizer,
            args.max_task_train_batches,
        )
        val_metrics = run_probe_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            transform,
            len(task_labels),
            None,
            args.max_eval_batches,
        )
        score = val_metrics["macro_f1"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        print(
            f"task_probe seed={seed} epoch={epoch + 1}/{args.task_epochs} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if patience_counter >= int(args.task_patience):
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


@torch.no_grad()
def evaluate_probe_model(model, loader, device, transform, num_tasks: int, max_batches: int | None = None) -> dict:
    return run_probe_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        transform,
        num_tasks,
        optimizer=None,
        max_batches=max_batches,
    )


@torch.no_grad()
def evaluate_prediction_consistency(model, loader, device, protected_transform, max_batches: int | None = None) -> float:
    model.eval()
    total = 0
    agree = 0
    for batch_idx, (x, _, _, _) in enumerate(tqdm(loader, desc="Consistency", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        protected = protected_transform(x)
        raw_pred = model(x).argmax(1)
        protected_pred = model(protected).argmax(1)
        agree += (raw_pred == protected_pred).sum().item()
        total += x.size(0)
    return agree / max(1, total)


def temporal_batch_loss(model, x: torch.Tensor, horizon: int) -> torch.Tensor:
    if x.shape[1] <= horizon:
        raise ValueError(f"Sequence length {x.shape[1]} must exceed horizon {horizon}")
    history = x[:, :-horizon, :]
    target = x[:, horizon:, :]
    pred = model(history)
    return (pred - target).square().mean()


def train_temporal_predictor(config: dict, loaders: dict, horizon: int, seed: int, device: torch.device, args):
    set_global_seed(seed + horizon)
    model = TemporalPredictor(
        input_dim=len(config["data"]["features"]),
        hidden_dim=int(args.temporal_hidden_dim),
        layers=int(args.temporal_layers),
        dropout=float(args.temporal_dropout),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.temporal_lr), weight_decay=float(args.temporal_weight_decay))
    best_state = None
    best_val = float("inf")
    patience_counter = 0
    identity = lambda x: x
    for epoch in range(int(args.temporal_epochs)):
        model.train()
        train_losses = []
        for batch_idx, (x, _, _, _) in enumerate(tqdm(loaders["train"], desc="TemporalTrain", leave=False)):
            if args.max_temporal_train_batches is not None and batch_idx >= args.max_temporal_train_batches:
                break
            x = identity(x.to(device))
            loss = temporal_batch_loss(model, x, horizon)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics = evaluate_temporal_predictor(
            model,
            loaders["val"],
            device,
            identity,
            horizon,
            config["data"]["tasks"],
            args.max_eval_batches,
        )
        val_loss = float(val_metrics["overall_mse"])
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        print(
            f"temporal seed={seed} horizon={horizon} epoch={epoch + 1}/{args.temporal_epochs} "
            f"train_mse={mean_clean(train_losses, 0.0):.6f} val_mse={val_loss:.6f}"
        )
        if patience_counter >= int(args.temporal_patience):
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


@torch.no_grad()
def evaluate_temporal_predictor(
    model,
    loader,
    device,
    transform,
    horizon: int,
    task_labels: list[str],
    max_batches: int | None = None,
) -> dict:
    model.eval()
    by_task = defaultdict(list)
    all_losses = []
    for batch_idx, (x, _, labels, _) in enumerate(tqdm(loader, desc="TemporalEval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = transform(x.to(device))
        history = x[:, :-horizon, :]
        target = x[:, horizon:, :]
        pred = model(history)
        sample_mse = (pred - target).square().mean(dim=(1, 2)).detach().cpu().numpy()
        all_losses.extend(sample_mse.tolist())
        for label, loss in zip(labels.tolist(), sample_mse.tolist()):
            by_task[task_labels[int(label)]].append(float(loss))
    return {
        "overall_mse": mean_clean(all_losses),
        "by_task_mse": {task: mean_clean(values) for task, values in by_task.items()},
        "n": len(all_losses),
    }


def velocity_arrays(x: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    diff = np.diff(x, axis=1)
    velocity = np.linalg.norm(diff, axis=2) * sampling_rate
    acc = np.abs(np.diff(velocity, axis=1)) * sampling_rate if velocity.shape[1] > 1 else np.zeros_like(velocity)
    return velocity, acc


def batch_event_features(x: np.ndarray, sampling_rate: float, velocity_threshold: float) -> dict[str, np.ndarray]:
    velocity, acc = velocity_arrays(x, sampling_rate)
    saccade_mask = velocity > velocity_threshold
    scanpath = np.linalg.norm(np.diff(x, axis=1), axis=2).sum(axis=1)
    dispersion = np.std(x[:, :, 0], axis=1) + np.std(x[:, :, 1], axis=1)
    return {
        "fixation_rate": 1.0 - saccade_mask.mean(axis=1),
        "saccade_rate": saccade_mask.mean(axis=1),
        "scanpath_length": scanpath,
        "dispersion": dispersion,
        "velocity_mean": velocity.mean(axis=1),
        "velocity_q50": np.quantile(velocity, 0.50, axis=1),
        "velocity_q90": np.quantile(velocity, 0.90, axis=1),
        "velocity_q95": np.quantile(velocity, 0.95, axis=1),
        "acceleration_q90": np.quantile(acc, 0.90, axis=1) if acc.size else np.zeros(x.shape[0]),
    }


@torch.no_grad()
def estimate_event_threshold(loader, device, sampling_rate: float, quantile: float, max_batches: int | None) -> float:
    velocities = []
    for batch_idx, (x, _, _, _) in enumerate(tqdm(loader, desc="EventThreshold", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        arr = x.to(device).detach().cpu().numpy()
        vel, _ = velocity_arrays(arr, sampling_rate)
        velocities.append(vel.reshape(-1))
    if not velocities:
        return 0.0
    return float(np.quantile(np.concatenate(velocities), quantile))


@torch.no_grad()
def collect_event_feature_values(
    loader,
    device,
    transform,
    task_labels: list[str],
    sampling_rate: float,
    velocity_threshold: float,
    max_batches: int | None,
) -> dict[str, dict[str, list[float]]]:
    values = {
        task: {feature: [] for feature in EVENT_FEATURES}
        for task in task_labels
    }
    for batch_idx, (x, _, labels, _) in enumerate(tqdm(loader, desc="EventStats", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        protected = transform(x.to(device)).detach().cpu().numpy()
        feats = batch_event_features(protected, sampling_rate, velocity_threshold)
        for row_idx, label in enumerate(labels.tolist()):
            task = task_labels[int(label)]
            for feature in EVENT_FEATURES:
                value = float(feats[feature][row_idx])
                if math.isfinite(value):
                    values[task][feature].append(value)
    return values


def compare_event_values(raw_values: dict, protected_values: dict) -> dict:
    rows = []
    feature_scores = []
    for task, raw_by_feature in raw_values.items():
        for feature, raw_list in raw_by_feature.items():
            protected_list = protected_values.get(task, {}).get(feature, [])
            raw_arr = np.asarray(raw_list, dtype=float)
            prot_arr = np.asarray(protected_list, dtype=float)
            if raw_arr.size == 0 or prot_arr.size == 0:
                continue
            raw_mean = float(raw_arr.mean())
            prot_mean = float(prot_arr.mean())
            raw_iqr = float(np.subtract(*np.percentile(raw_arr, [75, 25])))
            norm = max(raw_iqr, float(raw_arr.std()), 1e-8)
            wdist = float(wasserstein_distance(raw_arr, prot_arr))
            norm_wdist = wdist / norm
            rel_err = abs(prot_mean - raw_mean) / max(abs(raw_mean), 1e-8)
            raw_constant = np.allclose(raw_arr, raw_arr[0])
            prot_constant = np.allclose(prot_arr, prot_arr[0])
            if raw_constant or prot_constant:
                rho = 1.0 if raw_constant and prot_constant and math.isclose(raw_mean, prot_mean, rel_tol=1e-6, abs_tol=1e-8) else 0.0
            else:
                min_len = min(raw_arr.size, prot_arr.size)
                rho = float(spearmanr(raw_arr[:min_len], prot_arr[:min_len]).correlation)
                if not math.isfinite(rho):
                    rho = 0.0
            w_score = math.exp(-norm_wdist)
            rel_score = 1.0 / (1.0 + rel_err)
            rank_score = (max(-1.0, min(1.0, rho)) + 1.0) / 2.0
            feature_score = float(np.mean([w_score, rel_score, rank_score]))
            feature_scores.append(feature_score)
            rows.append({
                "task": task,
                "feature": feature,
                "raw_mean": raw_mean,
                "protected_mean": prot_mean,
                "wasserstein": wdist,
                "normalized_wasserstein": norm_wdist,
                "relative_error": rel_err,
                "spearman": rho,
                "feature_score": feature_score,
            })
    return {
        "feature_rows": rows,
        "event_score": mean_clean(feature_scores, 0.0),
        "mean_normalized_wasserstein": mean_clean([r["normalized_wasserstein"] for r in rows], 0.0),
        "mean_relative_error": mean_clean([r["relative_error"] for r in rows], 0.0),
        "mean_spearman": mean_clean([r["spearman"] for r in rows], 0.0),
    }


def compute_cross_domain_metrics(
    config: dict,
    loaders: dict,
    task_labels: list[str],
    seed: int,
    device: torch.device,
    methods: list[str],
    transforms: dict[str, object],
    args,
) -> dict[str, dict]:
    identity = transforms["no_protection"]
    raw_model = train_task_probe(config, loaders, identity, task_labels, seed, device, args)
    raw_reference = evaluate_probe_model(raw_model, loaders["test"], device, identity, len(task_labels), args.max_eval_batches)
    raw_acc = max(raw_reference["accuracy"], 1e-8)
    raw_f1 = max(raw_reference["macro_f1"], 1e-8)

    out = {}
    for method in methods:
        transform = transforms[method]
        raw_to_protected = evaluate_probe_model(
            raw_model, loaders["test"], device, transform, len(task_labels), args.max_eval_batches
        )
        consistency = evaluate_prediction_consistency(raw_model, loaders["test"], device, transform, args.max_eval_batches)

        if method == "no_protection":
            out[method] = {
                "raw_reference": raw_reference,
                "raw_to_protected": raw_reference,
                "protected_to_protected": raw_reference,
                "protected_to_raw": raw_reference,
                "prediction_consistency": 1.0,
                "raw_to_protected_f1_retention": 1.0,
                "protected_to_protected_f1_retention": 1.0,
                "protected_to_raw_f1_retention": 1.0,
                "cross_domain_score": 1.0,
                "raw_reference_accuracy": raw_acc,
                "raw_reference_macro_f1": raw_f1,
            }
            continue

        protected_model = train_task_probe(config, loaders, transform, task_labels, seed + 1000, device, args)
        protected_to_protected = evaluate_probe_model(
            protected_model, loaders["test"], device, transform, len(task_labels), args.max_eval_batches
        )
        protected_to_raw = evaluate_probe_model(
            protected_model, loaders["test"], device, identity, len(task_labels), args.max_eval_batches
        )
        components = [
            raw_to_protected["macro_f1"] / raw_f1,
            protected_to_protected["macro_f1"] / raw_f1,
            protected_to_raw["macro_f1"] / raw_f1,
            consistency,
        ]
        out[method] = {
            "raw_reference": raw_reference,
            "raw_to_protected": raw_to_protected,
            "protected_to_protected": protected_to_protected,
            "protected_to_raw": protected_to_raw,
            "prediction_consistency": consistency,
            "raw_to_protected_f1_retention": raw_to_protected["macro_f1"] / raw_f1,
            "protected_to_protected_f1_retention": protected_to_protected["macro_f1"] / raw_f1,
            "protected_to_raw_f1_retention": protected_to_raw["macro_f1"] / raw_f1,
            "cross_domain_score": mean_clean([bounded_score(v) for v in components], 0.0),
            "raw_reference_accuracy": raw_acc,
            "raw_reference_macro_f1": raw_f1,
        }
    return out


def compute_temporal_metrics(config: dict, loaders: dict, task_labels: list[str], seed: int, device, methods, transforms, args):
    identity = transforms["no_protection"]
    out = {method: {"horizons": {}} for method in methods}
    temporal_scores = defaultdict(list)
    for horizon in args.temporal_horizons:
        model = train_temporal_predictor(config, loaders, int(horizon), seed, device, args)
        raw_metrics = evaluate_temporal_predictor(
            model, loaders["test"], device, identity, int(horizon), task_labels, args.max_eval_batches
        )
        raw_mse = max(float(raw_metrics["overall_mse"]), 1e-12)
        for method in methods:
            metrics = evaluate_temporal_predictor(
                model, loaders["test"], device, transforms[method], int(horizon), task_labels, args.max_eval_batches
            )
            protected_mse = max(float(metrics["overall_mse"]), 1e-12)
            retention = bounded_score(raw_mse / protected_mse)
            by_task_retention = {}
            for task in task_labels:
                task_raw = max(float(raw_metrics["by_task_mse"].get(task, raw_mse)), 1e-12)
                task_protected = max(float(metrics["by_task_mse"].get(task, protected_mse)), 1e-12)
                by_task_retention[task] = bounded_score(task_raw / task_protected)
            out[method]["horizons"][str(horizon)] = {
                "raw_mse": raw_mse,
                "protected_mse": protected_mse,
                "retention": retention,
                "raw_by_task_mse": raw_metrics["by_task_mse"],
                "protected_by_task_mse": metrics["by_task_mse"],
                "by_task_retention": by_task_retention,
            }
            temporal_scores[method].append(retention)
    for method in methods:
        out[method]["temporal_score"] = mean_clean(temporal_scores[method], 0.0)
    return out


def compute_event_metrics(config: dict, loaders: dict, task_labels: list[str], device, methods, transforms, args):
    sampling_rate = float(config["data"].get("sampling_rate_out", config["data"].get("sampling_rate", 250)))
    threshold = estimate_event_threshold(
        loaders["test"],
        device,
        sampling_rate,
        float(args.event_velocity_quantile),
        args.max_event_batches,
    )
    raw_values = collect_event_feature_values(
        loaders["test"],
        device,
        transforms["no_protection"],
        task_labels,
        sampling_rate,
        threshold,
        args.max_event_batches,
    )
    out = {}
    for method in methods:
        values = collect_event_feature_values(
            loaders["test"],
            device,
            transforms[method],
            task_labels,
            sampling_rate,
            threshold,
            args.max_event_batches,
        )
        comparison = compare_event_values(raw_values, values)
        comparison["velocity_threshold"] = threshold
        comparison["event_score"] = bounded_score(comparison["event_score"])
        out[method] = comparison
    return out


def build_seed_rows(config: dict, task_set: str, seed: int, device: torch.device, methods: list[str], args) -> list[dict]:
    task_labels = TASK_SETS[task_set]
    set_global_seed(seed)
    _, loaders = build_loaders(config, task_labels, seed, args)
    transforms = {}
    checkpoints = {}
    for method in methods:
        set_global_seed(seed)
        transform, checkpoint = make_transform(method, config, seed, device, args)
        transforms[method] = transform
        checkpoints[method] = checkpoint

    cross = compute_cross_domain_metrics(config, loaders, task_labels, seed, device, methods, transforms, args)
    temporal = compute_temporal_metrics(config, loaders, task_labels, seed, device, methods, transforms, args)
    event = compute_event_metrics(config, loaders, task_labels, device, methods, transforms, args)

    rows = []
    for method in methods:
        cross_score = bounded_score(cross[method]["cross_domain_score"])
        temporal_score = bounded_score(temporal[method]["temporal_score"])
        event_score = bounded_score(event[method]["event_score"])
        sri = mean_clean([cross_score, temporal_score, event_score], 0.0)
        rows.append({
            "seed": seed,
            "task_set": task_set,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "parameter": METHOD_PARAMETER[method],
            "checkpoint_path": checkpoints[method],
            "cross_domain_score": cross_score,
            "temporal_score": temporal_score,
            "event_score": event_score,
            "semantic_retention_index": sri,
            "cross_domain": cross[method],
            "temporal": temporal[method],
            "event": event[method],
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_set"], row["method"])].append(row)
    summary = []
    metric_keys = [
        "cross_domain_score",
        "temporal_score",
        "event_score",
        "semantic_retention_index",
    ]
    for (task_set, method), group in sorted(grouped.items(), key=lambda item: (item[0][0], DEFAULT_METHODS.index(item[0][1]) if item[0][1] in DEFAULT_METHODS else 99)):
        item = {
            "task_set": task_set,
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "parameter": METHOD_PARAMETER[method],
            "n_seeds": len({row["seed"] for row in group}),
        }
        for key in metric_keys:
            mean, std = mean_std([row[key] for row in group])
            item[f"{key}_mean"] = mean
            item[f"{key}_std"] = std
            item[f"{key}_mean_std"] = fmt_pm(mean, std)
        summary.append(item)
    return summary


def write_latex_table(path: Path, summary: list[dict], task_set: str) -> None:
    rows = [row for row in summary if row["task_set"] == task_set]
    if not rows:
        return
    lines = [
        "% Auto-generated by scripts/evaluate_semantic_retention_benchmark.py",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Task-conditioned semantic retention benchmark. Higher values indicate stronger retention. SRI is the unweighted mean of the cross-domain task-transfer, temporal-prediction, and oculomotor-event scores.}",
        "\\label{tab:semantic_retention}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Method & Parameter & Cross-task (\\%) & Temporal (\\%) & Event (\\%) & SRI (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method_display']} & {row['parameter']} & "
            f"{row['cross_domain_score_mean_std']} & "
            f"{row['temporal_score_mean_std']} & "
            f"{row['event_score_mean_std']} & "
            f"{row['semantic_retention_index_mean_std']} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(path: Path, summary: list[dict], task_set: str) -> None:
    rows = [row for row in summary if row["task_set"] == task_set]
    if not rows:
        return
    components = [
        ("cross_domain_score_mean", "Cross-task"),
        ("temporal_score_mean", "Temporal"),
        ("event_score_mean", "Event"),
        ("semantic_retention_index_mean", "SRI"),
    ]
    methods = [row["method_display"] for row in rows]
    x = np.arange(len(components))
    width = 0.23
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=220)
    for idx, row in enumerate(rows):
        values = [row[key] * 100.0 for key, _ in components]
        ax.bar(x + (idx - (len(rows) - 1) / 2) * width, values, width=width, label=methods[idx], color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in components])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Retention score (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_latex_figure(path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/evaluate_semantic_retention_benchmark.py",
        "\\begin{figure}[!t]",
        "    \\centering",
        "    \\includegraphics[width=\\linewidth]{figures/semantic_retention_benchmark.png}",
        "    \\caption{Task-conditioned semantic retention components. The benchmark separates task-transfer, temporal-prediction, and oculomotor-event retention before reporting the aggregate SRI.}",
        "    \\label{fig:semantic_retention}",
        "\\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_consistency_checks(rows: list[dict]) -> dict:
    checks = {}
    raw_rows = [row for row in rows if row["method"] == "no_protection"]
    if raw_rows:
        checks["no_protection_sri_min"] = min(row["semantic_retention_index"] for row in raw_rows)
        checks["no_protection_sri_close_to_one"] = checks["no_protection_sri_min"] >= 0.95
    checks["event_values_finite"] = True
    for row in rows:
        for feature_row in row["event"].get("feature_rows", []):
            for key in ["raw_mean", "protected_mean", "normalized_wasserstein", "relative_error", "spearman", "feature_score"]:
                if not math.isfinite(float(feature_row[key])):
                    checks["event_values_finite"] = False
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task-conditioned semantic retention")
    parser.add_argument("--config", type=str, default="configs/proposed_aniae_stage2.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--task_set", choices=["non_ran", "four_class"], default="non_ran")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/round_protocol_results/semantic_retention"))
    parser.add_argument("--fixed_ae_checkpoint_template", type=str, default="experiments/round_protocol_results/fixed_noise_ae_sigma1_seed{seed}/checkpoint_best.pt")
    parser.add_argument("--aniae_checkpoint_template", type=str, default="experiments/round_protocol_results/proposed_aniae_seed{seed}/checkpoint_best.pt")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_num_workers", type=int, default=0)
    parser.add_argument("--task_epochs", type=int, default=12)
    parser.add_argument("--task_patience", type=int, default=4)
    parser.add_argument("--task_lr", type=float, default=0.001)
    parser.add_argument("--temporal_epochs", type=int, default=8)
    parser.add_argument("--temporal_patience", type=int, default=3)
    parser.add_argument("--temporal_lr", type=float, default=0.001)
    parser.add_argument("--temporal_weight_decay", type=float, default=0.0001)
    parser.add_argument("--temporal_hidden_dim", type=int, default=48)
    parser.add_argument("--temporal_layers", type=int, default=3)
    parser.add_argument("--temporal_dropout", type=float, default=0.1)
    parser.add_argument("--temporal_horizons", nargs="+", type=int, default=[15, 50], help="Prediction horizons in 250Hz samples. 15=60ms, 50=200ms.")
    parser.add_argument("--event_velocity_quantile", type=float, default=0.75)
    parser.add_argument("--max_task_train_batches", type=int, default=None)
    parser.add_argument("--max_temporal_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--max_event_batches", type=int, default=None)
    parser.add_argument("--figure_out", type=Path, default=Path("results/semantic_retention/semantic_retention_benchmark.png"))
    parser.add_argument("--latex_table_out", type=Path, default=None, help="Optional private manuscript LaTeX table path.")
    parser.add_argument("--latex_figure_out", type=Path, default=None, help="Optional private manuscript LaTeX figure path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = list(args.methods)
    unknown = [method for method in methods if method not in METHOD_DISPLAY]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Task set: {args.task_set}")
    print(f"Methods: {methods}")
    print(f"Seeds: {args.seeds}")

    rows = []
    for seed in args.seeds:
        print(f"=== Semantic retention seed={seed} ===")
        rows.extend(build_seed_rows(config, args.task_set, int(seed), device, methods, args))

    summary = summarize(rows)
    checks = run_consistency_checks(rows)

    with open(args.output_dir / "semantic_retention_rows.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(rows), f, indent=2, ensure_ascii=False)
    with open(args.output_dir / "semantic_retention_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    with open(args.output_dir / "semantic_retention_checks.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(checks), f, indent=2, ensure_ascii=False)
    write_csv(args.output_dir / "semantic_retention_summary.csv", summary)

    plot_summary(args.figure_out, summary, args.task_set)
    if args.latex_table_out is not None:
        write_latex_table(args.latex_table_out, summary, args.task_set)
    if args.latex_figure_out is not None:
        write_latex_figure(args.latex_figure_out)

    print(f"Rows written to {args.output_dir / 'semantic_retention_rows.json'}")
    print(f"Summary written to {args.output_dir / 'semantic_retention_summary.json'}")
    print(f"Figure written to {args.figure_out}")
    print(f"Checks: {json.dumps(json_safe(checks), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
