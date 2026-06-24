"""Evaluate protection methods with frozen biometric and task evaluators."""

import argparse
import json
import multiprocessing
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.round_protocol import (
    RoundProtocolChunkDataset,
    load_manifest,
    make_loader,
    records_by_split,
)
from src.evaluation.biometric_metrics import enrollment_probe_details
from src.models.autoencoder import ANIAE
from src.models.biometric import GazeTaskEvaluator, build_biometric_model


def set_global_seed(seed: int) -> None:
    """Seed all RNGs used by stochastic protection transforms."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


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


def make_biometric_model(config: dict, num_subjects: int, device: torch.device):
    return build_biometric_model(config, num_subjects).to(device)


def make_task_model(config: dict, device: torch.device):
    task_cfg = config.get("task_model", {})
    bio_cfg = config.get("biometric_model", {})
    return GazeTaskEvaluator(
        input_dim=len(config["data"]["features"]),
        num_tasks=len(config["data"]["tasks"]),
        hidden_dim=int(task_cfg.get("hidden_dim", 128)),
        base_channels=int(bio_cfg.get("base_channels", 64)),
        growth_rate=int(bio_cfg.get("growth_rate", 16)),
        block_layers=task_cfg.get("block_layers", [3, 3, 3]),
        dropout=float(task_cfg.get("dropout", 0.2)),
    ).to(device)


def load_biometric_checkpoint(config: dict, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_subjects = len(ckpt.get("label_map", {})) or ckpt["model_state"]["classifier.weight"].shape[0]
    model_config = ckpt.get("config", config)
    model = make_biometric_model(model_config, num_subjects, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_task_checkpoint(config: dict, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = make_task_model(config, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def protect_with_aniae(aniae: ANIAE, x: torch.Tensor, subwindow_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_features = x.shape
    if seq_len % subwindow_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by subwindow_size={subwindow_size}")
    n_sub = seq_len // subwindow_size
    sub = x.reshape(bsz, n_sub, subwindow_size, n_features).reshape(bsz * n_sub, subwindow_size, n_features)
    out = aniae(sub)
    x_hat = out["x_hat"].reshape(bsz, n_sub, subwindow_size, n_features).reshape(bsz, seq_len, n_features)
    return x_hat, out["sigma"]


LEARNED_AE_METHODS = {"fixed_ae", "aniae", "vae_ae", "grl_ae"}


def make_transform(
    method: str,
    parameter,
    config: dict,
    device: torch.device,
    ae_eval_noise_mode: str = "deterministic",
):
    if method == "no_protection":
        return lambda x: x
    if method == "gaussian":
        sigma = float(parameter)
        return lambda x: torch.clamp(x + torch.randn_like(x) * sigma, -1.0, 1.0)
    if method == "dp":
        epsilon = float(parameter)
        scale = float(config["baselines"].get("dp_sensitivity", 2.0)) / epsilon
        return lambda x: torch.clamp(x + torch.distributions.Laplace(0.0, scale).sample(x.shape).to(x.device), -1.0, 1.0)
    if method in LEARNED_AE_METHODS:
        ckpt_path = Path(str(parameter))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing AE checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ae_config = ckpt.get("config", config)
        aniae = ANIAE(ae_config).to(device)
        state = ckpt.get("aniae_state", ckpt.get("model_state"))
        aniae.load_state_dict(state)
        if hasattr(aniae, "set_eval_noise_mode"):
            aniae.set_eval_noise_mode(ae_eval_noise_mode)
        aniae.eval()
        subwindow_size = int(config["data"]["subwindow_size"])

        def transform(x):
            x_hat, _ = protect_with_aniae(aniae, x, subwindow_size)
            return torch.clamp(x_hat, -1.0, 1.0)

        return transform
    raise ValueError(f"Unknown method={method}")


@torch.no_grad()
def evaluate_task_and_reconstruction(task_model, loader, device, transform=None, max_batches=None) -> dict:
    task_model.eval()
    total = 0
    correct = 0
    total_mse = 0.0
    total_mae = 0.0
    correlations = []
    y_true = []
    y_pred = []
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc="TaskEval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        task_labels = task_labels.to(device)
        protected = transform(x) if transform is not None else x
        logits = task_model(protected)
        pred = logits.argmax(1)
        correct += (pred == task_labels).sum().item()
        total += task_labels.numel()
        y_true.extend(task_labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        diff = protected - x
        total_mse += diff.square().mean(dim=(1, 2)).sum().item()
        total_mae += diff.abs().mean(dim=(1, 2)).sum().item()
        x_flat = x.flatten(start_dim=1)
        p_flat = protected.flatten(start_dim=1)
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        p_centered = p_flat - p_flat.mean(dim=1, keepdim=True)
        corr = (x_centered * p_centered).sum(dim=1) / (
            x_centered.norm(dim=1) * p_centered.norm(dim=1)
        ).clamp_min(1e-8)
        correlations.extend(corr.detach().cpu().tolist())
    return {
        "task_accuracy": correct / max(1, total),
        "task_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else float("nan"),
        "task_confusion_matrix": confusion_matrix(y_true, y_pred).tolist() if y_true else [],
        "reconstruction_mse": total_mse / max(1, total),
        "reconstruction_mae": total_mae / max(1, total),
        "signal_correlation": sum(correlations) / max(1, len(correlations)),
    }


@torch.no_grad()
def evaluate_biometric_publication(
    biometric,
    cache_dir,
    records,
    device,
    transform=None,
    include_score_details: bool = False,
    **eval_kwargs,
) -> dict:
    details = enrollment_probe_details(
        biometric,
        cache_dir,
        records,
        device,
        transform=transform,
        **eval_kwargs,
    )
    scores = details["scores"]
    labels = details["labels"]
    metrics = dict(details["metrics"])
    if len(set(labels.tolist())) < 2:
        metrics.update({"auc": float("nan"), "tar_at_far_1pct": float("nan"), "tar_at_far_5pct": float("nan")})
        return metrics
    fpr, tpr, thresholds = roc_curve(labels, scores)
    metrics["auc"] = float(roc_auc_score(labels, scores))
    for far, key in [(0.01, "tar_at_far_1pct"), (0.05, "tar_at_far_5pct")]:
        eligible = tpr[fpr <= far]
        metrics[key] = float(eligible.max()) if eligible.size else 0.0
    if include_score_details:
        metrics["biometric_scores"] = scores.tolist()
        metrics["biometric_labels"] = labels.tolist()
        metrics["roc_fpr"] = fpr.tolist()
        metrics["roc_tpr"] = tpr.tolist()
        metrics["roc_thresholds"] = thresholds.tolist()
    return metrics


def planned_methods(
    config: dict,
    methods: list[str],
    results_dir: Path,
    aniae_checkpoint: str | None = None,
    aniae_label: str | None = None,
    fixed_ae_checkpoint_template: str | None = None,
    vae_checkpoint: str | None = None,
    vae_label: str | None = None,
    vae_checkpoint_template: str | None = None,
    grl_checkpoint: str | None = None,
    grl_label: str | None = None,
    grl_checkpoint_template: str | None = None,
    seed: int | None = None,
) -> list[tuple[str, str, object]]:
    if "all" in methods:
        methods = ["no_protection", "gaussian", "dp", "fixed_ae", "vae_ae", "grl_ae", "aniae"]
    planned = []
    if "no_protection" in methods:
        planned.append(("no_protection", "none", None))
    if "gaussian" in methods:
        for sigma in config["baselines"]["gaussian_sigmas"]:
            planned.append(("gaussian", f"sigma={sigma}", sigma))
    if "dp" in methods:
        for epsilon in config["baselines"]["dp_epsilons"]:
            planned.append(("dp", f"epsilon={epsilon}", epsilon))
    if "fixed_ae" in methods:
        excluded = {
            round(float(sigma), 12)
            for sigma in config["baselines"].get("excluded_fixed_ae_sigmas", [])
        }
        for sigma in config["baselines"]["fixed_ae_sigmas"]:
            if round(float(sigma), 12) in excluded:
                continue
            if fixed_ae_checkpoint_template:
                sigma_float = float(sigma)
                sigma_label = str(sigma)
                sigma_compact = f"{sigma_float:g}"
                path = Path(
                    fixed_ae_checkpoint_template.format(
                        sigma=sigma_label,
                        sigma_g=sigma_compact,
                        seed=seed,
                    )
                )
                if not path.exists():
                    compact_path = Path(
                        fixed_ae_checkpoint_template.format(
                            sigma=sigma_compact,
                            sigma_g=sigma_compact,
                            seed=seed,
                        )
                    )
                    if compact_path.exists():
                        path = compact_path
            else:
                path = results_dir / f"fixed_sigma_{sigma}" / "checkpoint_best.pt"
            planned.append(("fixed_ae", f"sigma={sigma}", path))
    if "vae_ae" in methods:
        if vae_checkpoint_template:
            path = Path(vae_checkpoint_template.format(seed=seed))
        elif vae_checkpoint:
            path = Path(vae_checkpoint)
        else:
            path = results_dir / "vae_ae" / "checkpoint_best.pt"
        planned.append(("vae_ae", vae_label or "VAE/KL", path))
    if "grl_ae" in methods:
        if grl_checkpoint_template:
            path = Path(grl_checkpoint_template.format(seed=seed))
        elif grl_checkpoint:
            path = Path(grl_checkpoint)
        else:
            path = results_dir / "grl_ae" / "checkpoint_best.pt"
        planned.append(("grl_ae", grl_label or "adversarial", path))
    if "aniae" in methods:
        path = Path(aniae_checkpoint) if aniae_checkpoint else results_dir / "aniae_adaptive" / "checkpoint_best.pt"
        planned.append(("aniae", aniae_label or "adaptive", path))
    return planned


def sigma_stats_for_method(method: str, parameter_label: str, parameter) -> tuple[float, float]:
    if method == "fixed_ae":
        try:
            return float(parameter_label.split("=", 1)[1]), 0.0
        except (IndexError, ValueError):
            return float("nan"), float("nan")
    if method == "aniae" and Path(parameter).exists():
        ckpt = torch.load(parameter, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {})
        val = metrics.get("val", {})
        return (
            float(val.get("val_sigma_mean", float("nan"))),
            float(val.get("val_sigma_std", float("nan"))),
        )
    return float("nan"), float("nan")


def average_metric_dicts(records: list[dict]) -> dict:
    if not records:
        return {}
    averaged = {}
    keys = records[0].keys()
    for key in keys:
        values = [record.get(key) for record in records]
        if all(isinstance(value, (int, float)) for value in values):
            averaged[key] = sum(float(value) for value in values) / len(values)
        else:
            averaged[key] = values[0]
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Evaluate protection methods")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--biometric_checkpoint", type=str, default=None)
    parser.add_argument("--task_checkpoint", type=str, default=None)
    parser.add_argument("--aniae_checkpoint", type=str, default=None)
    parser.add_argument("--aniae_label", type=str, default=None)
    parser.add_argument("--fixed_ae_checkpoint_template", type=str, default=None)
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--vae_label", type=str, default=None)
    parser.add_argument("--vae_checkpoint_template", type=str, default=None)
    parser.add_argument("--grl_checkpoint", type=str, default=None)
    parser.add_argument("--grl_label", type=str, default=None)
    parser.add_argument("--grl_checkpoint_template", type=str, default=None)
    parser.add_argument(
        "--ae_eval_noise_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "sample", "none"],
        help="Evaluation release mode for learned AE checkpoints.",
    )
    parser.add_argument(
        "--release_repeats",
        type=int,
        default=1,
        help="Number of stochastic releases to average for each method.",
    )
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include_score_details", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    config.setdefault("project", {})["seed"] = seed
    set_global_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(config["data"]["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), config["data"]["tasks"])
    eval_records = splits[args.split]
    label_map = {
        person_id: idx
        for idx, person_id in enumerate(sorted({int(r["person_id"]) for r in eval_records}))
    }
    dataset = RoundProtocolChunkDataset(
        cache_dir,
        eval_records,
        identity_field=config["data"]["identity_field"],
        label_map=label_map,
        tasks=config["data"]["tasks"],
        cache_lru_size=int(config["data"].get("cache_lru_size", 64)),
    )
    loader = make_loader(
        dataset,
        batch_size=int(config["task_training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("val_num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", torch.cuda.is_available())),
        seed=seed,
    )

    biometric = load_biometric_checkpoint(
        config,
        args.biometric_checkpoint or config["paths"]["biometric_evaluator"],
        device,
    )
    task_model = load_task_checkpoint(
        config,
        args.task_checkpoint or config["paths"]["task_evaluator"],
        device,
    )
    protocol = config["data"].get("round_protocol", {})
    eval_kwargs = dict(
        task=protocol.get("enrollment_task", "RAN"),
        enrollment_session=int(protocol.get("enrollment_session", 1)),
        probe_session=int(protocol.get("probe_session", 2)),
        batch_size=int(config["biometric_training"].get("eval_batch_size", 128)),
    )
    results_dir = Path(config["paths"].get("results_dir", "./experiments/round_protocol_results"))

    raw_transform = make_transform("no_protection", None, config, device)
    raw_task = evaluate_task_and_reconstruction(task_model, loader, device, raw_transform, args.max_eval_batches)
    raw_bio = evaluate_biometric_publication(biometric, cache_dir, eval_records, device, transform=raw_transform, **eval_kwargs)
    raw_rank1 = raw_bio["rank1_ir"]
    raw_task_acc = raw_task["task_accuracy"]

    rows = []
    for method, parameter_label, parameter in planned_methods(
        config,
        args.methods,
        results_dir,
        args.aniae_checkpoint,
        args.aniae_label,
        args.fixed_ae_checkpoint_template,
        args.vae_checkpoint,
        args.vae_label,
        args.vae_checkpoint_template,
        args.grl_checkpoint,
        args.grl_label,
        args.grl_checkpoint_template,
        seed,
    ):
        if method in LEARNED_AE_METHODS and not Path(parameter).exists():
            print(f"Skipping {method} {parameter_label}: missing {parameter}")
            continue
        set_global_seed(seed)
        transform = make_transform(method, parameter, config, device, args.ae_eval_noise_mode)
        repeat_count = max(1, int(args.release_repeats))
        bio_records = []
        task_records = []
        for repeat_idx in range(repeat_count):
            repeat_seed = seed + repeat_idx * 1009
            set_global_seed(repeat_seed)
            bio_records.append(
                evaluate_biometric_publication(
                    biometric,
                    cache_dir,
                    eval_records,
                    device,
                    transform=transform,
                    include_score_details=args.include_score_details and repeat_idx == 0,
                    **eval_kwargs,
                )
            )
            set_global_seed(repeat_seed + 500000)
            task_records.append(
                evaluate_task_and_reconstruction(task_model, loader, device, transform, args.max_eval_batches)
            )
        bio = average_metric_dicts(bio_records)
        task = average_metric_dicts(task_records)
        sigma_mean, sigma_std = sigma_stats_for_method(method, parameter_label, parameter)
        row = {
            "method": method,
            "parameter": parameter_label,
            "seed": seed,
            "ae_eval_noise_mode": args.ae_eval_noise_mode,
            "release_repeats": repeat_count,
            **bio,
            **task,
            "privacy_gain_rank1": 1.0 - bio["rank1_ir"] / max(raw_rank1, 1e-8),
            "task_retention": task["task_accuracy"] / max(raw_task_acc, 1e-8),
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std,
        }
        rows.append(row)
        compact_row = {
            key: value for key, value in row.items()
            if key not in {
                "biometric_scores",
                "biometric_labels",
                "roc_fpr",
                "roc_tpr",
                "roc_thresholds",
                "task_confusion_matrix",
            }
        }
        print(json.dumps(json_safe(compact_row), ensure_ascii=False))

    output_path = Path(args.output or results_dir / f"{args.split}_protection_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(rows), f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(exit_code)
