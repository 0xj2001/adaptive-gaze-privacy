"""
Train an EKYT-style biometric embedding evaluator.

The model is trained on train split chunks with metric learning, then evaluated
with enrollment/probe matching on validation and test splits.
"""

import argparse
import json
import multiprocessing
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.round_protocol import (
    PKBatchSampler,
    RoundProtocolChunkDataset,
    load_manifest,
    records_by_split,
)
from src.evaluation.biometric_metrics import evaluate_enrollment_probe
from src.models.biometric import build_biometric_model
from src.training.metric_losses import MultiSimilarityLoss


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


def add_val_score(metrics: dict) -> dict:
    """Add a combined validation score for checkpoint selection."""
    scored = dict(metrics)
    scored["val_score"] = (
        float(metrics["rank1_ir"])
        + float(metrics["rank5_ir"])
        - float(metrics["eer"])
    )
    return scored


def prefix_val_metrics(metrics: dict) -> dict:
    prefixed = {}
    for key, value in metrics.items():
        out_key = key if key.startswith("val_") else f"val_{key}"
        prefixed[out_key] = value
    return prefixed


def metric_for_monitor(metrics: dict, monitor: str) -> float:
    """Return the metric named by config, accepting either raw or val_ keys."""
    if not monitor:
        monitor = "val_rank1_ir"
    candidates = [monitor]
    if monitor.startswith("val_"):
        candidates.append(monitor[4:])
    else:
        candidates.append(f"val_{monitor}")

    for key in candidates:
        if key in metrics:
            return float(metrics[key])

    available = ", ".join(sorted(metrics.keys()))
    raise KeyError(f"Unknown biometric monitor '{monitor}'. Available metrics: {available}")


def make_model(config: dict, num_subjects: int, device: torch.device):
    return build_biometric_model(config, num_subjects).to(device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    metric_loss,
    ce_loss,
    device,
    ce_weight: float,
    max_batches: int | None = None,
) -> dict:
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    total_ce = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch_idx, (x, labels, _, _) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        labels = labels.to(device)
        embeddings, logits = model(x)
        loss_metric = metric_loss(embeddings, labels)
        loss_ce = ce_loss(logits, labels)
        loss = loss_metric + ce_weight * loss_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_metric += loss_metric.item()
        total_ce += loss_ce.item()
        total_acc += (logits.argmax(1) == labels).float().mean().item()
        n_batches += 1

    n = max(1, n_batches)
    return {
        "loss": total_loss / n,
        "metric_loss": total_metric / n,
        "ce_loss": total_ce / n,
        "train_class_acc": total_acc / n,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_score: float, config: dict, metrics: dict, label_map: dict):
    torch.save({
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
        "label_map": label_map,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train EKYT-style biometric evaluator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_subjects", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    config.setdefault("project", {})["seed"] = seed
    set_global_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"biometric_evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_dir or config["logging"]["output_dir"])
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    data_cfg = config["data"]
    train_cfg = config["biometric_training"]
    cache_dir = Path(data_cfg["cache_dir"])
    manifest = load_manifest(cache_dir)
    splits = records_by_split(manifest, data_cfg["tasks"])

    train_label_map = {
        person_id: idx
        for idx, person_id in enumerate(sorted({int(r["person_id"]) for r in splits["train"]}))
    }
    train_ds = RoundProtocolChunkDataset(
        cache_dir,
        splits["train"],
        identity_field=data_cfg["identity_field"],
        label_map=train_label_map,
        tasks=data_cfg["tasks"],
        cache_lru_size=int(data_cfg.get("cache_lru_size", 64)),
    )
    sampler = PKBatchSampler(
        train_ds,
        p=int(train_cfg.get("pk_p", 32)),
        k=int(train_cfg.get("pk_k", 4)),
        steps_per_epoch=int(train_cfg.get("steps_per_epoch", 300)),
        seed=seed,
    )
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0))
    use_persistent_workers = num_workers > 0 and args.max_train_batches is None
    worker_kwargs = {"persistent_workers": True} if use_persistent_workers else {}
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        **worker_kwargs,
    )

    model = make_model(config, len(train_label_map), device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    metric_loss = MultiSimilarityLoss(
        alpha=float(train_cfg.get("ms_alpha", 2.0)),
        beta=float(train_cfg.get("ms_beta", 50.0)),
        base=float(train_cfg.get("ms_base", 0.5)),
    )
    ce_loss = nn.CrossEntropyLoss()

    print(f"Config loaded: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output dir: {output_dir}")
    print(f"Train subjects: {len(train_label_map)} | chunks: {len(train_ds)}")
    print(f"Val subjects: {len({r['person_id'] for r in splits['val']})}")
    print(f"Test subjects: {len({r['person_id'] for r in splits['test']})}")

    best_score = -float("inf")
    patience = int(train_cfg["patience"])
    patience_counter = 0
    monitor = str(train_cfg.get("monitor", "val_rank1_ir"))
    epochs = int(train_cfg["epochs"])
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    metrics_path = output_dir / "metrics.jsonl"

    eval_kwargs = dict(
        task=data_cfg.get("round_protocol", {}).get("enrollment_task", "RAN"),
        enrollment_session=int(data_cfg.get("round_protocol", {}).get("enrollment_session", 1)),
        probe_session=int(data_cfg.get("round_protocol", {}).get("probe_session", 2)),
        batch_size=int(train_cfg.get("eval_batch_size", 128)),
    )

    val_records = splits["val"]
    test_records = splits["test"]
    if args.max_eval_subjects is not None:
        keep_val = set(sorted({int(r["person_id"]) for r in val_records})[:args.max_eval_subjects])
        keep_test = set(sorted({int(r["person_id"]) for r in test_records})[:args.max_eval_subjects])
        val_records = [r for r in val_records if int(r["person_id"]) in keep_val]
        test_records = [r for r in test_records if int(r["person_id"]) in keep_test]

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            metric_loss,
            ce_loss,
            device,
            ce_weight=float(train_cfg.get("ce_weight", 0.2)),
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate_enrollment_probe(
            model, cache_dir, val_records, device, **eval_kwargs
        )
        val_metrics = add_val_score(val_metrics)
        val_metrics_prefixed = prefix_val_metrics(val_metrics)
        score = metric_for_monitor(val_metrics, monitor)
        record = {
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "train": train_metrics,
            "val": val_metrics_prefixed,
            "monitor": monitor,
            "monitor_score": score,
            "best_score": best_score,
            "patience_counter": patience_counter,
        }
        if score > best_score:
            best_score = score
            patience_counter = 0
            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, epoch, best_score, config, record, train_label_map)
        else:
            patience_counter += 1
        record["best_score"] = best_score
        record["patience_counter"] = patience_counter
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, epoch, best_score, config, record, train_label_map)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss={train_metrics['loss']:.4f} | "
            f"ClsAcc={train_metrics['train_class_acc']*100:.2f}% | "
            f"ValRank1={val_metrics['rank1_ir']*100:.2f}% | "
            f"ValEER={val_metrics['eer']*100:.2f}% | "
            f"{monitor}={score:.4f}"
        )
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    best_path = output_dir / "checkpoint_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_enrollment_probe(model, cache_dir, test_records, device, **eval_kwargs)
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(test_metrics), f, indent=2, ensure_ascii=False)
    with open(output_dir / "training_complete.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "seed": seed,
                "best_score": best_score,
                "checkpoint_best": str(output_dir / "checkpoint_best.pt"),
                "checkpoint_latest": str(output_dir / "checkpoint_latest.pt"),
                "test_results": test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("\n=== Test Results ===")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
