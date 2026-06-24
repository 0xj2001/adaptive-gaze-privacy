"""Train a frozen task utility evaluator for FXS/HSS/RAN/TEX retention."""

import argparse
import json
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.round_protocol import (
    RoundProtocolChunkDataset,
    load_manifest,
    make_loader,
    records_by_split,
)
from src.models.biometric import GazeTaskEvaluator


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


def make_model(config: dict, device: torch.device) -> GazeTaskEvaluator:
    data_cfg = config["data"]
    task_cfg = config.get("task_model", {})
    bio_cfg = config.get("biometric_model", {})
    return GazeTaskEvaluator(
        input_dim=len(data_cfg["features"]),
        num_tasks=len(data_cfg["tasks"]),
        hidden_dim=int(task_cfg.get("hidden_dim", 128)),
        base_channels=int(bio_cfg.get("base_channels", 64)),
        growth_rate=int(bio_cfg.get("growth_rate", 16)),
        block_layers=task_cfg.get("block_layers", [3, 3, 3]),
        dropout=float(task_cfg.get("dropout", 0.2)),
    ).to(device)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    desc = "Train" if is_train else "Eval"
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        task_labels = task_labels.to(device)
        logits = model(x)
        loss = criterion(logits, task_labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += loss.item() * task_labels.size(0)
        total_correct += (logits.argmax(1) == task_labels).sum().item()
        total += task_labels.size(0)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": total_correct / max(1, total),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_score: float, config: dict, metrics: dict):
    torch.save({
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train task utility evaluator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"task_evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_dir or config["logging"]["output_dir"])
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    data_cfg = config["data"]
    train_cfg = config["task_training"]
    cache_dir = Path(data_cfg["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), data_cfg["tasks"])

    label_map = {
        person_id: idx
        for idx, person_id in enumerate(sorted({int(r["person_id"]) for records in splits.values() for r in records}))
    }
    datasets = {
        split: RoundProtocolChunkDataset(
            cache_dir,
            records,
            identity_field=data_cfg["identity_field"],
            label_map=label_map,
            tasks=data_cfg["tasks"],
            cache_lru_size=int(data_cfg.get("cache_lru_size", 64)),
        )
        for split, records in splits.items()
    }
    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        ),
        "val": make_loader(
            datasets["val"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("val_num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        ),
        "test": make_loader(
            datasets["test"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("val_num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        ),
    }

    model = make_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    print(f"Config loaded: {args.config}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"Train chunks: {len(datasets['train'])} | Val chunks: {len(datasets['val'])} | Test chunks: {len(datasets['test'])}")

    best_score = -float("inf")
    patience = int(train_cfg["patience"])
    patience_counter = 0
    epochs = int(train_cfg["epochs"])
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(epochs):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer, args.max_train_batches
        )
        val_metrics = run_epoch(
            model, loaders["val"], criterion, device, None, args.max_eval_batches
        )
        score = val_metrics["accuracy"]
        record = {
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "train": train_metrics,
            "val": {f"val_{k}": v for k, v in val_metrics.items()},
            "best_score": best_score,
            "patience_counter": patience_counter,
        }
        if score > best_score:
            best_score = score
            patience_counter = 0
            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, epoch, best_score, config, record)
        else:
            patience_counter += 1
        record["best_score"] = best_score
        record["patience_counter"] = patience_counter
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, epoch, best_score, config, record)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss={train_metrics['loss']:.4f} | "
            f"Acc={train_metrics['accuracy']*100:.2f}% | "
            f"ValAcc={val_metrics['accuracy']*100:.2f}%"
        )
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    ckpt = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device, None, args.max_eval_batches)
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(test_metrics), f, indent=2, ensure_ascii=False)

    print("\n=== Test Results ===")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
