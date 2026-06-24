"""
Train a raw gaze identity baseline.

This baseline answers a prerequisite question for privacy experiments:
can identity be predicted from the original gaze window before ANIAE?
"""

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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.gazebase_loader import create_dataloaders
from src.models.attacker import PrivacyAttacker


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
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


def make_model(config: dict, num_subjects: int, device: torch.device) -> PrivacyAttacker:
    model_cfg = config.get("baseline_model", {})
    return PrivacyAttacker(
        input_dim=len(config["data"]["features"]),
        num_subjects=num_subjects,
        hidden_dims=model_cfg.get("hidden_dims", [256, 128]),
        dropout=float(model_cfg.get("dropout", 0.3)),
        input_type=model_cfg.get("input_type", "sequence"),
    ).to(device)


def compute_batch_topk(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)) -> dict[int, int]:
    max_k = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    counts = {}
    for k in ks:
        k_eff = min(k, logits.shape[1])
        counts[k] = correct[:k_eff].reshape(-1).float().sum().item()
    return counts


def _recording_key(sample: dict) -> str:
    return str(sample.get("file", sample.get("cache_file", "")))


def _dataset_samples_for_batch(dataset, offset: int, batch_size: int) -> list[dict]:
    if hasattr(dataset, "get_samples_for_range"):
        return dataset.get_samples_for_range(offset, batch_size)
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return []
    return samples[offset:offset + batch_size]


def _topk_from_probs(probs: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)) -> dict[int, float]:
    counts = compute_batch_topk(probs, labels, ks)
    total = max(1, labels.numel())
    return {k: counts[k] / total for k in ks}


def _eer_from_recording_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    scores = []
    targets = []
    for row, label in zip(probs, labels):
        label_idx = int(label.item())
        scores.append(float(row[label_idx].item()))
        targets.append(1)
        for class_idx in range(row.numel()):
            if class_idx == label_idx:
                continue
            scores.append(float(row[class_idx].item()))
            targets.append(0)
    if len(set(targets)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(targets, scores)
    fnr = 1 - tpr
    idx = abs(fpr - fnr).argmin()
    return float((fpr[idx] + fnr[idx]) / 2)


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None) -> dict:
    model.train()
    total_loss = 0.0
    total = 0
    top_counts = {1: 0.0, 5: 0.0, 10: 0.0}

    for batch_idx, (x, labels, _) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        labels = labels.to(device)
        logits = model(x)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = labels.size(0)
        counts = compute_batch_topk(logits.detach(), labels)
        for k, value in counts.items():
            top_counts[k] += value
        total_loss += loss.item() * batch_size
        total += batch_size

    return {
        "loss": total_loss / max(1, total),
        "top1_accuracy": top_counts[1] / max(1, total),
        "top5_accuracy": top_counts[5] / max(1, total),
        "top10_accuracy": top_counts[10] / max(1, total),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, max_batches=None) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    top_counts = {1: 0.0, 5: 0.0, 10: 0.0}
    all_preds = []
    all_labels = []
    recording_probs: dict[str, torch.Tensor] = {}
    recording_labels: dict[str, int] = {}
    sample_offset = 0

    for batch_idx, (x, labels, _) in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_samples = _dataset_samples_for_batch(loader.dataset, sample_offset, labels.size(0))
        sample_offset += labels.size(0)

        x = x.to(device)
        labels = labels.to(device)
        logits = model(x)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1).detach().cpu()

        batch_size = labels.size(0)
        counts = compute_batch_topk(logits, labels)
        for k, value in counts.items():
            top_counts[k] += value
        total_loss += loss.item() * batch_size
        total += batch_size

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if batch_samples:
            for sample, prob, label in zip(batch_samples, probs, labels.detach().cpu()):
                key = _recording_key(sample)
                if key not in recording_probs:
                    recording_probs[key] = prob.clone()
                    recording_labels[key] = int(label.item())
                else:
                    recording_probs[key] += prob

    if total == 0:
        balanced = float("nan")
    else:
        balanced = balanced_accuracy_score(all_labels, all_preds)

    if recording_probs:
        rec_keys = sorted(recording_probs)
        rec_probs = torch.stack([recording_probs[key] for key in rec_keys], dim=0)
        rec_probs = rec_probs / rec_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        rec_labels = torch.tensor([recording_labels[key] for key in rec_keys], dtype=torch.long)
        rec_topk = _topk_from_probs(rec_probs, rec_labels)
        rec_eer = _eer_from_recording_probs(rec_probs, rec_labels)
    else:
        rec_topk = {1: float("nan"), 5: float("nan"), 10: float("nan")}
        rec_eer = float("nan")

    window_top1 = top_counts[1] / max(1, total)
    window_top5 = top_counts[5] / max(1, total)
    window_top10 = top_counts[10] / max(1, total)
    return {
        "loss": total_loss / max(1, total),
        "top1_accuracy": window_top1,
        "top5_accuracy": window_top5,
        "top10_accuracy": window_top10,
        "window_top1": window_top1,
        "window_top5": window_top5,
        "window_top10": window_top10,
        "recording_top1": rec_topk[1],
        "recording_top5": rec_topk[5],
        "recording_top10": rec_topk[10],
        "eer": rec_eer,
        "balanced_accuracy": balanced,
        "random_top1": 1.0 / num_classes,
        "random_top5": min(5, num_classes) / num_classes,
        "random_top10": min(10, num_classes) / num_classes,
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_score: float, config: dict, metrics: dict):
    torch.save({
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train raw gaze identity baseline")
    parser.add_argument("--config", type=str, default="configs/raw_identity_1s.yaml")
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

    run_name = args.run_name or f"raw_identity_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_dir or config["logging"]["output_dir"])
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"Config loaded: {args.config}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    num_subjects = len(train_loader.dataset.subject_to_label)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    print(f"  Subjects: {num_subjects}")
    print(f"  Random top-1: {100 / num_subjects:.3f}%")
    print(f"  Random top-10: {1000 / num_subjects:.3f}%")

    model = make_model(config, num_subjects, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))

    best_score = -float("inf")
    patience_counter = 0
    monitor = config["training"].get("monitor", "val_top1_accuracy")
    patience = int(config["training"]["patience"])
    metrics_path = output_dir / "metrics.jsonl"

    epochs = int(config["training"]["epochs"])
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.max_train_batches
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, num_subjects, args.max_eval_batches
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "train": train_metrics,
            "val": {f"val_{k}": v for k, v in val_metrics.items()},
            "best_score": best_score,
            "patience_counter": patience_counter,
        }
        score = record["val"][monitor]
        if score > best_score:
            best_score = score
            patience_counter = 0
            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, scheduler, epoch, best_score, config, record)
        else:
            patience_counter += 1

        record["best_score"] = best_score
        record["patience_counter"] = patience_counter
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, scheduler, epoch, best_score, config, record)

        if (epoch + 1) % int(config["logging"].get("save_interval", 5)) == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch+1}.pt", model, optimizer, scheduler, epoch, best_score, config, record)

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Loss={train_metrics['loss']:.4f} | "
            f"Top1={train_metrics['top1_accuracy']*100:.2f}% | "
            f"Top10={train_metrics['top10_accuracy']*100:.2f}% | "
            f"ValWinTop1={val_metrics['window_top1']*100:.2f}% | "
            f"ValRecTop1={val_metrics['recording_top1']*100:.2f}% | "
            f"ValRecTop10={val_metrics['recording_top10']*100:.2f}%"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    best_path = output_dir / "checkpoint_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(
        model, test_loader, criterion, device, num_subjects, args.max_eval_batches
    )
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(test_metrics), f, indent=2, ensure_ascii=False)

    print("\n=== Test Results ===")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
