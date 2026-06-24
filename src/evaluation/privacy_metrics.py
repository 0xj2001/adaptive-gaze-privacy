"""
Privacy and utility evaluation metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve


def _attacker_input(attacker: nn.Module, ae_out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Choose latent or protected sequence input based on attacker type."""
    if getattr(attacker, "input_type", "latent") in ("sequence", "sequence_light"):
        return ae_out["x_hat"]
    return ae_out["z_noisy"]


def _recording_key(sample: dict) -> str:
    return str(sample.get("file", sample.get("cache_file", "")))


def _dataset_samples_for_batch(dataset, offset: int, batch_size: int) -> list[dict]:
    if hasattr(dataset, "get_samples_for_range"):
        return dataset.get_samples_for_range(offset, batch_size)
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return []
    return samples[offset:offset + batch_size]


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.shape[1])
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1)).any(dim=1).float().mean()
    return float(correct.item())


def _eer_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
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
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_identification_metrics(
    model: nn.Module,
    attacker: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate identity leakage at window and recording aggregation levels."""
    model.eval()
    attacker.eval()

    total = 0
    window_correct = {1: 0.0, 5: 0.0, 10: 0.0}
    recording_probs: dict[str, torch.Tensor] = {}
    recording_labels: dict[str, int] = {}
    sample_offset = 0
    num_classes = None

    with torch.no_grad():
        for x, labels, _ in dataloader:
            batch_size = labels.size(0)
            batch_samples = _dataset_samples_for_batch(dataloader.dataset, sample_offset, batch_size)
            sample_offset += batch_size

            x = x.to(device)
            labels = labels.to(device)

            ae_out = model(x)
            logits = attacker(_attacker_input(attacker, ae_out))
            num_classes = logits.shape[1]
            probs = torch.softmax(logits, dim=1).detach().cpu()

            valid = labels < logits.shape[1]
            if not valid.any():
                continue
            logits_valid = logits[valid]
            labels_valid = labels[valid]

            for k in window_correct:
                window_correct[k] += _topk_accuracy(logits_valid, labels_valid, k) * labels_valid.numel()
            total += labels_valid.numel()

            if batch_samples:
                valid_cpu = valid.detach().cpu().tolist()
                labels_cpu = labels.detach().cpu()
                for sample, keep, prob, label in zip(batch_samples, valid_cpu, probs, labels_cpu):
                    if not keep:
                        continue
                    key = _recording_key(sample)
                    if key not in recording_probs:
                        recording_probs[key] = prob.clone()
                        recording_labels[key] = int(label.item())
                    else:
                        recording_probs[key] += prob

    if total == 0 or num_classes is None:
        return {
            "window_top1": float("nan"),
            "window_top5": float("nan"),
            "window_top10": float("nan"),
            "recording_top1": float("nan"),
            "recording_top5": float("nan"),
            "recording_top10": float("nan"),
            "random_top1": float("nan"),
            "random_top10": float("nan"),
            "identification_accuracy": float("nan"),
            "privacy_gain": float("nan"),
        }

    metrics = {
        "window_top1": window_correct[1] / total,
        "window_top5": window_correct[5] / total,
        "window_top10": window_correct[10] / total,
        "random_top1": 1.0 / num_classes,
        "random_top10": min(10, num_classes) / num_classes,
    }

    if recording_probs:
        rec_keys = sorted(recording_probs)
        rec_probs = torch.stack([recording_probs[key] for key in rec_keys], dim=0)
        rec_probs = rec_probs / rec_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        rec_labels = torch.tensor([recording_labels[key] for key in rec_keys], dtype=torch.long)
        metrics["recording_top1"] = _topk_accuracy(rec_probs, rec_labels, 1)
        metrics["recording_top5"] = _topk_accuracy(rec_probs, rec_labels, 5)
        metrics["recording_top10"] = _topk_accuracy(rec_probs, rec_labels, 10)
        metrics["recording_eer"] = _eer_from_probs(rec_probs, rec_labels)
    else:
        metrics["recording_top1"] = float("nan")
        metrics["recording_top5"] = float("nan")
        metrics["recording_top10"] = float("nan")
        metrics["recording_eer"] = float("nan")

    metrics["identification_accuracy"] = metrics["window_top1"]
    metrics["privacy_gain"] = 1.0 - metrics["window_top1"] / (1.0 + 1e-8)
    return metrics


def compute_identification_accuracy(
    model: nn.Module,
    attacker: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Evaluate identity classification accuracy on protected data.

    Lower accuracy = better privacy.
    """
    metrics = compute_identification_metrics(model, attacker, dataloader, device)
    return {
        "identification_accuracy": metrics["identification_accuracy"],
        "random_baseline": metrics["random_top1"],
        "privacy_gain": metrics["privacy_gain"],
    }


def compute_eer(
    model: nn.Module,
    attacker: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Compute classifier-probability Equal Error Rate.

    This legacy helper treats each true-class probability as a genuine score
    and every wrong-class probability as an impostor score. The publication
    protocol uses enrollment/probe similarity matrices in
    src.evaluation.biometric_metrics instead.
    """
    model.eval()
    attacker.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, labels, _ in dataloader:
            x = x.to(device)
            ae_out = model(x)
            logits = attacker(_attacker_input(attacker, ae_out))
            probs = torch.softmax(logits, dim=1)
            valid = labels < probs.shape[1]
            if not valid.any():
                continue
            probs = probs[valid]
            labels = labels[valid]

            for row, label in zip(probs, labels):
                true_label = int(label.item())
                for class_idx in range(row.numel()):
                    all_scores.append(float(row[class_idx].item()))
                    all_labels.append(1 if class_idx == true_label else 0)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    if len(np.unique(all_labels)) < 2:
        return float("nan")

    # Compute EER
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    # EER is where FPR = FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer


def compute_reconstruction_quality(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Measure reconstruction quality (utility preservation).

    Lower distortion = better utility.
    """
    model.eval()

    total_mse = 0
    total_mae = 0
    total_samples = 0

    all_correlations = []

    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.to(device)
            ae_out = model(x)
            x_hat = ae_out["x_hat"]

            # MSE and MAE
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
            mae = (x - x_hat).abs().mean(dim=(1, 2))

            total_mse += mse.sum().item()
            total_mae += mae.sum().item()
            total_samples += x.size(0)

            # Pearson correlation per sample (averaged over features)
            for i in range(min(x.size(0), 50)):  # subsample for speed
                x_flat = x[i].cpu().numpy().flatten()
                xh_flat = x_hat[i].cpu().numpy().flatten()
                r, _ = pearsonr(x_flat, xh_flat)
                if not np.isnan(r):
                    all_correlations.append(r)

    if total_samples == 0:
        return {
            "reconstruction_mse": float("nan"),
            "reconstruction_mae": float("nan"),
            "signal_correlation": float("nan"),
        }

    return {
        "reconstruction_mse": total_mse / total_samples,
        "reconstruction_mae": total_mae / total_samples,
        "signal_correlation": np.mean(all_correlations) if all_correlations else 0.0,
    }


def compute_noise_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """
    Analyze the learned noise distribution across latent dimensions.

    Returns per-dimension noise statistics.
    """
    model.eval()
    all_sigmas = []

    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.to(device)
            ae_out = model(x)
            all_sigmas.append(ae_out["sigma"].cpu().numpy())

    if not all_sigmas:
        return {
            "sigma_mean_per_dim": np.array([]),
            "sigma_std_per_dim": np.array([]),
            "sigma_global_mean": float("nan"),
            "sigma_global_std": float("nan"),
        }

    all_sigmas = np.concatenate(all_sigmas, axis=0)

    return {
        "sigma_mean_per_dim": all_sigmas.mean(axis=0),
        "sigma_std_per_dim": all_sigmas.std(axis=0),
        "sigma_global_mean": all_sigmas.mean(),
        "sigma_global_std": all_sigmas.std(),
    }


def full_evaluation(
    model: nn.Module,
    attacker: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict:
    """Run all evaluation metrics."""
    results = {}

    # Privacy metrics
    id_metrics = compute_identification_metrics(model, attacker, dataloader, device)
    results.update(id_metrics)

    eer = compute_eer(model, attacker, dataloader, device)
    results["equal_error_rate"] = eer

    # Utility metrics
    recon_metrics = compute_reconstruction_quality(model, dataloader, device)
    results.update(recon_metrics)

    # Noise analysis
    noise_metrics = compute_noise_analysis(model, dataloader, device)
    results["sigma_global_mean"] = noise_metrics["sigma_global_mean"]

    return results
