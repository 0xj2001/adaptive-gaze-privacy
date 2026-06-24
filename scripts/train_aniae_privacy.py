"""Train Fixed-Noise AE or ANIAE against frozen biometric/task evaluators."""

import argparse
import copy
import json
import multiprocessing
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.round_protocol import (
    RoundProtocolChunkDataset,
    load_manifest,
    load_recording_windows,
    make_loader,
    records_by_split,
)
from src.evaluation.biometric_metrics import embed_recording, evaluate_enrollment_probe
from src.models.attacker import PrivacyAttacker, gradient_reversal
from src.models.autoencoder import ANIAE
from src.models.biometric import GazeTaskEvaluator, build_biometric_model


def set_global_seed(seed: int) -> None:
    """Seed all RNGs used by the training/evaluation path."""
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


def fixed_sigma_is_excluded(config: dict, sigma: float | None) -> bool:
    if sigma is None:
        return False
    excluded = config.get("baselines", {}).get("excluded_fixed_ae_sigmas", [])
    sigma_value = round(float(sigma), 12)
    return any(round(float(item), 12) == sigma_value for item in excluded)


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


def load_biometric(config: dict, device: torch.device, checkpoint_path: str | None = None):
    ckpt = torch.load(
        checkpoint_path or config["paths"]["biometric_evaluator"],
        map_location=device,
        weights_only=False,
    )
    num_subjects = len(ckpt.get("label_map", {})) or ckpt["model_state"]["classifier.weight"].shape[0]
    model_config = ckpt.get("config", config)
    model = make_biometric_model(model_config, num_subjects, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_task(config: dict, device: torch.device, checkpoint_path: str | None = None):
    ckpt = torch.load(
        checkpoint_path or config["paths"]["task_evaluator"],
        map_location=device,
        weights_only=False,
    )
    model = make_task_model(config, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def ae_config_for_mode(config: dict, mode: str, sigma: float | None, vae_beta: float | None = None) -> dict:
    ae_config = copy.deepcopy(config)
    ae_config["data"]["window_size"] = int(config["data"]["subwindow_size"])
    if mode == "fixed":
        ae_config["model"]["noise"]["mode"] = "fixed"
        ae_config["model"]["noise"]["fixed_sigma"] = float(sigma)
    elif mode == "adaptive":
        ae_config["model"]["noise"]["mode"] = "adaptive"
    elif mode == "vae":
        ae_config["model"]["noise"]["mode"] = "vae"
        ae_config["aniae_training"]["lambda_privacy"] = 0.0
        ae_config["aniae_training"]["lambda_noise"] = 0.0
        if vae_beta is not None:
            ae_config["aniae_training"]["vae_beta"] = float(vae_beta)
    elif mode == "grl":
        ae_config["model"]["noise"]["mode"] = "grl"
        ae_config["aniae_training"]["lambda_privacy"] = 0.0
        ae_config["aniae_training"]["lambda_noise"] = 0.0
        ae_config["aniae_training"].setdefault("lambda_grl", 0.1)
        ae_config["aniae_training"].setdefault("grl_alpha", 1.0)
    else:
        raise ValueError(f"Unknown mode={mode}")
    return ae_config


def protect_with_aniae(
    aniae: ANIAE,
    x: torch.Tensor,
    subwindow_size: int,
    return_outputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
    bsz, seq_len, n_features = x.shape
    if seq_len % subwindow_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by subwindow_size={subwindow_size}")
    n_sub = seq_len // subwindow_size
    sub = x.reshape(bsz, n_sub, subwindow_size, n_features).reshape(bsz * n_sub, subwindow_size, n_features)
    out = aniae(sub)
    x_hat = out["x_hat"].reshape(bsz, n_sub, subwindow_size, n_features).reshape(bsz, seq_len, n_features)
    sigma = out["sigma"].reshape(bsz, n_sub, -1)
    x_hat = torch.clamp(x_hat, -1.0, 1.0)
    if return_outputs:
        return x_hat, sigma, out
    return x_hat, sigma


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x_hat, x)
    smooth = F.mse_loss(torch.diff(x_hat, dim=1), torch.diff(x, dim=1))
    return mse + 0.01 * smooth


def adaptive_noise_loss(aniae: ANIAE, sigma: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    if getattr(aniae, "noise_mode", None) != "adaptive":
        return sigma.new_tensor(0.0)
    sigma_mean = sigma.mean()
    target_sigma = sigma.new_tensor(float(train_cfg.get("target_sigma", 0.3)))
    sigma_upper = sigma.new_tensor(float(train_cfg.get("sigma_upper", 0.35)))
    under_target = F.relu(target_sigma - sigma_mean).square()
    over_upper = F.relu(sigma_mean - sigma_upper).square()
    return under_target + 0.1 * over_upper


def compute_validation_scores(
    val_bio: dict,
    val_task: dict,
    raw_rank1: float,
    raw_rank5: float,
    raw_task_acc: float,
    train_cfg: dict,
) -> dict:
    privacy_gain_rank1 = 1.0 - val_bio["rank1_ir"] / raw_rank1
    privacy_gain_rank5 = 1.0 - val_bio["rank5_ir"] / raw_rank5
    task_retention = val_task["task_accuracy"] / raw_task_acc
    reconstruction_mse = val_task["reconstruction_mse"]
    legacy_score = privacy_gain_rank1 + task_retention - val_task["reconstruction_mse"]
    balanced_score = (
        privacy_gain_rank1
        + 0.5 * privacy_gain_rank5
        + 0.75 * val_bio["eer"]
        + task_retention
        - 2.0 * reconstruction_mse
    )
    min_task_retention = float(train_cfg.get("min_task_retention", 0.922))
    max_reconstruction_mse = float(train_cfg.get("max_reconstruction_mse", 0.025))
    task_shortfall = max(0.0, min_task_retention - task_retention)
    mse_excess = max(0.0, reconstruction_mse - max_reconstruction_mse)
    acceptance_score = (
        balanced_score
        - float(train_cfg.get("task_shortfall_penalty", 2.0)) * task_shortfall
        - float(train_cfg.get("mse_excess_penalty", 10.0)) * mse_excess
    )
    max_val_rank1 = float(train_cfg.get("max_val_rank1", 1.0))
    max_val_rank5 = float(train_cfg.get("max_val_rank5", 1.0))
    rank1_excess = max(0.0, float(val_bio["rank1_ir"]) - max_val_rank1)
    rank5_excess = max(0.0, float(val_bio["rank5_ir"]) - max_val_rank5)
    privacy_penalty = (
        float(train_cfg.get("rank1_excess_penalty", 4.0)) * rank1_excess
        + float(train_cfg.get("rank5_excess_penalty", 1.0)) * rank5_excess
    )
    utility_constrained_score = (
        task_retention
        - 1.5 * reconstruction_mse
        + 0.25 * val_bio["eer"]
        - privacy_penalty
    )
    rank1_recovery_score = (
        task_retention
        - 1.5 * reconstruction_mse
        + 0.35 * privacy_gain_rank1
        + 0.10 * privacy_gain_rank5
        + 0.25 * val_bio["eer"]
        - float(train_cfg.get("task_shortfall_penalty", 2.0)) * task_shortfall
        - float(train_cfg.get("mse_excess_penalty", 10.0)) * mse_excess
        - privacy_penalty
    )
    reconstruction_utility_score = task_retention - 1.5 * reconstruction_mse
    return {
        "val_privacy_gain_rank1": privacy_gain_rank1,
        "val_privacy_gain_rank5": privacy_gain_rank5,
        "val_task_retention": task_retention,
        "val_score": legacy_score,
        "val_balanced_score": balanced_score,
        "val_acceptance_score": acceptance_score,
        "val_task_shortfall": task_shortfall,
        "val_mse_excess": mse_excess,
        "val_utility_constrained_score": utility_constrained_score,
        "val_rank1_recovery_score": rank1_recovery_score,
        "val_reconstruction_utility_score": reconstruction_utility_score,
        "val_privacy_penalty": privacy_penalty,
        "val_rank1_excess": rank1_excess,
        "val_rank5_excess": rank5_excess,
    }


def task_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("task_distill_temperature must be positive")
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def set_trainable_phase(aniae: ANIAE, phase: str) -> None:
    for param in aniae.parameters():
        param.requires_grad_(True)
    if phase == "decoder_recovery":
        for param in aniae.encoder.parameters():
            param.requires_grad_(False)
        if getattr(aniae, "noise_injector", None) is not None:
            for param in aniae.noise_injector.parameters():
                param.requires_grad_(False)
        for module_name in ("vae_mu", "vae_logvar"):
            module = getattr(aniae, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)
    elif phase != "joint_recovery":
        raise ValueError(f"Unknown ANIAE training phase: {phase}")


def get_training_phase(epoch: int, train_cfg: dict) -> str:
    decoder_recovery_epochs = int(train_cfg.get("decoder_recovery_epochs", 0))
    return "decoder_recovery" if epoch < decoder_recovery_epochs else "joint_recovery"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def build_template_bank(config: dict, biometric, cache_dir: Path, train_records: list[dict], label_map: dict[int, int], device):
    protocol = config["data"].get("round_protocol", {})
    task = protocol.get("enrollment_task", "RAN")
    session = int(protocol.get("enrollment_session", 1))
    records = {}
    for rec in sorted(train_records, key=lambda r: (r["person_id"], r["file"])):
        if rec["task"] == task and int(rec["session"]) == session:
            records.setdefault(int(rec["person_id"]), rec)
    embeddings = []
    for person_id, _ in sorted(label_map.items(), key=lambda item: item[1]):
        if int(person_id) in records:
            emb = embed_recording(
                biometric,
                cache_dir,
                records[int(person_id)],
                device,
                batch_size=int(config["biometric_training"].get("eval_batch_size", 128)),
            )
        else:
            person_records = [r for r in train_records if int(r["person_id"]) == int(person_id)]
            chunks = [load_recording_windows(cache_dir, rec) for rec in person_records]
            x = torch.cat(chunks, dim=0).to(device)
            emb = biometric.embed(x).mean(dim=0).detach().cpu()
            emb = F.normalize(emb, p=2, dim=0)
        embeddings.append(emb)
    return torch.stack(embeddings, dim=0).to(device)


def train_one_epoch(
    aniae,
    biometric,
    task_model,
    template_bank,
    loader,
    optimizer,
    config,
    device,
    grl_attacker=None,
    max_batches=None,
) -> dict:
    aniae.train()
    if grl_attacker is not None:
        grl_attacker.train()
    task_ce = nn.CrossEntropyLoss()
    train_cfg = config["aniae_training"]
    subwindow_size = int(config["data"]["subwindow_size"])
    temperature = float(train_cfg.get("privacy_temperature", 0.1))
    lambda_privacy = float(train_cfg.get("lambda_privacy", 0.0))
    lambda_grl = float(train_cfg.get("lambda_grl", 0.0))
    vae_beta = float(train_cfg.get("vae_beta", 0.0))
    totals = {
        "loss": 0.0,
        "loss_rec": 0.0,
        "loss_privacy": 0.0,
        "loss_utility": 0.0,
        "loss_task_ce": 0.0,
        "loss_task_distill": 0.0,
        "loss_noise": 0.0,
        "loss_kl": 0.0,
        "loss_grl": 0.0,
        "grl_accuracy": 0.0,
        "sigma_mean": 0.0,
        "sigma_std": 0.0,
        "n_batches": 0,
    }

    for batch_idx, (x, labels, task_labels, _) in enumerate(tqdm(loader, desc="TrainAE", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        labels = labels.to(device)
        task_labels = task_labels.to(device)
        x_hat, sigma, ae_out = protect_with_aniae(aniae, x, subwindow_size, return_outputs=True)
        if lambda_privacy > 0.0:
            if template_bank is None:
                raise ValueError("template_bank is required when lambda_privacy > 0")
            embeddings = biometric.embed(x_hat)
            similarity = embeddings @ template_bank.t() / temperature
            uniform = torch.full_like(similarity, 1.0 / similarity.shape[1])
            privacy_loss = F.kl_div(F.log_softmax(similarity, dim=1), uniform, reduction="batchmean")
        else:
            privacy_loss = x.new_tensor(0.0)
        task_logits = task_model(x_hat)
        task_ce_loss = task_ce(task_logits, task_labels)
        distill_weight = float(train_cfg.get("lambda_task_distill", 0.0))
        if distill_weight > 0:
            with torch.no_grad():
                teacher_logits = task_model(x)
            distill_loss = task_distillation_loss(
                task_logits,
                teacher_logits,
                float(train_cfg.get("task_distill_temperature", 2.0)),
            )
        else:
            distill_loss = x.new_tensor(0.0)
        utility_loss = task_ce_loss + distill_weight * distill_loss
        rec_loss = reconstruction_loss(x, x_hat)
        noise_loss = adaptive_noise_loss(aniae, sigma, train_cfg)
        kl_loss = ae_out.get("latent_kl", x.new_tensor(0.0))
        grl_loss = x.new_tensor(0.0)
        grl_accuracy = x.new_tensor(0.0)
        if grl_attacker is not None and lambda_grl > 0.0:
            latent = ae_out["z_noisy"]
            labels_sub = labels.repeat_interleave(x.shape[1] // subwindow_size)
            grl_logits = grl_attacker(
                gradient_reversal(latent, float(train_cfg.get("grl_alpha", 1.0)))
            )
            grl_loss = task_ce(grl_logits, labels_sub)
            grl_accuracy = (grl_logits.argmax(1) == labels_sub).float().mean()
        loss = (
            float(train_cfg["lambda_reconstruction"]) * rec_loss
            + lambda_privacy * privacy_loss
            + float(train_cfg["lambda_utility"]) * utility_loss
            + float(train_cfg.get("lambda_noise", 0.01)) * noise_loss
            + vae_beta * kl_loss
            + lambda_grl * grl_loss
        )

        optimizer.zero_grad()
        loss.backward()
        clip_params = list(aniae.parameters())
        if grl_attacker is not None:
            clip_params.extend(grl_attacker.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["loss_rec"] += rec_loss.item()
        totals["loss_privacy"] += privacy_loss.item()
        totals["loss_utility"] += utility_loss.item()
        totals["loss_task_ce"] += task_ce_loss.item()
        totals["loss_task_distill"] += distill_loss.item()
        totals["loss_noise"] += noise_loss.item()
        totals["loss_kl"] += kl_loss.item()
        totals["loss_grl"] += grl_loss.item()
        totals["grl_accuracy"] += grl_accuracy.item()
        totals["sigma_mean"] += sigma.mean().item()
        totals["sigma_std"] += sigma.std().item()
        totals["n_batches"] += 1

    n = max(1, totals.pop("n_batches"))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate_task_reconstruction(aniae, task_model, loader, config, device, max_batches=None) -> dict:
    aniae.eval()
    task_model.eval()
    subwindow_size = int(config["data"]["subwindow_size"])
    total = 0
    correct = 0
    mse = 0.0
    mae = 0.0
    sigma_values = []
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc="ValAE", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        task_labels = task_labels.to(device)
        x_hat, sigma = protect_with_aniae(aniae, x, subwindow_size)
        logits = task_model(x_hat)
        correct += (logits.argmax(1) == task_labels).sum().item()
        total += task_labels.numel()
        diff = x_hat - x
        mse += diff.square().mean(dim=(1, 2)).sum().item()
        mae += diff.abs().mean(dim=(1, 2)).sum().item()
        sigma_values.append(sigma.detach().flatten().cpu())
    sigmas = torch.cat(sigma_values) if sigma_values else torch.tensor([])
    return {
        "task_accuracy": correct / max(1, total),
        "reconstruction_mse": mse / max(1, total),
        "reconstruction_mae": mae / max(1, total),
        "sigma_mean": float(sigmas.mean().item()) if sigmas.numel() else float("nan"),
        "sigma_std": float(sigmas.std().item()) if sigmas.numel() else float("nan"),
    }


@torch.no_grad()
def evaluate_plain_task(task_model, loader, device, max_batches=None) -> dict:
    task_model.eval()
    total = 0
    correct = 0
    for batch_idx, (x, _, task_labels, _) in enumerate(tqdm(loader, desc="RawTask", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        task_labels = task_labels.to(device)
        logits = task_model(x)
        correct += (logits.argmax(1) == task_labels).sum().item()
        total += task_labels.numel()
    return {"task_accuracy": correct / max(1, total)}


def average_metric_dicts(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = records[0].keys()
    averaged = {}
    for key in keys:
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float)) for value in values):
            averaged[key] = sum(float(value) for value in values) / len(values)
        else:
            averaged[key] = values[0]
    return averaged


def save_checkpoint(
    path: Path,
    aniae,
    optimizer,
    epoch: int,
    best_score: float,
    config: dict,
    metrics: dict,
    grl_attacker=None,
):
    payload = {
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
        "aniae_state": aniae.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if grl_attacker is not None:
        payload["grl_attacker_state"] = grl_attacker.state_dict()
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="Train Fixed-Noise AE or ANIAE")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["fixed", "adaptive", "vae", "grl"], required=True)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--vae_beta", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--biometric_checkpoint", type=str, default=None)
    parser.add_argument("--task_checkpoint", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_num_workers", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    config.setdefault("project", {})["seed"] = seed
    set_global_seed(seed)
    if args.mode == "fixed" and args.sigma is None:
        raise SystemExit("--sigma is required for --mode fixed")
    if args.mode == "fixed" and fixed_sigma_is_excluded(config, args.sigma):
        raise SystemExit(f"Fixed AE sigma={args.sigma} is excluded by baselines.excluded_fixed_ae_sigmas")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(config["data"]["cache_dir"])
    splits = records_by_split(load_manifest(cache_dir), config["data"]["tasks"])
    train_records = splits["train"]
    val_records = splits["val"]
    label_map = {
        person_id: idx
        for idx, person_id in enumerate(sorted({int(r["person_id"]) for r in train_records}))
    }
    train_ds = RoundProtocolChunkDataset(
        cache_dir,
        train_records,
        identity_field=config["data"]["identity_field"],
        label_map=label_map,
        tasks=config["data"]["tasks"],
        cache_lru_size=int(config["data"].get("cache_lru_size", 64)),
    )
    val_label_map = {
        person_id: idx
        for idx, person_id in enumerate(sorted({int(r["person_id"]) for r in val_records}))
    }
    val_ds = RoundProtocolChunkDataset(
        cache_dir,
        val_records,
        identity_field=config["data"]["identity_field"],
        label_map=val_label_map,
        tasks=config["data"]["tasks"],
        cache_lru_size=int(config["data"].get("cache_lru_size", 64)),
    )
    train_loader = make_loader(
        train_ds,
        batch_size=int(config["aniae_training"]["batch_size"]),
        shuffle=True,
        num_workers=int(args.num_workers if args.num_workers is not None else config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", torch.cuda.is_available())),
        seed=seed,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(config["aniae_training"]["batch_size"]),
        shuffle=False,
        num_workers=int(args.val_num_workers if args.val_num_workers is not None else config["data"].get("val_num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", torch.cuda.is_available())),
        seed=seed + 1,
    )

    ae_config = ae_config_for_mode(config, args.mode, args.sigma, args.vae_beta)
    config = ae_config
    ae_config.setdefault("project", {})["seed"] = seed
    biometric = load_biometric(config, device, args.biometric_checkpoint)
    task_model = load_task(config, device, args.task_checkpoint)
    if float(config["aniae_training"].get("lambda_privacy", 0.0)) > 0.0:
        template_bank = build_template_bank(config, biometric, cache_dir, train_records, label_map, device)
    else:
        template_bank = None
    aniae = ANIAE(ae_config).to(device)
    init_checkpoint = args.init_checkpoint or config.get("init_checkpoint") or config["aniae_training"].get("init_checkpoint")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Missing init checkpoint: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        init_state = init_ckpt.get("aniae_state", init_ckpt.get("model_state"))
        if init_state is None:
            raise KeyError(f"No ANIAE state found in init checkpoint: {init_path}")
        strict_load = args.mode in {"fixed", "adaptive"}
        load_result = aniae.load_state_dict(init_state, strict=strict_load)
        ae_config["init_checkpoint"] = str(init_path)
        print(f"Initialized ANIAE from: {init_path}")
        if not strict_load:
            print(f"Non-strict init load result: {load_result}")
    grl_attacker = None
    if args.mode == "grl":
        attacker_cfg = ae_config.get("model", {}).get("attacker", {})
        train_cfg = ae_config["aniae_training"]
        grl_attacker = PrivacyAttacker(
            input_dim=int(ae_config["model"]["encoder"]["latent_dim"]),
            num_subjects=len(label_map),
            hidden_dims=train_cfg.get("grl_hidden_dims", attacker_cfg.get("hidden_dims", [128, 64])),
            dropout=float(train_cfg.get("grl_dropout", attacker_cfg.get("dropout", 0.2))),
            input_type="latent",
        ).to(device)
    optim_params = list(aniae.parameters())
    if grl_attacker is not None:
        optim_params.extend(grl_attacker.parameters())
    optimizer = optim.Adam(
        optim_params,
        lr=float(ae_config["aniae_training"]["lr_autoencoder"]),
        weight_decay=float(ae_config["aniae_training"]["weight_decay"]),
    )

    if args.run_name:
        run_name = args.run_name
    elif args.mode == "fixed":
        run_name = f"fixed_sigma_{args.sigma}"
    elif args.mode == "vae":
        beta = float(ae_config["aniae_training"].get("vae_beta", 0.0))
        beta_tag = f"{beta:g}".replace(".", "p")
        run_name = f"vae_ae_beta{beta_tag}"
    elif args.mode == "grl":
        run_name = "grl_ae"
    else:
        run_name = "aniae_adaptive"
    output_dir = Path(config["paths"].get("results_dir", "./experiments/round_protocol_results")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(ae_config, f, sort_keys=False, allow_unicode=True)

    print(f"Mode: {args.mode} sigma={args.sigma} vae_beta={ae_config['aniae_training'].get('vae_beta')}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output dir: {output_dir}")
    print(f"Train chunks: {len(train_ds)} | Val chunks: {len(val_ds)}")
    if grl_attacker is not None:
        print(f"GRL attacker parameters: {count_trainable_parameters(grl_attacker)}")

    protocol = config["data"].get("round_protocol", {})
    eval_kwargs = dict(
        task=protocol.get("enrollment_task", "RAN"),
        enrollment_session=int(protocol.get("enrollment_session", 1)),
        probe_session=int(protocol.get("probe_session", 2)),
        batch_size=int(config["biometric_training"].get("eval_batch_size", 128)),
    )
    raw_bio = evaluate_enrollment_probe(biometric, cache_dir, val_records, device, **eval_kwargs)
    raw_task = evaluate_plain_task(task_model, val_loader, device, args.max_eval_batches)
    raw_rank1 = max(raw_bio["rank1_ir"], 1e-8)
    raw_rank5 = max(raw_bio["rank5_ir"], 1e-8)
    raw_task_acc = max(raw_task["task_accuracy"], 1e-8)
    monitor = config["aniae_training"].get("monitor", "val_score")
    eval_repeats = max(1, int(config["aniae_training"].get("eval_repeats", 1)))

    best_score = -float("inf")
    patience_counter = 0
    epochs = int(config["aniae_training"]["epochs"])
    if args.max_epochs is not None:
        epochs = min(epochs, args.max_epochs)
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(epochs):
        phase = get_training_phase(epoch, config["aniae_training"])
        set_trainable_phase(aniae, phase)
        train_metrics = train_one_epoch(
            aniae,
            biometric,
            task_model,
            template_bank,
            train_loader,
            optimizer,
            config,
            device,
            grl_attacker,
            args.max_train_batches,
        )

        def transform(x):
            x_hat, _ = protect_with_aniae(aniae, x, int(config["data"]["subwindow_size"]))
            return x_hat

        val_bio_records = []
        val_task_records = []
        for repeat_idx in range(eval_repeats):
            repeat_seed = seed + epoch * 1000 + repeat_idx
            torch.manual_seed(repeat_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(repeat_seed)
            val_bio_records.append(
                evaluate_enrollment_probe(
                    biometric, cache_dir, val_records, device, transform=transform, **eval_kwargs
                )
            )
            val_task_records.append(
                evaluate_task_reconstruction(
                    aniae, task_model, val_loader, config, device, args.max_eval_batches
                )
            )
        val_bio = average_metric_dicts(val_bio_records)
        val_task = average_metric_dicts(val_task_records)
        score_metrics = compute_validation_scores(
            val_bio,
            val_task,
            raw_rank1,
            raw_rank5,
            raw_task_acc,
            config["aniae_training"],
        )
        val_metrics = {
            **{f"val_{k}": v for k, v in val_bio.items()},
            **{f"val_{k}": v for k, v in val_task.items()},
            **score_metrics,
        }
        if monitor not in val_metrics:
            raise ValueError(f"Unknown aniae_training.monitor={monitor!r}. Available metrics: {sorted(val_metrics)}")
        score = val_metrics[monitor]
        record = {
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "monitor": monitor,
            "phase": phase,
            "trainable_parameters": count_trainable_parameters(aniae),
            "best_score": best_score,
            "patience_counter": patience_counter,
        }
        if score > best_score:
            best_score = score
            patience_counter = 0
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                aniae,
                optimizer,
                epoch,
                best_score,
                ae_config,
                record,
                grl_attacker,
            )
        else:
            patience_counter += 1
        record["best_score"] = best_score
        record["patience_counter"] = patience_counter
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        save_checkpoint(
            output_dir / "checkpoint_latest.pt",
            aniae,
            optimizer,
            epoch,
            best_score,
            ae_config,
            record,
            grl_attacker,
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Rec={train_metrics['loss_rec']:.4f} | "
            f"Priv={train_metrics['loss_privacy']:.4f} | "
            f"Util={train_metrics['loss_utility']:.4f} | "
            f"Phase={phase} | "
            f"ValRank1={val_bio['rank1_ir']*100:.2f}% | "
            f"ValTask={val_task['task_accuracy']*100:.2f}% | "
            f"Score={val_metrics['val_score']:.4f} | "
            f"Balanced={val_metrics['val_balanced_score']:.4f} | "
            f"Accept={val_metrics['val_acceptance_score']:.4f} | "
            f"UtilityScore={val_metrics['val_utility_constrained_score']:.4f} | "
            f"Rank1Recovery={val_metrics['val_rank1_recovery_score']:.4f} | "
            f"Monitor={monitor}:{score:.4f}"
        )

        if patience_counter >= int(config["aniae_training"]["patience"]):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Training complete. Best score: {best_score:.4f}")
    with open(output_dir / "training_complete.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "mode": args.mode,
                "sigma": args.sigma,
                "vae_beta": ae_config["aniae_training"].get("vae_beta"),
                "seed": seed,
                "best_score": best_score,
                "checkpoint_best": str(output_dir / "checkpoint_best.pt"),
                "checkpoint_latest": str(output_dir / "checkpoint_latest.pt"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)
