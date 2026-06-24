"""
Training pipeline for ANIAE with adversarial privacy training.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.autoencoder import ANIAE
from ..models.attacker import PrivacyAttacker, UtilityModel
from .losses import CombinedLoss, PrivacyLoss


class Trainer:
    """Adversarial training pipeline for ANIAE."""

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        train_cfg = config["training"]
        self.train_cfg = train_cfg

        # Models
        self.aniae = ANIAE(config).to(self.device)

        self.num_subjects = 322  # GazeBase total subjects (adjusted at runtime)
        num_tasks = len(config["data"]["tasks"])
        latent_dim = config["model"]["encoder"]["latent_dim"]
        self.adversary_input = train_cfg.get("adversary_input", "latent")
        attacker_input_dim = (
            config["model"]["decoder"]["output_dim"]
            if self.adversary_input in ("sequence", "sequence_light")
            else latent_dim
        )

        self.attacker = PrivacyAttacker(
            input_dim=attacker_input_dim,
            num_subjects=self.num_subjects,
            hidden_dims=config["model"]["attacker"]["hidden_dims"],
            dropout=config["model"]["attacker"]["dropout"],
            input_type=self.adversary_input,
        ).to(self.device)

        self.utility_model = UtilityModel(
            input_dim=latent_dim,
            num_classes=num_tasks,
            hidden_dims=config["model"]["utility"]["hidden_dims"],
            dropout=config["model"]["utility"]["dropout"],
            input_type="latent",
        ).to(self.device)

        # Optimizers
        weight_decay = float(train_cfg["weight_decay"])
        self.opt_ae = optim.Adam(
            self.aniae.parameters(),
            lr=float(train_cfg["lr_autoencoder"]),
            weight_decay=weight_decay,
        )
        self.opt_attacker = optim.Adam(
            self.attacker.parameters(),
            lr=float(train_cfg["lr_attacker"]),
            weight_decay=weight_decay,
        )
        self.opt_utility = optim.Adam(
            self.utility_model.parameters(),
            lr=float(train_cfg["lr_utility"]),
            weight_decay=weight_decay,
        )

        # Schedulers
        self.sched_ae = CosineAnnealingLR(self.opt_ae, T_max=train_cfg["epochs"])
        self.sched_attacker = CosineAnnealingLR(self.opt_attacker, T_max=train_cfg["epochs"])

        # Losses
        self.combined_loss = CombinedLoss(config)
        self.privacy_loss = PrivacyLoss()

        # Training params
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        self.attacker_warmup_epochs = int(train_cfg.get("attacker_warmup_epochs", 0))
        if "joint_epochs" in train_cfg:
            self.epochs = (
                self.warmup_epochs
                + self.attacker_warmup_epochs
                + int(train_cfg["joint_epochs"])
            )
        else:
            self.epochs = int(train_cfg["epochs"])
        self.attacker_pretrain_epochs = int(train_cfg.get("attacker_pretrain_epochs", 0))
        self.attacker_steps = int(train_cfg.get("attacker_steps_per_ae_step", 1))
        self.patience = int(train_cfg["patience"])
        self.start_epoch = 0
        self.best_score = -float("inf")
        self.patience_counter = 0

        # Output
        self.output_dir = Path(config["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_num_subjects(self, num_subjects: int):
        """Rebuild attacker and optimizer when the identity class count changes."""
        if num_subjects == self.num_subjects:
            return

        self.num_subjects = num_subjects
        latent_dim = self.config["model"]["encoder"]["latent_dim"]
        attacker_input_dim = (
            self.config["model"]["decoder"]["output_dim"]
            if self.adversary_input in ("sequence", "sequence_light")
            else latent_dim
        )
        self.attacker = PrivacyAttacker(
            input_dim=attacker_input_dim,
            num_subjects=num_subjects,
            hidden_dims=self.config["model"]["attacker"]["hidden_dims"],
            dropout=self.config["model"]["attacker"]["dropout"],
            input_type=self.adversary_input,
        ).to(self.device)

        weight_decay = float(self.train_cfg["weight_decay"])
        self.opt_attacker = optim.Adam(
            self.attacker.parameters(),
            lr=float(self.train_cfg["lr_attacker"]),
            weight_decay=weight_decay,
        )
        self.sched_attacker = CosineAnnealingLR(
            self.opt_attacker,
            T_max=self.train_cfg["epochs"],
        )

    def _attacker_input(self, ae_out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select the representation used by the identity attacker."""
        if self.adversary_input in ("sequence", "sequence_light"):
            return ae_out["x_hat"]
        return ae_out["z_noisy"]

    def _phase_for_epoch(self, epoch: int) -> str:
        if epoch < self.warmup_epochs:
            return "warmup"
        if epoch < self.warmup_epochs + self.attacker_warmup_epochs:
            return "attacker_warmup"
        return "joint"

    def pretrain_attacker(self, train_loader: DataLoader):
        """Pretrain attacker on raw (unprotected) data to establish baseline."""
        print("Pretraining attacker on raw data...")
        self.aniae.eval()
        self.attacker.train()

        for epoch in range(self.attacker_pretrain_epochs):
            total_loss = 0
            correct = 0
            total = 0

            for x, labels, tasks in tqdm(train_loader, desc=f"Pretrain {epoch+1}"):
                x = x.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    z = self.aniae.encode(x)

                logits = self.attacker(z)
                loss = self.privacy_loss.for_attacker(logits, labels)

                self.opt_attacker.zero_grad()
                loss.backward()
                self.opt_attacker.step()

                total_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

            acc = correct / total * 100
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1f}%")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train one epoch with adversarial training."""
        phase = self._phase_for_epoch(epoch)
        self.aniae.train(phase != "attacker_warmup")
        self.attacker.train(phase != "warmup")
        self.utility_model.train(phase != "attacker_warmup")

        # Map task names to indices
        task_to_idx = {t: i for i, t in enumerate(self.config["data"]["tasks"])}

        metrics = {
            "loss_total": 0, "loss_rec": 0, "loss_priv": 0, "loss_util": 0,
            "attacker_acc": 0, "utility_acc": 0, "mean_sigma": 0, "n_batches": 0,
        }

        for x, labels, tasks in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x = x.to(self.device)
            labels = labels.to(self.device)
            task_labels = torch.tensor(
                [task_to_idx.get(t, 0) for t in tasks], device=self.device
            )

            if phase == "attacker_warmup":
                with torch.no_grad():
                    ae_out = self.aniae(x)
                    util_logits = self.utility_model(ae_out["z_noisy"])
                    rec_loss = nn.MSELoss()(ae_out["x_hat"], x)

                att_logits = self.attacker(self._attacker_input(ae_out).detach())
                att_loss = self.privacy_loss.for_attacker(att_logits, labels)

                self.opt_attacker.zero_grad()
                att_loss.backward()
                self.opt_attacker.step()

                util_loss = nn.CrossEntropyLoss()(util_logits, task_labels)
                losses = {
                    "total": att_loss.detach(),
                    "reconstruction": rec_loss.detach(),
                    "privacy": att_loss.detach(),
                    "utility": util_loss.detach(),
                }
                sigma = ae_out["sigma"]
            else:
                # --- Step 1: Update attacker during joint training ---
                if phase == "joint":
                    for _ in range(self.attacker_steps):
                        with torch.no_grad():
                            ae_out = self.aniae(x)
                        att_logits = self.attacker(self._attacker_input(ae_out).detach())
                        att_loss = self.privacy_loss.for_attacker(att_logits, labels)

                        self.opt_attacker.zero_grad()
                        att_loss.backward()
                        self.opt_attacker.step()

                # --- Step 2: Update utility model ---
                with torch.no_grad():
                    ae_out = self.aniae(x)
                z_noisy = ae_out["z_noisy"]

                util_logits = self.utility_model(z_noisy.detach())
                util_loss = nn.CrossEntropyLoss()(util_logits, task_labels)

                self.opt_utility.zero_grad()
                util_loss.backward()
                self.opt_utility.step()

                # --- Step 3: Update autoencoder ---
                ae_out = self.aniae(x)
                x_hat = ae_out["x_hat"]
                z_noisy = ae_out["z_noisy"]
                sigma = ae_out["sigma"]

                if phase == "joint":
                    att_logits = self.attacker(self._attacker_input(ae_out))
                    util_logits = self.utility_model(z_noisy)
                    losses = self.combined_loss(
                        x, x_hat, att_logits, labels, util_logits, task_labels, sigma
                    )
                else:
                    util_logits = self.utility_model(z_noisy)
                    l_rec = self.combined_loss.reconstruction(x, x_hat)
                    l_util = self.combined_loss.utility(util_logits, task_labels)
                    l_noise = self.combined_loss.noise_reg(sigma)
                    losses = {
                        "total": (
                            self.combined_loss.lambda_rec * l_rec
                            + self.combined_loss.lambda_util * l_util
                            + 0.01 * l_noise
                        ),
                        "reconstruction": l_rec,
                        "privacy": torch.zeros((), device=self.device),
                        "utility": l_util,
                    }
                    with torch.no_grad():
                        att_logits = self.attacker(self._attacker_input(ae_out).detach())
                    util_logits = util_logits

                self.opt_ae.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.aniae.parameters(), max_norm=1.0)
                self.opt_ae.step()

            # Record metrics
            metrics["loss_total"] += losses["total"].item()
            metrics["loss_rec"] += losses["reconstruction"].item()
            metrics["loss_priv"] += losses["privacy"].item()
            metrics["loss_util"] += losses["utility"].item()
            metrics["attacker_acc"] += (att_logits.argmax(1) == labels).float().mean().item()
            metrics["utility_acc"] += (util_logits.argmax(1) == task_labels).float().mean().item()
            metrics["mean_sigma"] += sigma.mean().item()
            metrics["n_batches"] += 1

        # Average metrics
        n = metrics.pop("n_batches")
        averaged = {k: v / n for k, v in metrics.items()}
        averaged["phase"] = phase
        return averaged

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set."""
        self.aniae.eval()
        self.attacker.eval()
        self.utility_model.eval()

        task_to_idx = {t: i for i, t in enumerate(self.config["data"]["tasks"])}
        metrics = {
            "val_loss_rec": 0, "val_attacker_acc": 0, "val_utility_acc": 0,
            "val_mean_sigma": 0, "n_batches": 0,
        }

        for x, labels, tasks in val_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            task_labels = torch.tensor(
                [task_to_idx.get(t, 0) for t in tasks], device=self.device
            )

            ae_out = self.aniae(x)
            z_noisy = ae_out["z_noisy"]

            att_logits = self.attacker(self._attacker_input(ae_out))
            util_logits = self.utility_model(z_noisy)

            rec_loss = nn.MSELoss()(ae_out["x_hat"], x)

            metrics["val_loss_rec"] += rec_loss.item()
            metrics["val_attacker_acc"] += (att_logits.argmax(1) == labels).float().mean().item()
            metrics["val_utility_acc"] += (util_logits.argmax(1) == task_labels).float().mean().item()
            metrics["val_mean_sigma"] += ae_out["sigma"].mean().item()
            metrics["n_batches"] += 1

        n = metrics.pop("n_batches")
        metrics = {k: v / n for k, v in metrics.items()}

        # Privacy-utility composite score (higher is better)
        # Low attacker acc = good privacy; high utility acc = good utility
        privacy_score = 1.0 - metrics["val_attacker_acc"]
        utility_score = metrics["val_utility_acc"]
        metrics["val_privacy_utility_score"] = 2 * privacy_score * utility_score / (
            privacy_score + utility_score + 1e-8
        )
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        # Legacy latent-attacker pretraining is only used for old configs.
        uses_v2_schedule = self.warmup_epochs > 0 or self.attacker_warmup_epochs > 0
        if (
            self.start_epoch == 0
            and self.attacker_pretrain_epochs > 0
            and not uses_v2_schedule
        ):
            self.pretrain_attacker(train_loader)
        else:
            print("Skipping attacker pretraining.")

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            phase = train_metrics.get("phase", "joint")

            # Update schedulers
            self.sched_ae.step()
            self.sched_attacker.step()

            # Logging
            print(
                f"Epoch {epoch+1}/{self.epochs} [{phase}] | "
                f"Rec={train_metrics['loss_rec']:.4f} | "
                f"AttAcc={train_metrics['attacker_acc']*100:.1f}% | "
                f"UtilAcc={train_metrics['utility_acc']*100:.1f}% | "
                f"sigma={train_metrics['mean_sigma']:.3f} | "
                f"ValScore={val_metrics['val_privacy_utility_score']:.4f}"
            )

            # Early stopping
            score = val_metrics["val_privacy_utility_score"]
            can_early_stop = phase == "joint"
            if score > self.best_score:
                self.best_score = score
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, "best")
            else:
                if can_early_stop:
                    self.patience_counter += 1
                if can_early_stop and self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.save_checkpoint(epoch, val_metrics, "latest")
                    break

            self.log_metrics(epoch, train_metrics, val_metrics)
            self.save_checkpoint(epoch, val_metrics, "latest")

            # Periodic save
            if (epoch + 1) % self.config["logging"]["save_interval"] == 0:
                self.save_checkpoint(epoch, val_metrics, f"epoch_{epoch+1}")

        print(f"Training complete. Best score: {self.best_score:.4f}")

    def _json_safe(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Append epoch metrics to a JSONL log."""
        record = {
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
        }
        path = self.output_dir / "metrics.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._json_safe(record), ensure_ascii=False) + "\n")

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str):
        """Save model checkpoint."""
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
            "num_subjects": self.num_subjects,
            "config": self.config,
            "aniae_state": self.aniae.state_dict(),
            "attacker_state": self.attacker.state_dict(),
            "utility_state": self.utility_model.state_dict(),
            "opt_ae_state": self.opt_ae.state_dict(),
            "opt_attacker_state": self.opt_attacker.state_dict(),
            "opt_utility_state": self.opt_utility.state_dict(),
            "sched_ae_state": self.sched_ae.state_dict(),
            "sched_attacker_state": self.sched_attacker.state_dict(),
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "num_subjects" in ckpt:
            self.set_num_subjects(int(ckpt["num_subjects"]))
        self.aniae.load_state_dict(ckpt["aniae_state"])
        self.attacker.load_state_dict(ckpt["attacker_state"])
        self.utility_model.load_state_dict(ckpt["utility_state"])
        if "opt_ae_state" in ckpt:
            self.opt_ae.load_state_dict(ckpt["opt_ae_state"])
        if "opt_attacker_state" in ckpt:
            self.opt_attacker.load_state_dict(ckpt["opt_attacker_state"])
        if "opt_utility_state" in ckpt:
            self.opt_utility.load_state_dict(ckpt["opt_utility_state"])
        if "sched_ae_state" in ckpt:
            self.sched_ae.load_state_dict(ckpt["sched_ae_state"])
        if "sched_attacker_state" in ckpt:
            self.sched_attacker.load_state_dict(ckpt["sched_attacker_state"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_score = float(ckpt.get("best_score", -float("inf")))
        self.patience_counter = int(ckpt.get("patience_counter", 0))
        return ckpt["metrics"]
