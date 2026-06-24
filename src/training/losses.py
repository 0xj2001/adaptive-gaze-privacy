"""
Loss functions for ANIAE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """MSE reconstruction loss with optional temporal smoothness regularization."""

    def __init__(self, smooth_weight: float = 0.01):
        super().__init__()
        self.smooth_weight = smooth_weight

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: original sequence (batch, seq_len, features)
            x_hat: reconstructed sequence (batch, seq_len, features)
        """
        mse = F.mse_loss(x_hat, x)

        # Temporal smoothness: penalize jittery reconstructions
        if self.smooth_weight > 0:
            diff_original = torch.diff(x, dim=1)
            diff_recon = torch.diff(x_hat, dim=1)
            smooth_loss = F.mse_loss(diff_recon, diff_original)
            return mse + self.smooth_weight * smooth_loss
        return mse


class PrivacyLoss(nn.Module):
    """
    Adversarial privacy loss.

    For the autoencoder: MAXIMIZE attacker's cross-entropy (confuse the attacker).
    For the attacker: MINIMIZE cross-entropy (standard classification).
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def for_autoencoder(self, attacker_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Loss for AE: we want attacker to fail → maximize CE (minimize negative CE)."""
        # Uniform target: ideal output is 1/num_classes for all classes
        num_classes = attacker_logits.shape[1]
        uniform = torch.ones_like(attacker_logits) / num_classes
        # KL divergence from uniform (attacker output should be random)
        log_probs = F.log_softmax(attacker_logits, dim=1)
        kl_from_uniform = F.kl_div(log_probs, uniform, reduction="batchmean")
        # Minimize KL from uniform = make attacker predict uniformly.
        return kl_from_uniform

    def for_attacker(self, attacker_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Loss for attacker: standard classification loss."""
        return self.ce(attacker_logits, labels)


class UtilityLoss(nn.Module):
    """Utility preservation loss: protected data should still be useful for downstream tasks."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, utility_logits: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """Standard CE for utility task."""
        return self.ce(utility_logits, task_labels)


class NoiseRegularization(nn.Module):
    """
    Regularize noise to avoid trivial solutions.

    Prevents: all sigma → 0 (no privacy) or all sigma → max (no utility).
    """

    def __init__(self, target_mean_sigma: float = 0.3):
        super().__init__()
        self.target = target_mean_sigma

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Penalize deviation of mean sigma from target."""
        mean_sigma = sigma.mean()
        return (mean_sigma - self.target) ** 2


class CombinedLoss(nn.Module):
    """Combined loss for ANIAE training."""

    def __init__(self, config: dict):
        super().__init__()
        train_cfg = config["training"]

        self.lambda_rec = train_cfg["lambda_reconstruction"]
        self.lambda_priv = train_cfg["lambda_privacy"]
        self.lambda_util = train_cfg["lambda_utility"]

        self.reconstruction = ReconstructionLoss(smooth_weight=0.01)
        self.privacy = PrivacyLoss()
        self.utility = UtilityLoss()
        self.noise_reg = NoiseRegularization(target_mean_sigma=0.3)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        attacker_logits: torch.Tensor,
        identity_labels: torch.Tensor,
        utility_logits: torch.Tensor,
        task_labels: torch.Tensor,
        sigma: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses for the autoencoder."""
        l_rec = self.reconstruction(x, x_hat)
        l_priv = self.privacy.for_autoencoder(attacker_logits, identity_labels)
        l_util = self.utility(utility_logits, task_labels)
        l_noise = self.noise_reg(sigma)

        total = (
            self.lambda_rec * l_rec
            + self.lambda_priv * l_priv
            + self.lambda_util * l_util
            + 0.01 * l_noise
        )

        return {
            "total": total,
            "reconstruction": l_rec,
            "privacy": l_priv,
            "utility": l_util,
            "noise_reg": l_noise,
        }
