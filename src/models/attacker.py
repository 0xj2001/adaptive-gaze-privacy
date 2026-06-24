"""
Privacy attacker and utility models.

Attacker: tries to identify the person from protected gaze data.
Utility Model: evaluates if protected data retains task-relevant information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivacyAttacker(nn.Module):
    """
    Identity classifier (attacker).

    Given protected gaze data or latent features, predict subject identity.
    Used adversarially: if attacker succeeds, privacy is insufficient.
    """

    def __init__(
        self,
        input_dim: int,
        num_subjects: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
        input_type: str = "latent",
    ):
        """
        Args:
            input_dim: Dimension of input (latent_dim or seq_len * n_features).
            num_subjects: Number of identity classes.
            hidden_dims: MLP hidden layer sizes.
            dropout: Dropout rate.
            input_type: "latent" (operates on z) or "sequence" (operates on x_hat).
        """
        super().__init__()
        hidden_dims = hidden_dims or [64, 128, 64]
        self.input_type = input_type

        # If operating on sequences, use the same feature extractor for raw
        # identity baselines and protected-output attackers.
        if input_type == "sequence":
            self.feature_extractor = nn.ModuleDict({
                "conv": nn.Sequential(
                    nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                ),
                "gru": nn.GRU(
                    input_size=128,
                    hidden_size=128,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                ),
            })
            mlp_input = 256
        elif input_type == "sequence_light":
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            mlp_input = 64
        else:
            self.feature_extractor = None
            mlp_input = input_dim

        # MLP classifier
        layers = []
        in_dim = mlp_input
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_subjects))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, latent_dim) if input_type="latent"
               or (batch, seq_len, n_features) if input_type="sequence"
        Returns:
            logits: (batch, num_subjects)
        """
        if self.input_type == "sequence":
            # (batch, seq_len, features) -> (batch, features, seq_len)
            h = x.transpose(1, 2)
            h = self.feature_extractor["conv"](h)
            h = h.transpose(1, 2)
            h, _ = self.feature_extractor["gru"](h)
            h = h.mean(dim=1)
        elif self.input_type == "sequence_light":
            h = x.transpose(1, 2)
            h = self.feature_extractor(h).squeeze(-1)
        else:
            h = x

        return self.classifier(h)


class UtilityModel(nn.Module):
    """
    Downstream task model to evaluate data utility after protection.

    Example tasks:
    - Gaze event classification (fixation vs saccade vs smooth pursuit)
    - Task identification (which visual task was the user performing)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
        input_type: str = "latent",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.input_type = input_type

        if input_type == "sequence":
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            mlp_input = 32
        else:
            self.feature_extractor = None
            mlp_input = input_dim

        layers = []
        in_dim = mlp_input
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, latent_dim) or (batch, seq_len, n_features)
        Returns:
            logits: (batch, num_classes)
        """
        if self.input_type == "sequence":
            h = x.transpose(1, 2)
            h = self.feature_extractor(h).squeeze(-1)
        else:
            h = x
        return self.classifier(h)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal (for adversarial privacy training)."""
    return GradientReversalLayer.apply(x, alpha)
