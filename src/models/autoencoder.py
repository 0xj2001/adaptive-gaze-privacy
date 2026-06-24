"""
Adaptive Noise-Infused Autoencoder (ANIAE).

Core architecture:
    Input → Encoder → Latent z → SensitivityEstimator → adaptive σ
                                → z + σ·ε → Decoder → Protected output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Temporal encoder using 1D convolutions + GRU."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list[int] = None,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        # 1D Conv layers for local pattern extraction
        layers = []
        in_ch = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_ch, h_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = h_dim
        self.conv = nn.Sequential(*layers)

        # GRU for temporal dependencies
        self.gru = nn.GRU(
            input_size=hidden_dims[-1],
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Project bidirectional output to latent dim
        self.proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            z: (batch, latent_dim) - sequence-level latent representation
        """
        # Conv expects (batch, channels, seq_len)
        h = x.transpose(1, 2)
        h = self.conv(h)

        # GRU expects (batch, seq_len, features)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)

        # Take mean over time for sequence-level representation
        z = h.mean(dim=1)
        z = self.proj(z)
        return z


class Decoder(nn.Module):
    """Temporal decoder: latent → reconstructed sequence."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: list[int] = None,
        output_dim: int = 5,
        seq_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 64, 128]
        self.seq_len = seq_len

        # Expand latent to sequence
        self.expand = nn.Linear(latent_dim, hidden_dims[0] * seq_len)

        # Transposed conv layers
        layers = []
        in_ch = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.extend([
                nn.Conv1d(in_ch, h_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = h_dim

        # Final projection to output dim
        layers.append(nn.Conv1d(in_ch, output_dim, kernel_size=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_hat: (batch, seq_len, output_dim)
        """
        h = self.expand(z)
        h = h.view(z.size(0), -1, self.seq_len)  # (batch, hidden, seq_len)
        h = self.deconv(h)
        x_hat = h.transpose(1, 2)  # (batch, seq_len, output_dim)
        return x_hat


class SensitivityEstimator(nn.Module):
    """
    Estimates per-dimension noise scale σ_i for each latent dimension.

    Key idea: learns which latent dimensions carry identity information
    and assigns higher noise to those dimensions.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: list[int] = None,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [16, 16]
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            sigma: (batch, latent_dim) - noise scale per dimension
        """
        raw = self.net(z)
        # Sigmoid to bound between [sigma_min, sigma_max]
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)
        return sigma


class AdaptiveNoiseInjector(nn.Module):
    """Inject adaptive noise into latent space."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], sigma_min: float, sigma_max: float):
        super().__init__()
        self.sensitivity = SensitivityEstimator(latent_dim, hidden_dims, sigma_min, sigma_max)

    def forward(
        self,
        z: torch.Tensor,
        training: bool = True,
        eval_noise_mode: str = "deterministic",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, latent_dim)
            training: if False, use eval_noise_mode
            eval_noise_mode: "deterministic" keeps the historical evaluation
                release, "sample" samples a fresh release, and "none" disables
                noise for diagnostic ablations.
        Returns:
            z_noisy: (batch, latent_dim)
            sigma: (batch, latent_dim) - for logging/regularization
        """
        sigma = self.sensitivity(z)

        if training or eval_noise_mode == "sample":
            eps = torch.randn_like(z)
            z_noisy = z + sigma * eps
        elif eval_noise_mode == "none":
            z_noisy = z
        else:
            # Historical deterministic evaluation path retained for checkpoint
            # compatibility and sensitivity analyses.
            z_noisy = z + sigma * 0.5  # half noise for deterministic output
        return z_noisy, sigma


class ANIAE(nn.Module):
    """
    Adaptive Noise-Infused Autoencoder.

    Full pipeline: x → Encoder → z → AdaptiveNoise → z̃ → Decoder → x̃
    """

    def __init__(self, config: dict):
        super().__init__()
        enc_cfg = config["model"]["encoder"]
        dec_cfg = config["model"]["decoder"]
        noise_cfg = config["model"]["noise"]
        data_cfg = config["data"]

        self.noise_mode = noise_cfg["mode"]
        self.fixed_sigma = noise_cfg.get("fixed_sigma", 0.1)
        self.eval_noise_mode = noise_cfg.get("eval_noise_mode", "deterministic")

        self.encoder = Encoder(
            input_dim=enc_cfg["input_dim"],
            hidden_dims=enc_cfg["hidden_dims"],
            latent_dim=enc_cfg["latent_dim"],
            dropout=enc_cfg["dropout"],
        )

        self.decoder = Decoder(
            latent_dim=enc_cfg["latent_dim"],
            hidden_dims=dec_cfg["hidden_dims"],
            output_dim=dec_cfg["output_dim"],
            seq_len=data_cfg["window_size"],
            dropout=0.1,
        )

        latent_dim = enc_cfg["latent_dim"]

        if self.noise_mode == "adaptive":
            self.noise_injector = AdaptiveNoiseInjector(
                latent_dim=latent_dim,
                hidden_dims=noise_cfg["sensitivity_hidden"],
                sigma_min=noise_cfg["sigma_min"],
                sigma_max=noise_cfg["sigma_max"],
            )
        else:
            self.noise_injector = None

        if self.noise_mode == "vae":
            self.vae_mu = nn.Linear(latent_dim, latent_dim)
            self.vae_logvar = nn.Linear(latent_dim, latent_dim)
        else:
            self.vae_mu = None
            self.vae_logvar = None

    def set_eval_noise_mode(self, mode: str) -> None:
        """Set the release behavior used while the module is in eval mode."""
        valid = {"deterministic", "sample", "none"}
        if mode not in valid:
            raise ValueError(f"eval_noise_mode must be one of {sorted(valid)}, got {mode!r}")
        self.eval_noise_mode = mode

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def inject_noise(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply noise to latent representation."""
        if self.noise_mode == "adaptive":
            return self.noise_injector(
                z,
                training=self.training,
                eval_noise_mode=self.eval_noise_mode,
            )
        elif self.noise_mode == "fixed":
            sigma = torch.full_like(z, self.fixed_sigma)
            if self.training or self.eval_noise_mode == "sample":
                eps = torch.randn_like(z)
                z_noisy = z + sigma * eps
            else:
                z_noisy = z
            return z_noisy, sigma
        elif self.noise_mode == "vae":
            if self.vae_mu is None or self.vae_logvar is None:
                raise RuntimeError("VAE mode requires vae_mu and vae_logvar heads")
            mu = self.vae_mu(z)
            logvar = self.vae_logvar(z).clamp(min=-8.0, max=8.0)
            std = torch.exp(0.5 * logvar)
            if self.training or self.eval_noise_mode == "sample":
                eps = torch.randn_like(std)
                z_noisy = mu + std * eps
            else:
                z_noisy = mu
            return z_noisy, std
        else:  # "none"
            return z, torch.zeros_like(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstructed sequence."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with:
            x_hat: reconstructed (protected) sequence
            z: original latent
            z_noisy: noisy latent
            sigma: noise scales per dimension
        """
        z = self.encode(x)
        z_noisy, sigma = self.inject_noise(z)
        x_hat = self.decode(z_noisy)

        out = {
            "x_hat": x_hat,
            "z": z,
            "z_noisy": z_noisy,
            "sigma": sigma,
        }
        if self.noise_mode == "vae":
            mu = self.vae_mu(z)
            logvar = self.vae_logvar(z).clamp(min=-8.0, max=8.0)
            out["mu"] = mu
            out["logvar"] = logvar
            out["latent_kl"] = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()
        return out
