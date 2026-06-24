"""EKYT-style biometric and task evaluators."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer1D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.cat([x, out], dim=1)


class DenseBlock1D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer1D(channels, growth_rate, dropout))
            channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Transition1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseSequenceEncoder(nn.Module):
    """Compact DenseNet-style 1D encoder for gaze sequences."""

    def __init__(
        self,
        input_dim: int = 2,
        base_channels: int = 64,
        growth_rate: int = 16,
        block_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        block_layers = block_layers or [4, 4, 4]
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        channels = base_channels
        blocks = []
        for idx, num_layers in enumerate(block_layers):
            block = DenseBlock1D(channels, growth_rate, num_layers, dropout)
            blocks.append(block)
            channels = block.out_channels
            if idx != len(block_layers) - 1:
                out_channels = max(base_channels, channels // 2)
                blocks.append(Transition1D(channels, out_channels))
                channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, features) -> (batch, features, seq_len)
        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.blocks(h)
        return self.pool(h).squeeze(-1)


class GazeBiometricEmbeddingNet(nn.Module):
    """Biometric evaluator that maps gaze chunks to normalized embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_subjects: int,
        embedding_dim: int = 128,
        base_channels: int = 64,
        growth_rate: int = 16,
        block_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = DenseSequenceEncoder(
            input_dim=input_dim,
            base_channels=base_channels,
            growth_rate=growth_rate,
            block_layers=block_layers,
            dropout=dropout,
        )
        self.embedding = nn.Linear(self.encoder.out_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_subjects)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        embedding = F.normalize(self.embedding(features), p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class BiGRUBiometricEmbeddingNet(nn.Module):
    """Independent recurrent biometric evaluator for robustness checks."""

    def __init__(
        self,
        input_dim: int,
        num_subjects: int,
        embedding_dim: int = 128,
        conv_channels: int = 64,
        gru_hidden_dim: int = 128,
        gru_layers: int = 2,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.embedding = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_subjects)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        h = self.stem(h).transpose(1, 2)
        h, _ = self.gru(h)
        features = h.mean(dim=1)
        embedding = F.normalize(self.embedding(features), p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class TemporalResidualBlock1D(nn.Module):
    """Residual dilated temporal block used by the alternative CNN attacker."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalCNNBiometricEmbeddingNet(nn.Module):
    """Fast independent temporal-CNN biometric evaluator for robustness checks."""

    def __init__(
        self,
        input_dim: int,
        num_subjects: int,
        embedding_dim: int = 128,
        channels: int = 128,
        num_blocks: int = 6,
        dilations: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16, 32]
        if len(dilations) < num_blocks:
            repeats = (num_blocks + len(dilations) - 1) // len(dilations)
            dilations = (dilations * repeats)[:num_blocks]
        else:
            dilations = dilations[:num_blocks]
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, stride=2, padding=2, bias=False),
        )
        self.blocks = nn.Sequential(*[
            TemporalResidualBlock1D(channels, dilation=dilation, dropout=dropout)
            for dilation in dilations
        ])
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )
        self.embedding = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_subjects)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        h = self.blocks(self.stem(h))
        weights = torch.softmax(self.attention(h), dim=-1)
        mean = (h * weights).sum(dim=-1)
        second = (h.square() * weights).sum(dim=-1)
        std = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        embedding = F.normalize(self.embedding(torch.cat([mean, std], dim=1)), p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


def build_biometric_model(config: dict, num_subjects: int) -> nn.Module:
    """Build a biometric embedding evaluator from config."""
    data_cfg = config["data"]
    model_cfg = config["biometric_model"]
    model_type = str(model_cfg.get("type", "densenet")).lower()
    input_dim = len(data_cfg["features"])
    if model_type in {"densenet", "dense", "ekyt"}:
        return GazeBiometricEmbeddingNet(
            input_dim=input_dim,
            num_subjects=num_subjects,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            growth_rate=int(model_cfg.get("growth_rate", 16)),
            block_layers=model_cfg.get("block_layers", [4, 4, 4]),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if model_type in {"bigru", "gru"}:
        return BiGRUBiometricEmbeddingNet(
            input_dim=input_dim,
            num_subjects=num_subjects,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            conv_channels=int(model_cfg.get("conv_channels", 64)),
            gru_hidden_dim=int(model_cfg.get("gru_hidden_dim", 128)),
            gru_layers=int(model_cfg.get("gru_layers", 2)),
            classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.25)),
        )
    if model_type in {"temporal_cnn", "tcn", "resnet1d"}:
        return TemporalCNNBiometricEmbeddingNet(
            input_dim=input_dim,
            num_subjects=num_subjects,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            channels=int(model_cfg.get("channels", 128)),
            num_blocks=int(model_cfg.get("num_blocks", 6)),
            dilations=model_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown biometric_model.type={model_type!r}")


class GazeTaskEvaluator(nn.Module):
    """Frozen utility evaluator for FXS/HSS/RAN/TEX task retention."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dim: int = 128,
        base_channels: int = 64,
        growth_rate: int = 16,
        block_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = DenseSequenceEncoder(
            input_dim=input_dim,
            base_channels=base_channels,
            growth_rate=growth_rate,
            block_layers=block_layers or [3, 3, 3],
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
