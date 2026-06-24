"""Metric-learning losses for biometric embeddings."""

import torch
import torch.nn as nn


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss for normalized embeddings.

    This implementation keeps the core behavior used by biometric embedding
    training: positives are same-person chunks and negatives are different-person
    chunks within the PK batch.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0, base: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = embeddings @ embeddings.t()
        losses = []
        for idx in range(embeddings.size(0)):
            label = labels[idx]
            pos_mask = labels == label
            pos_mask[idx] = False
            neg_mask = labels != label

            pos_pair = sim[idx][pos_mask]
            neg_pair = sim[idx][neg_mask]
            if pos_pair.numel() == 0 or neg_pair.numel() == 0:
                continue

            pos_loss = torch.log1p(torch.exp(-self.alpha * (pos_pair - self.base)).sum()) / self.alpha
            neg_loss = torch.log1p(torch.exp(self.beta * (neg_pair - self.base)).sum()) / self.beta
            losses.append(pos_loss + neg_loss)

        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()
