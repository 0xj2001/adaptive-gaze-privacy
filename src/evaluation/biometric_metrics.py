"""Enrollment/probe biometric evaluation for gaze embeddings."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

from src.data.round_protocol import load_recording_windows


@torch.no_grad()
def embed_recording(
    model: torch.nn.Module,
    cache_dir: str | Path,
    record: dict,
    device: torch.device | str,
    batch_size: int = 128,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Average chunk embeddings for one cached recording."""
    device = torch.device(device)
    x = load_recording_windows(cache_dir, record)
    embeddings = []
    model.eval()
    for start in range(0, x.size(0), batch_size):
        batch = x[start:start + batch_size].to(device)
        if transform is not None:
            batch = transform(batch)
        emb = model.embed(batch) if hasattr(model, "embed") else model(batch)[0]
        embeddings.append(emb.detach().cpu())
    if not embeddings:
        raise ValueError(f"No chunks for recording {record.get('cache_file')}")
    emb = torch.cat(embeddings, dim=0).mean(dim=0)
    return F.normalize(emb, p=2, dim=0)


def _select_enrollment_probe(
    records: list[dict],
    task: str = "RAN",
    enrollment_session: int = 1,
    probe_session: int = 2,
) -> tuple[dict[int, dict], dict[int, dict]]:
    enrollment = {}
    probe = {}
    for rec in sorted(records, key=lambda r: (r["person_id"], r["session"], r["file"])):
        if rec["task"] != task:
            continue
        person_id = int(rec["person_id"])
        session = int(rec["session"])
        if session == enrollment_session and person_id not in enrollment:
            enrollment[person_id] = rec
        elif session == probe_session and person_id not in probe:
            probe[person_id] = rec
    people = sorted(set(enrollment) & set(probe))
    return {p: enrollment[p] for p in people}, {p: probe[p] for p in people}


def _eer_from_similarity(similarity: torch.Tensor) -> float:
    n = similarity.shape[0]
    scores = []
    labels = []
    for i in range(n):
        for j in range(n):
            scores.append(float(similarity[i, j].item()))
            labels.append(1 if i == j else 0)
    if len(set(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def similarity_score_labels(similarity: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Flatten an enrollment/probe similarity matrix into verification scores."""
    n = similarity.shape[0]
    scores = []
    labels = []
    for i in range(n):
        for j in range(n):
            scores.append(float(similarity[i, j].item()))
            labels.append(1 if i == j else 0)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32)


def metrics_from_similarity(similarity: torch.Tensor) -> dict[str, float]:
    """Compute enrollment/probe identification and verification metrics."""
    sorted_idx = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(similarity.shape[0])
    rank1 = (sorted_idx[:, 0] == targets).float().mean().item()
    rank5 = (sorted_idx[:, :min(5, similarity.shape[0])] == targets[:, None]).any(dim=1).float().mean().item()
    ranks = []
    for row, target in zip(sorted_idx, targets):
        ranks.append(int((row == target).nonzero(as_tuple=False)[0].item()) + 1)
    genuine_scores = similarity.diag()
    impostor_scores = similarity[~torch.eye(similarity.shape[0], dtype=torch.bool)]
    return {
        "num_eval_subjects": similarity.shape[0],
        "rank1_ir": rank1,
        "rank5_ir": rank5,
        "eer": _eer_from_similarity(similarity),
        "random_rank1": 1.0 / similarity.shape[0],
        "random_rank5": min(5, similarity.shape[0]) / similarity.shape[0],
        "mean_rank": float(np.mean(ranks)),
        "genuine_score_mean": float(genuine_scores.mean().item()),
        "impostor_score_mean": float(impostor_scores.mean().item()),
    }


@torch.no_grad()
def enrollment_probe_details(
    model: torch.nn.Module,
    cache_dir: str | Path,
    records: list[dict],
    device: torch.device | str,
    task: str = "RAN",
    enrollment_session: int = 1,
    probe_session: int = 2,
    batch_size: int = 128,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """Return embeddings, similarity matrix, ranks, and score labels for publication metrics."""
    enrollment_records, probe_records = _select_enrollment_probe(
        records, task, enrollment_session, probe_session
    )
    people = sorted(set(enrollment_records) & set(probe_records))
    if not people:
        raise ValueError("No enrollment/probe pairs found")

    enroll_emb = torch.stack([
        embed_recording(model, cache_dir, enrollment_records[p], device, batch_size, transform)
        for p in people
    ])
    probe_emb = torch.stack([
        embed_recording(model, cache_dir, probe_records[p], device, batch_size, transform)
        for p in people
    ])
    similarity = probe_emb @ enroll_emb.t()
    sorted_idx = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(len(people))
    ranks = []
    for row, target in zip(sorted_idx, targets):
        ranks.append(int((row == target).nonzero(as_tuple=False)[0].item()) + 1)
    scores, labels = similarity_score_labels(similarity)
    return {
        "people": people,
        "similarity": similarity,
        "ranks": np.asarray(ranks, dtype=np.int32),
        "scores": scores,
        "labels": labels,
        "metrics": metrics_from_similarity(similarity),
    }


@torch.no_grad()
def evaluate_enrollment_probe(
    model: torch.nn.Module,
    cache_dir: str | Path,
    records: list[dict],
    device: torch.device | str,
    task: str = "RAN",
    enrollment_session: int = 1,
    probe_session: int = 2,
    batch_size: int = 128,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Evaluate rank-k identification and verification EER."""
    return enrollment_probe_details(
        model,
        cache_dir,
        records,
        device,
        task=task,
        enrollment_session=enrollment_session,
        probe_session=probe_session,
        batch_size=batch_size,
        transform=transform,
    )["metrics"]
