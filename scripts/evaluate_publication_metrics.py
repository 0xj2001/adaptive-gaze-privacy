"""Create publication tables and diagnostic figures from seed-level result JSON files."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

_mpl_cache = project_root / "experiments" / "round_protocol_results" / ".matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache.resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.summarize_seed_results import (  # noqa: E402
    METHOD_DISPLAY,
    display_parameter,
    load_rows,
    plot_seed_stability,
    plot_tradeoff,
    summarize,
    write_csv,
    write_excel,
    write_markdown,
)

MAIN_VERIFICATION_POINTS = {
    ("no_protection", "-"): "No Protection",
    ("gaussian", "sigma=0.05"): "Gaussian Noise",
    ("dp", "epsilon=20"): "Laplace DP",
    ("fixed_ae", "sigma=1.0"): "Fixed-Noise AE",
    ("vae_ae", "beta=0.01"): "VAE/KL AE",
    ("grl_ae", "adversarial"): "GRL Adversarial AE",
    ("aniae", "Ours"): "Proposed ANIAE",
}

VERIFICATION_ORDER = [
    "No Protection",
    "Gaussian Noise",
    "Laplace DP",
    "Fixed-Noise AE",
    "VAE/KL AE",
    "GRL Adversarial AE",
    "Proposed ANIAE",
]

VERIFICATION_COLORS = {
    "No Protection": "#d55e00",
    "Gaussian Noise": "#0072b2",
    "Laplace DP": "#cc79a7",
    "Fixed-Noise AE": "#e69f00",
    "VAE/KL AE": "#56b4e9",
    "GRL Adversarial AE": "#000000",
    "Proposed ANIAE": "#009e73",
}

VERIFICATION_SHORT_LABELS = {
    "No Protection": "No\nProtection",
    "Gaussian Noise": "Gaussian\nNoise",
    "Laplace DP": "Laplace\nDP",
    "Fixed-Noise AE": "Fixed-Noise\nAE",
    "VAE/KL AE": "VAE/KL\nAE",
    "GRL Adversarial AE": "GRL\nAE",
    "Proposed ANIAE": "Proposed\nANIAE",
}

SEPARATION_SHORT_LABELS = {
    "No Protection": "Raw",
    "Gaussian Noise": "Gauss.",
    "Laplace DP": "DP",
    "Fixed-Noise AE": "Fixed",
    "VAE/KL AE": "VAE",
    "GRL Adversarial AE": "GRL",
    "Proposed ANIAE": "ANIAE",
}


def group_key(row: dict) -> tuple[str, str]:
    return METHOD_DISPLAY.get(row.get("method"), row.get("method")), display_parameter(row)


def verification_display_label(row: dict) -> str | None:
    return MAIN_VERIFICATION_POINTS.get((row.get("method"), display_parameter(row)))


def write_summary(output_dir: Path, rows: list[dict]) -> list[dict]:
    summary = summarize(rows)
    with open(output_dir / "seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / "seed_summary.csv", summary)
    write_markdown(output_dir / "seed_summary.md", summary)
    write_excel(output_dir / "seed_summary.xlsx", summary)
    plot_tradeoff(output_dir / "privacy_utility_tradeoff.png", summary)
    plot_seed_stability(output_dir / "seed_stability.png", rows)
    return summary


def rows_with_scores(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if "biometric_scores" in row and "biometric_labels" in row
    ]


def representative_score_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows_with_scores(rows):
        label = verification_display_label(row)
        if label:
            grouped[label].append(row)
    return grouped


def plot_roc(path: Path, rows: list[dict]) -> None:
    scored = rows_with_scores(rows)
    if not scored:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=170)
    used_labels = set()
    for row in scored:
        labels = np.asarray(row["biometric_labels"], dtype=int)
        scores = np.asarray(row["biometric_scores"], dtype=float)
        if len(set(labels.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        label = " ".join(part for part in group_key(row) if part not in ("-", "Ours"))
        if row.get("method") == "aniae":
            label = "Proposed ANIAE"
        if label in used_labels:
            ax.plot(fpr, tpr, linewidth=1.0, alpha=0.18)
        else:
            ax.plot(fpr, tpr, linewidth=1.6, alpha=0.85, label=label)
            used_labels.add(label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Acceptance Rate")
    ax.set_title("ROC Curves Across Seed Runs")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_det(path: Path, rows: list[dict]) -> None:
    scored = rows_with_scores(rows)
    if not scored:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=170)
    used_labels = set()
    for row in scored:
        labels = np.asarray(row["biometric_labels"], dtype=int)
        scores = np.asarray(row["biometric_scores"], dtype=float)
        if len(set(labels.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        label = " ".join(part for part in group_key(row) if part not in ("-", "Ours"))
        if row.get("method") == "aniae":
            label = "Proposed ANIAE"
        if label in used_labels:
            ax.plot(np.maximum(fpr, 1e-4), np.maximum(fnr, 1e-4), linewidth=1.0, alpha=0.18)
        else:
            ax.plot(np.maximum(fpr, 1e-4), np.maximum(fnr, 1e-4), linewidth=1.6, alpha=0.85, label=label)
            used_labels.add(label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("False Rejection Rate")
    ax.set_title("DET Curves Across Seed Runs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_score_distribution(path: Path, rows: list[dict]) -> None:
    scored = rows_with_scores(rows)
    if not scored:
        return
    grouped = defaultdict(lambda: {"genuine": [], "impostor": []})
    for row in scored:
        label = group_key(row)
        scores = np.asarray(row["biometric_scores"], dtype=float)
        labels = np.asarray(row["biometric_labels"], dtype=int)
        grouped[label]["genuine"].extend(scores[labels == 1].tolist())
        grouped[label]["impostor"].extend(scores[labels == 0].tolist())
    labels = []
    data = []
    for key, values in grouped.items():
        method, parameter = key
        display = method if parameter in ("-", "Ours") else f"{method}\n{parameter}"
        labels.extend([f"{display}\nGenuine", f"{display}\nImpostor"])
        data.extend([values["genuine"], values["impostor"]])
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.45), 5), dpi=170)
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Enrollment/Probe Score Distributions")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_verification_summary(path: Path, rows: list[dict]) -> None:
    grouped = representative_score_rows(rows)
    if not grouped:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        dpi=220,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    roc_ax, score_ax = axes
    fpr_grid = np.linspace(0.0, 1.0, 201)

    for label in VERIFICATION_ORDER:
        rows_for_label = grouped.get(label, [])
        if not rows_for_label:
            continue
        curves = []
        for row in rows_for_label:
            labels = np.asarray(row["biometric_labels"], dtype=int)
            scores = np.asarray(row["biometric_scores"], dtype=float)
            if len(set(labels.tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            curves.append(np.interp(fpr_grid, fpr, tpr))
        if not curves:
            continue
        curve_array = np.vstack(curves)
        mean_curve = curve_array.mean(axis=0)
        std_curve = curve_array.std(axis=0)
        color = VERIFICATION_COLORS[label]
        roc_ax.plot(fpr_grid, mean_curve, color=color, linewidth=1.8, label=label)
        if len(curves) > 1:
            roc_ax.fill_between(
                fpr_grid,
                np.clip(mean_curve - std_curve, 0.0, 1.0),
                np.clip(mean_curve + std_curve, 0.0, 1.0),
                color=color,
                alpha=0.10,
                linewidth=0,
            )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=0.9, alpha=0.55)
    roc_ax.set_xlabel("False acceptance rate")
    roc_ax.set_ylabel("True acceptance rate")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.grid(True, alpha=0.25)
    roc_ax.set_title("(a) ROC", loc="left", fontsize=9, pad=3)
    roc_ax.legend(fontsize=7.0, frameon=False, loc="lower right")

    box_data = []
    positions = []
    centers = []
    xlabels = []
    pos = 1.0
    for label in VERIFICATION_ORDER:
        rows_for_label = grouped.get(label, [])
        if not rows_for_label:
            continue
        genuine = []
        impostor = []
        for row in rows_for_label:
            scores = np.asarray(row["biometric_scores"], dtype=float)
            labels = np.asarray(row["biometric_labels"], dtype=int)
            genuine.extend(scores[labels == 1].tolist())
            impostor.extend(scores[labels == 0].tolist())
        box_data.extend([genuine, impostor])
        positions.extend([pos, pos + 0.55])
        centers.append(pos + 0.275)
        xlabels.append(VERIFICATION_SHORT_LABELS[label])
        pos += 1.55

    boxes = score_ax.boxplot(
        box_data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for idx, box in enumerate(boxes["boxes"]):
        if idx % 2 == 0:
            box.set_facecolor("#56b4e9")
            box.set_alpha(0.60)
        else:
            box.set_facecolor("#999999")
            box.set_alpha(0.45)
    score_ax.set_xticks(centers, xlabels)
    score_ax.set_ylabel("Cosine similarity")
    score_ax.grid(True, axis="y", alpha=0.25)
    score_ax.set_title("(b) Scores", loc="left", fontsize=9, pad=3)
    score_ax.tick_params(axis="x", labelsize=7.0)
    score_ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#56b4e9", alpha=0.60, edgecolor="black", label="Genuine"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#999999", alpha=0.45, edgecolor="black", label="Impostor"),
        ],
        frameon=False,
        fontsize=7.0,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.13),
        borderaxespad=0.0,
    )

    fig.tight_layout(w_pad=1.2)
    fig.savefig(path)
    plt.close(fig)


def averaged_task_confusion(rows: list[dict], task_labels: list[str]) -> tuple[np.ndarray, list[str]] | None:
    candidates = [row for row in rows if row.get("method") == "aniae" and row.get("task_confusion_matrix")]
    if not candidates:
        candidates = [row for row in rows if row.get("task_confusion_matrix")]
    if not candidates:
        return None
    matrices = [np.asarray(row["task_confusion_matrix"], dtype=float) for row in candidates]
    matrix = sum(matrices) / len(matrices)
    if len(task_labels) != matrix.shape[0]:
        task_labels = task_labels[:matrix.shape[0]] or [str(idx) for idx in range(matrix.shape[0])]
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, np.maximum(row_sums, 1.0))
    return norm * 100.0, task_labels


def plot_evaluation_diagnostics(path: Path, rows: list[dict], task_labels: list[str]) -> None:
    grouped = representative_score_rows(rows)
    confusion = averaged_task_confusion(rows, task_labels)
    if not grouped or confusion is None:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.0),
        dpi=240,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    roc_ax, task_ax = axes
    fpr_grid = np.linspace(0.0, 1.0, 201)

    for label in VERIFICATION_ORDER:
        rows_for_label = grouped.get(label, [])
        if not rows_for_label:
            continue
        curves = []
        for row in rows_for_label:
            labels = np.asarray(row["biometric_labels"], dtype=int)
            scores = np.asarray(row["biometric_scores"], dtype=float)
            if len(set(labels.tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            curves.append(np.interp(fpr_grid, fpr, tpr))
        if not curves:
            continue
        curve_array = np.vstack(curves)
        mean_curve = curve_array.mean(axis=0)
        std_curve = curve_array.std(axis=0)
        color = VERIFICATION_COLORS[label]
        roc_ax.plot(fpr_grid, mean_curve, color=color, linewidth=1.5, label=label)
        if len(curves) > 1:
            roc_ax.fill_between(
                fpr_grid,
                np.clip(mean_curve - std_curve, 0.0, 1.0),
                np.clip(mean_curve + std_curve, 0.0, 1.0),
                color=color,
                alpha=0.09,
                linewidth=0,
            )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=0.8, alpha=0.55)
    roc_ax.set_title("(a) ROC", loc="left", fontsize=8.5, pad=3)
    roc_ax.set_xlabel("False acceptance rate", fontsize=8)
    roc_ax.set_ylabel("True acceptance rate", fontsize=8)
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.grid(True, alpha=0.25)
    roc_ax.tick_params(axis="both", labelsize=7)
    roc_ax.legend(fontsize=5.8, frameon=False, loc="lower right", handlelength=1.4)

    percent, task_labels = confusion
    im = task_ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    task_ax.set_title("(b) Task utility", loc="left", fontsize=8.5, pad=3)
    task_ax.set_xticks(range(len(task_labels)), task_labels, fontsize=7.3)
    task_ax.set_yticks(range(len(task_labels)), task_labels, fontsize=7.3)
    task_ax.set_xlabel("Predicted", fontsize=8)
    task_ax.set_ylabel("True", fontsize=8)
    for i in range(percent.shape[0]):
        for j in range(percent.shape[1]):
            value = percent[i, j]
            task_ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if value > 55 else "black",
            )
    cbar = fig.colorbar(im, ax=task_ax, fraction=0.046, pad=0.03)
    cbar.set_label("%", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    fig.tight_layout(w_pad=0.85)
    fig.savefig(path)
    plt.close(fig)


def plot_task_confusion(path: Path, rows: list[dict], task_labels: list[str]) -> None:
    confusion = averaged_task_confusion(rows, task_labels)
    if confusion is None:
        return
    percent, task_labels = confusion
    fig, ax = plt.subplots(figsize=(4.8, 4.15), dpi=200)
    im = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(task_labels)), task_labels)
    ax.set_yticks(range(len(task_labels)), task_labels)
    ax.set_xlabel("Predicted Task")
    ax.set_ylabel("True Task")
    ax.tick_params(axis="both", labelsize=9)
    for i in range(percent.shape[0]):
        for j in range(percent.shape[1]):
            value = percent[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if value > 55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized percentage (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_metric_notes(path: Path) -> None:
    notes = {
        "rank1_ir": "Probability that the correct enrollment identity is the top match for a probe.",
        "rank5_ir": "Probability that the correct identity appears in the top five enrollment matches.",
        "eer": "Equal-error rate where false acceptance and false rejection are approximately equal.",
        "auc": "Area under the verification ROC curve.",
        "tar_at_far_1pct": "True acceptance rate when false acceptance rate is constrained to 1%.",
        "tar_at_far_5pct": "True acceptance rate when false acceptance rate is constrained to 5%.",
        "task_accuracy": "Accuracy of the frozen task evaluator on protected gaze chunks.",
        "task_macro_f1": "Unweighted mean F1 score across FXS, HSS, RAN, and TEX.",
        "task_retention": "Protected task accuracy divided by raw-data task accuracy.",
        "reconstruction_mse": "Mean squared difference between protected and raw gaze signals.",
        "reconstruction_mae": "Mean absolute difference between protected and raw gaze signals.",
        "signal_correlation": "Average correlation between protected and raw flattened gaze chunks.",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create publication metrics and figures")
    parser.add_argument("result_files", nargs="*", type=Path)
    parser.add_argument("--input_dir", type=Path, default=Path("experiments/round_protocol_results/seed_runs"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/round_protocol_results/publication_summary"))
    parser.add_argument("--task_labels", nargs="*", default=["FXS", "HSS", "RAN", "TEX"])
    args = parser.parse_args()

    result_files = args.result_files or sorted(
        path for path in args.input_dir.glob("*.json")
        if not path.name.lower().startswith("smoke")
    )
    if not result_files:
        raise SystemExit(f"No result JSON files found in {args.input_dir}")
    rows = load_rows(result_files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir, rows)
    plot_roc(args.output_dir / "roc_curve.png", rows)
    plot_det(args.output_dir / "det_curve.png", rows)
    plot_score_distribution(args.output_dir / "score_distribution.png", rows)
    plot_verification_summary(args.output_dir / "verification_summary.png", rows)
    plot_evaluation_diagnostics(args.output_dir / "evaluation_diagnostics.png", rows, args.task_labels)
    plot_task_confusion(args.output_dir / "task_confusion_matrix.png", rows, args.task_labels)
    write_metric_notes(args.output_dir / "metric_definitions.json")
    print(f"Publication metrics written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
