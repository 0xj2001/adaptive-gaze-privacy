"""
Run comparative experiments across all baselines.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint experiments/checkpoint_best.pt
    python scripts/evaluate.py --run_baselines  # compare all methods
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.gazebase_loader import create_dataloaders
from src.training.trainer import Trainer
from src.evaluation.privacy_metrics import full_evaluation, compute_noise_analysis


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baseline_comparison(config: dict, device: str = "cuda"):
    """
    Compare all methods:
    1. No protection (raw)
    2. Gaussian noise (various sigma)
    3. Differential Privacy (Laplace, various epsilon)
    4. Fixed-noise AE
    5. ANIAE (ours)
    """
    results = {}

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # --- Method 1: No protection ---
    config_none = deepcopy(config)
    config_none["model"]["noise"]["mode"] = "none"
    trainer_none = Trainer(config_none, device)
    # Train AE without noise (just reconstruction)
    trainer_none.config["training"]["lambda_privacy"] = 0.0
    trainer_none.train(train_loader, val_loader)
    results["no_protection"] = full_evaluation(
        trainer_none.aniae, trainer_none.attacker, test_loader, device
    )

    # --- Method 2: Fixed Gaussian noise (multiple sigma values) ---
    for sigma in [0.05, 0.1, 0.2, 0.5, 1.0]:
        config_fixed = deepcopy(config)
        config_fixed["model"]["noise"]["mode"] = "fixed"
        config_fixed["model"]["noise"]["fixed_sigma"] = sigma
        trainer_fixed = Trainer(config_fixed, device)
        trainer_fixed.train(train_loader, val_loader)
        results[f"gaussian_sigma={sigma}"] = full_evaluation(
            trainer_fixed.aniae, trainer_fixed.attacker, test_loader, device
        )

    # --- Method 3: ANIAE (ours) ---
    config_aniae = deepcopy(config)
    config_aniae["model"]["noise"]["mode"] = "adaptive"
    trainer_aniae = Trainer(config_aniae, device)
    trainer_aniae.train(train_loader, val_loader)
    results["ANIAE"] = full_evaluation(
        trainer_aniae.aniae, trainer_aniae.attacker, test_loader, device
    )

    return results


def plot_pareto_front(results: dict, output_path: str):
    """Plot privacy vs utility Pareto front."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    methods = []
    privacy_scores = []  # 1 - identification_accuracy (higher = more private)
    utility_scores = []  # signal_correlation (higher = more useful)

    for method, metrics in results.items():
        methods.append(method)
        privacy_scores.append(1.0 - metrics.get("identification_accuracy", 0))
        utility_scores.append(metrics.get("signal_correlation", 0))

    # Color coding
    colors = []
    for m in methods:
        if "ANIAE" in m:
            colors.append("red")
        elif "gaussian" in m:
            colors.append("blue")
        elif "no_protection" in m:
            colors.append("gray")
        else:
            colors.append("green")

    ax.scatter(utility_scores, privacy_scores, c=colors, s=100, zorder=5)
    for i, method in enumerate(methods):
        ax.annotate(
            method, (utility_scores[i], privacy_scores[i]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )

    ax.set_xlabel("Utility (Signal Correlation)", fontsize=12)
    ax.set_ylabel("Privacy (1 - Identification Accuracy)", fontsize=12)
    ax.set_title("Privacy-Utility Pareto Front", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Pareto front saved to: {output_path}")


def plot_noise_distribution(model, dataloader, device, output_path):
    """Visualize learned noise distribution across latent dimensions."""
    noise_info = compute_noise_analysis(model, dataloader, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Per-dimension mean sigma
    dims = range(len(noise_info["sigma_mean_per_dim"]))
    axes[0].bar(dims, noise_info["sigma_mean_per_dim"], yerr=noise_info["sigma_std_per_dim"])
    axes[0].set_xlabel("Latent Dimension")
    axes[0].set_ylabel("Mean Noise Scale (sigma)")
    axes[0].set_title("Adaptive Noise per Latent Dimension")

    # Histogram of all sigma values
    axes[1].text(
        0.5, 0.5,
        f"Global mean sigma: {noise_info['sigma_global_mean']:.4f}\n"
        f"Global std sigma: {noise_info['sigma_global_std']:.4f}",
        transform=axes[1].transAxes, ha="center", va="center", fontsize=12,
    )
    axes[1].set_title("Noise Statistics Summary")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Noise distribution saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ANIAE")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_baselines:
        print("Running baseline comparison (this may take several hours)...")
        results = run_baseline_comparison(config, args.device)

        # Save results
        import json
        with open(output_dir / "baseline_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Plot Pareto front
        plot_pareto_front(results, str(output_dir / "pareto_front.png"))

    elif args.checkpoint:
        # Evaluate single checkpoint
        _, _, test_loader = create_dataloaders(config)
        trainer = Trainer(config, args.device)
        trainer.load_checkpoint(args.checkpoint)

        results = full_evaluation(
            trainer.aniae, trainer.attacker, test_loader, args.device
        )
        print("\n=== Evaluation Results ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # Noise visualization
        plot_noise_distribution(
            trainer.aniae, test_loader, args.device,
            str(output_dir / "noise_distribution.png"),
        )
    else:
        print("Specify --checkpoint or --run_baselines")


if __name__ == "__main__":
    main()
