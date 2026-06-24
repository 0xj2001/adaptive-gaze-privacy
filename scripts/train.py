"""
Main training script.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.gazebase_loader import create_dataloaders
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train ANIAE model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--run_name", type=str, default=None, help="Optional experiment run name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root or exact run dir")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.run_name:
        output_root = Path(args.output_dir or config["logging"]["output_dir"])
        config["logging"]["output_dir"] = str(output_root / args.run_name)
    elif args.resume and args.output_dir is None:
        config["logging"]["output_dir"] = str(Path(args.resume).resolve().parent)
    elif args.output_dir:
        config["logging"]["output_dir"] = args.output_dir

    print(f"Config loaded: {args.config}")
    print(f"Device: {args.device}")
    print(f"Noise mode: {config['model']['noise']['mode']}")
    print(f"Output dir: {config['logging']['output_dir']}")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Update num_subjects based on actual data
    num_train_subjects = len(train_loader.dataset.subject_to_label)
    print(f"  Subjects (train): {num_train_subjects}")

    # Create trainer
    trainer = Trainer(config, device=args.device)

    # Override attacker's output size if needed
    if num_train_subjects != 322:
        print(f"  Adjusting attacker output to {num_train_subjects} subjects")
        trainer.set_num_subjects(num_train_subjects)

    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)

    # Final evaluation on test set
    eval_loader = test_loader
    eval_split = "test"
    if len(test_loader.dataset) == 0:
        print("\nTest set is empty; falling back to validation set for final evaluation.")
        eval_loader = val_loader
        eval_split = "validation"

    print(f"\nFinal evaluation on {eval_split} set...")
    from src.evaluation.privacy_metrics import full_evaluation
    results = full_evaluation(
        trainer.aniae, trainer.attacker, eval_loader, device=args.device
    )
    print("\n=== Test Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
