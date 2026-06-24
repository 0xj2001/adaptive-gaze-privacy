"""
完整测试脚本：使用仿真数据验证整个pipeline

运行步骤:
1. 生成仿真数据
2. 加载并预处理
3. 训练ANIAE模型（少量epoch）
4. 评估隐私和效用
5. 生成报告

Usage:
    python scripts/test_pipeline.py
"""

import multiprocessing
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 70)
    print("ANIAE Pipeline Test with Synthetic Data")
    print("=" * 70)
    print()

    # Step 1: Generate synthetic data
    print("[1/5] Generating synthetic data...")
    from generate_synthetic_data import generate_synthetic_gazebase

    data_dir = project_root / "data" / "synthetic_gazebase"
    if not (data_dir / "Round_1").exists():
        generate_synthetic_gazebase(
            output_dir=str(data_dir),
            num_subjects=30,  # 少量被试快速测试
            num_rounds=2,
            tasks=["FXS", "HSS", "RAN", "TEX"],
        )
        print("✓ Synthetic data generated\n")
    else:
        print("✓ Synthetic data already exists\n")

    # Step 2: Test data loading
    print("[2/5] Testing data loader...")
    try:
        import torch
        import yaml

        from src.data.gazebase_loader import create_dataloaders

        with open(project_root / "configs" / "test.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        train_loader, val_loader, test_loader = create_dataloaders(config)

        print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        print(f"  Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")

        # Test one batch
        x, labels, tasks = next(iter(train_loader))
        print(f"  Batch shape: {x.shape}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        print(f"  Tasks: {set(tasks)}")
        print("✓ Data loader working\n")

    except Exception as e:
        print(f"✗ Data loader failed: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 3: Test model forward pass
    print("[3/5] Testing model forward pass...")
    try:
        from src.models.attacker import PrivacyAttacker, UtilityModel
        from src.models.autoencoder import ANIAE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")

        aniae = ANIAE(config).to(device)
        num_subjects = len(train_loader.dataset.subject_to_label)
        latent_dim = config["model"]["encoder"]["latent_dim"]

        attacker = PrivacyAttacker(
            input_dim=latent_dim,
            num_subjects=num_subjects,
            hidden_dims=config["model"]["attacker"]["hidden_dims"],
            dropout=config["model"]["attacker"]["dropout"],
            input_type="latent",
        ).to(device)

        utility_model = UtilityModel(
            input_dim=latent_dim,
            num_classes=len(config["data"]["tasks"]),
            hidden_dims=config["model"]["utility"]["hidden_dims"],
            dropout=config["model"]["utility"]["dropout"],
            input_type="latent",
        ).to(device)

        # Forward pass test
        x_test = x[:4].to(device)
        with torch.no_grad():
            ae_out = aniae(x_test)
            att_out = attacker(ae_out["z_noisy"])
            util_out = utility_model(ae_out["z_noisy"])

        print(f"  Input shape: {x_test.shape}")
        print(f"  Reconstructed shape: {ae_out['x_hat'].shape}")
        print(f"  Latent shape: {ae_out['z'].shape}")
        print(f"  Noisy latent shape: {ae_out['z_noisy'].shape}")
        print(f"  Sigma shape: {ae_out['sigma'].shape}")
        print(f"  Sigma mean: {ae_out['sigma'].mean().item():.4f}")
        print(f"  Attacker output shape: {att_out.shape}")
        print(f"  Utility output shape: {util_out.shape}")
        print("✓ Model forward pass working\n")

    except Exception as e:
        print(f"✗ Model test failed: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 4: Quick training test (3 epochs)
    print("[4/5] Running quick training test (3 epochs)...")
    try:
        from src.training.trainer import Trainer

        # Override config for quick test
        config["training"]["epochs"] = 3
        config["training"]["attacker_pretrain_epochs"] = 1
        config["training"]["patience"] = 5

        trainer = Trainer(config, device=str(device))

        # Adjust attacker output size
        trainer.attacker = PrivacyAttacker(
            input_dim=latent_dim,
            num_subjects=num_subjects,
            hidden_dims=config["model"]["attacker"]["hidden_dims"],
            dropout=config["model"]["attacker"]["dropout"],
            input_type="latent",
        ).to(device)

        print("  Starting training...")
        trainer.train(train_loader, val_loader)
        print("✓ Training completed\n")

    except Exception as e:
        print(f"✗ Training failed: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 5: Evaluation
    print("[5/5] Running evaluation...")
    try:
        from src.evaluation.privacy_metrics import full_evaluation

        results = full_evaluation(
            trainer.aniae, trainer.attacker, test_loader, device=str(device)
        )

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print("\nPrivacy Metrics (lower = more private):")
        print(f"  Identification Accuracy: {results['identification_accuracy']*100:.2f}%")
        print(f"  Equal Error Rate (EER): {results['equal_error_rate']:.4f}")
        print(f"  Random Baseline: {results['random_baseline']*100:.2f}%")

        print("\nUtility Metrics (higher = better):")
        print(f"  Signal Correlation: {results['signal_correlation']:.4f}")
        print(f"  Reconstruction MSE: {results['reconstruction_mse']:.6f}")

        print("\nNoise Analysis:")
        print(f"  Mean Sigma: {results['sigma_global_mean']:.4f}")

        print("\n✓ All tests passed!")
        print("\n" + "=" * 70)
        print("Pipeline validated successfully with synthetic data.")
        print("You can now proceed to train on real GazeBase data:")
        print("  python scripts/train.py --config configs/default.yaml")
        print("=" * 70)

    except Exception as e:
        print(f"✗ Evaluation failed: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
