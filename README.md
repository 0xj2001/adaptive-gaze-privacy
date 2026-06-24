# Adaptive Gaze Privacy

This repository contains the public code package for the accompanying manuscript on adaptive latent-noise release for biometric privacy in eye-tracking streams. The package is intentionally venue-neutral: it contains the code, configurations, compact result artifacts, and provenance notes needed to trace the reported results, but it does not include the manuscript source/PDF, the original GazeBase dataset, or trained model checkpoints.

ANIAE is an empirical, evaluator-dependent privacy-preserving release mechanism. It is not a formal differential privacy mechanism; formal privacy wording applies only to explicit differential-privacy baselines or cited mechanisms.

## Repository Layout

```text
configs/   Final manuscript and baseline configurations
src/       Data loading, preprocessing, models, losses, and evaluation metrics
scripts/   Training, evaluation, diagnostics, and table/figure summary scripts
results/   Compact final result summaries and seed-level JSON artifacts
docs/      Artifact manifest and reproducibility notes
```

## What Is Included

- Final main-paper ANIAE configuration: `configs/proposed_aniae_stage2.yaml`
- Historical/alternate configuration retained for provenance: `configs/round_protocol_ekyt_aniae.yaml`
- Baseline configurations for Fixed-Noise AE, VAE/KL AE, GRL AE, raw biometric/task evaluators, and no-privacy ablations
- Final result source of truth: `results/final_results_summary/seed_summary.json`
- Final stochastic seed-level outputs: `results/final_seed_runs/stochastic_seed42.json`, `stochastic_seed43.json`, and `stochastic_seed44.json`

## What Is Not Included

- Manuscript source files, manuscript PDF files, publisher templates, and paper figures
- The original GazeBase recordings or any full dataset copy
- Trained checkpoint files (`*.pt`, `*.pth`, `*.ckpt`)
- Local virtual environments, caches, temporary files, logs, and toolchains
- Additional draft materials or internal notes

To reproduce from raw data, obtain GazeBase from the dataset maintainers and place it under a local `data/` directory according to the paths in the configs. Dataset access and redistribution must follow the dataset license, access terms, and participant-consent restrictions.

## Environment

Install Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The experiments use PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, PyYAML, and tqdm. CUDA is recommended for training, but some inspection and summary scripts run on CPU.

## Reproducing the Main Result Chain

The final manuscript numbers are based on a subject-disjoint Round Protocol with seeds `42`, `43`, and `44`. The final learned-release evaluation mode is stochastic sampling:

```text
ae_eval_noise_mode = sample
release_repeats = 5
```

The final main-table source of truth is:

```text
results/final_results_summary/seed_summary.json
```

The corresponding seed-level stochastic release files are:

```text
results/final_seed_runs/stochastic_seed42.json
results/final_seed_runs/stochastic_seed43.json
results/final_seed_runs/stochastic_seed44.json
```

Robustness and ablation summaries can be regenerated from the compact public JSON artifacts with:

```bash
python scripts/summarize_robustness_checks.py
```

This writes refreshed robustness-check summaries under `results/robustness_checks/`. If you keep a private local manuscript directory and want to regenerate the corresponding LaTeX table, pass `--paper_dir path/to/local/paper`.

## Training and Evaluation Entry Points

Common entry points are:

```bash
python scripts/build_cache.py --config configs/proposed_aniae_stage2.yaml
python scripts/train_biometric_baseline.py --config configs/biometric_attacker_bigru.yaml --device cuda
python scripts/train_task_evaluator.py --config configs/proposed_aniae_stage2.yaml --device cuda
python scripts/train_aniae_privacy.py --config configs/proposed_aniae_stage2.yaml --device cuda --seed 42
python scripts/evaluate_protection_methods.py --config configs/proposed_aniae_stage2.yaml --device cuda
python scripts/summarize_seed_results.py --input_dir results/final_seed_runs --output_dir results/final_results_summary
```

Exact command arguments may need to be adjusted for local data paths and hardware. The compact JSON artifacts in `results/` document the final reported values without requiring the raw dataset to be redistributed.

## Manuscript Integration

The public repository intentionally excludes manuscript source files, compiled PDFs, publisher templates, and paper figures. The compact artifacts in `results/` and the mapping in `docs/final_artifact_manifest.md` are provided so the reported manuscript values can be traced without redistributing the manuscript files themselves.

## Artifact Manifest

See `docs/final_artifact_manifest.md` for a table mapping each main-paper table or figure to its compact input artifact, generator script, config, seed policy, and release mode.

## License

The original code, scripts, configurations, compact result summaries, and project documentation in this repository are released under the Apache License 2.0. See `LICENSE`.

This license does not grant rights to the original GazeBase recordings, which are not redistributed here and remain governed by the dataset maintainers' access terms. It also does not apply to manuscript files, publisher templates, or paper figures, which are intentionally not included in the public repository.
