# Adaptive Gaze Privacy

Public artifact repository for research on adaptive latent-noise release for
biometric privacy in eye-tracking streams. The codebase supports empirical
evaluation of ANIAE, an evaluator-dependent privacy-preserving release
mechanism designed to reduce biometric identifiability while retaining
task-relevant gaze dynamics.

ANIAE should be interpreted as an empirical privacy mechanism rather than a
formal differential privacy method. Formal differential-privacy terminology in
this repository applies only to explicit differential-privacy baselines or to
cited formal mechanisms.

## Repository Scope

This repository is structured as a manuscript companion artifact. It provides
the implementation, final experiment configurations, compact result artifacts,
and provenance notes required to inspect the reported computational results.
Raw third-party data, trained checkpoints, manuscript source files, publisher
templates, and compiled manuscript files are managed outside the public code
release.

Researchers who wish to rerun the full experimental pipeline from raw gaze
recordings must obtain GazeBase directly from the dataset maintainers and place
the data under a local `data/` directory consistent with the paths specified in
the configuration files. Dataset access, storage, and redistribution remain
subject to the GazeBase license, access terms, and participant-consent
restrictions.

## Repository Organization

| Path | Purpose |
|---|---|
| `configs/` | Final ANIAE, baseline, ablation, and evaluator configurations. |
| `src/` | Data loading, preprocessing, model definitions, training losses, and evaluation metrics. |
| `scripts/` | Entry points for cache construction, training, evaluation, diagnostics, and result summarization. |
| `results/` | Compact public artifacts for seed-level outputs, summary tables, robustness checks, and diagnostic analyses. |
| `docs/` | Provenance manifest linking manuscript-level claims to public artifacts and generator scripts. |

## Primary Provenance Artifacts

The final reported ANIAE results are anchored to the subject-disjoint Round
Protocol with seeds `42`, `43`, and `44`. The final learned-release evaluation
uses stochastic sampling:

```text
ae_eval_noise_mode = sample
release_repeats = 5
```

The main result source of truth is:

```text
results/final_results_summary/seed_summary.json
```

The corresponding seed-level stochastic release artifacts are:

```text
results/final_seed_runs/stochastic_seed42.json
results/final_seed_runs/stochastic_seed43.json
results/final_seed_runs/stochastic_seed44.json
```

The final ANIAE configuration is:

```text
configs/proposed_aniae_stage2.yaml
```

Additional baseline and ablation configurations cover Fixed-Noise AE, VAE/KL
AE, GRL AE, raw biometric and task evaluators, and no-privacy-loss variants.
The historical `configs/round_protocol_ekyt_aniae.yaml` file is retained as
configuration provenance rather than as the final main-result configuration.

## Environment

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The experiments use PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib,
PyYAML, and tqdm. CUDA is recommended for full training runs; inspection,
diagnostic, and summary scripts can be executed on CPU where applicable.

## Reproducibility Workflow

Typical end-to-end execution starts by building local data caches, training the
biometric and task evaluators, training the ANIAE release model, evaluating the
protection methods, and summarizing the seed-level outputs:

```bash
python scripts/build_cache.py --config configs/proposed_aniae_stage2.yaml
python scripts/train_biometric_baseline.py --config configs/biometric_attacker_bigru.yaml --device cuda
python scripts/train_task_evaluator.py --config configs/proposed_aniae_stage2.yaml --device cuda
python scripts/train_aniae_privacy.py --config configs/proposed_aniae_stage2.yaml --device cuda --seed 42
python scripts/evaluate_protection_methods.py --config configs/proposed_aniae_stage2.yaml --device cuda
python scripts/summarize_seed_results.py --input_dir results/final_seed_runs --output_dir results/final_results_summary
```

Command arguments may need adjustment for local dataset paths, compute
resources, and experiment naming conventions. The compact JSON artifacts under
`results/` preserve the final reported values without redistributing raw gaze
recordings or local model checkpoints.

Robustness and ablation summaries can be regenerated from the public compact
artifacts with:

```bash
python scripts/summarize_robustness_checks.py
```

This command refreshes summary artifacts under `results/robustness_checks/`.
When a private manuscript directory is available locally, the corresponding
LaTeX table can also be regenerated with:

```bash
python scripts/summarize_robustness_checks.py --paper_dir path/to/local/paper
```

## Result Traceability

The artifact map in `docs/final_artifact_manifest.md` links each manuscript
table, figure, or quantitative statement to its public input artifact, generator
script, configuration, seed policy, and release mode. This manifest should be
treated as the first reference point when auditing numerical consistency between
the code repository and the manuscript.

## Distribution Boundaries

The public release is limited to original code, configurations, compact result
summaries, and documentation. Raw GazeBase recordings are governed by the
dataset maintainers and must be obtained through the appropriate data-access
process. Trained checkpoints are treated as local generated artifacts and are
excluded from version control through `.gitignore`.

Manuscript sources, compiled PDFs, publisher templates, and production figures
are maintained separately from this code artifact so that the repository remains
a clean computational companion to the paper.

## License

The original code, scripts, configurations, compact result summaries, and
project documentation in this repository are released under the Apache License
2.0. See `LICENSE`.

The Apache License 2.0 applies only to the materials distributed in this
repository. It does not grant rights to GazeBase recordings or other third-party
data assets, which remain governed by their respective access terms.
