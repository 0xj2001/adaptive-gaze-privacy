# Final Artifact Manifest

This manifest records the result provenance for the manuscript and the public code package. It uses venue-neutral names so that the repository can be shared before a final submission venue is chosen. The public repository intentionally excludes the manuscript source, compiled manuscript PDF, publisher templates, and paper figures.

## Final Source of Truth

- Final ANIAE config for the main-paper results: `configs/proposed_aniae_stage2.yaml`
- Historical or alternate config not used as the final main-paper ANIAE config: `configs/round_protocol_ekyt_aniae.yaml`
- Main table source of truth: `results/final_results_summary/seed_summary.json`
- Final per-seed stochastic release rows:
  - `results/final_seed_runs/stochastic_seed42.json`
  - `results/final_seed_runs/stochastic_seed43.json`
  - `results/final_seed_runs/stochastic_seed44.json`
- Final seeds: `42`, `43`, `44`
- Final learned-release evaluation mode: `ae_eval_noise_mode=sample`
- Final stochastic release repeats: `release_repeats=5`

The public package does not include manuscript files or model checkpoints. Training metadata for the final ANIAE runs is retained under `results/final_training_metadata/` for provenance.

## Main-Paper Result Map

| Reported manuscript item | Public input artifact(s) | Generator / code path | Notes |
|---|---|---|---|
| Representative privacy-utility rows | `results/final_results_summary/seed_summary.json`; `results/final_seed_runs/stochastic_seed*.json` | `scripts/evaluate_protection_methods.py`; `scripts/evaluate_publication_metrics.py`; `scripts/summarize_seed_results.py` | Source of truth for No Protection, Gaussian, Laplace DP, Fixed-Noise AE, VAE/KL AE, GRL AE, and Proposed ANIAE rows. |
| Compact full sweep | `results/final_results_summary/seed_summary.json` | `scripts/evaluate_publication_metrics.py`; `scripts/summarize_seed_results.py` | Compact sweep reported in the main paper. |
| Seed stability paragraph | `results/final_seed_runs/stochastic_seed42.json`; `stochastic_seed43.json`; `stochastic_seed44.json` | `scripts/evaluate_protection_methods.py` | Per-seed ANIAE Rank-1 IR and task retention. |
| Privacy-utility figure | `results/final_results_summary/seed_summary.json` | `scripts/evaluate_publication_metrics.py`; `scripts/summarize_seed_results.py` | Uses final summary rows. |
| Evaluation diagnostics figure and AUC text | Final seed-run JSON rows with score details; `results/final_results_summary/seed_summary.json` | `scripts/evaluate_publication_metrics.py`; `src/evaluation/biometric_metrics.py` | ROC/DET/AUC are based on enrollment/probe score arrays when present. |
| Semantic retention results | `results/semantic_retention/semantic_retention_summary.json` | `scripts/evaluate_semantic_retention_benchmark.py` | Source artifact for the manuscript semantic-retention table and figure. |
| Semantic utility results | `results/semantic_utility/non_ran_summary.json`; `results/semantic_utility/four_class_summary.json` | `scripts/evaluate_semantic_utility_probes.py` | Source artifact for the manuscript semantic-utility table. |
| Adaptive sigma figure/text | `results/adaptive_sigma/adaptive_sigma_by_task.json` | `scripts/analyze_adaptive_sigma.py` | Reports pooled task-level mean sigma and coefficient of variation. |
| Sigma-only side-channel table | `results/sigma_leakage/test_sigma_leakage_summary.json` | `scripts/evaluate_sigma_leakage.py` | Diagnostic only; sigma is internal state and is not a released signal. |
| Alternative attacker table | `results/robustness_checks/alternative_attacker_seed42/43/44.json`; `results/robustness_checks/robustness_checks_summary.json` | `scripts/summarize_robustness_checks.py` | Post-hoc robustness under a separate Temporal-CNN evaluator. |
| Core ablation table | Full/Fixed rows from `results/final_seed_runs/stochastic_seed*.json`; no-privacy-loss rows from `results/robustness_checks/core_ablation_seed42/43/44.json` | `scripts/summarize_robustness_checks.py` | Mixed source is intentional: Full and Fixed match the final stochastic manuscript source of truth; no-privacy-loss remains an ablation. |

## Regeneration Notes

1. Regenerate final seed-level summaries only from the final stochastic seed JSON files before updating any main-paper result table.
2. Regenerate robustness and ablation summaries with:

   ```bash
   python scripts/summarize_robustness_checks.py
   ```

3. If you keep a private local manuscript directory, regenerate the corresponding LaTeX table with:

   ```bash
   python scripts/summarize_robustness_checks.py --paper_dir path/to/local/paper
   ```

4. Confirm that the final core-ablation rows remain:
   - Full Proposed ANIAE: Rank-1 IR `0.89 +/- 0.29`, Rank-5 IR `7.24 +/- 1.65`, EER `46.33 +/- 1.33`, task retention `91.54 +/- 2.03`, MSE `0.02379 +/- 0.00427`
   - Fixed-Noise AE: Rank-1 IR `1.21 +/- 0.40`, Rank-5 IR `6.92 +/- 0.90`, EER `47.11 +/- 0.41`, task retention `88.97 +/- 0.83`, MSE `0.03592 +/- 0.00240`

5. Manuscript compilation is outside this public repository.

## Boundary Statement

The manuscript presents ANIAE as an empirical, evaluator-dependent privacy-preserving release mechanism for biometric eye-tracking streams. The learned ANIAE model is not a formal differential privacy mechanism. Formal differential-privacy wording applies only to the Laplace baseline or to cited formal mechanisms, not to ANIAE.
