"""Summarize robustness and ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    "rank1_ir",
    "rank5_ir",
    "eer",
    "task_retention",
    "reconstruction_mse",
]


def load_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_row(rows: list[dict], method: str, parameter: str | None = None) -> dict:
    for row in rows:
        if row.get("method") != method:
            continue
        if parameter is not None and row.get("parameter") != parameter:
            continue
        return row
    raise KeyError(f"Missing row method={method!r} parameter={parameter!r}")


def summarize(records: list[dict], metrics: list[str]) -> dict:
    out = {}
    for metric in metrics:
        values = [float(record[metric]) for record in records if metric in record]
        if not values:
            continue
        out[f"{metric}_mean"] = mean(values)
        out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return out


def pct(summary: dict, key: str) -> str:
    return f"{summary[f'{key}_mean'] * 100:.2f} $\\pm$ {summary[f'{key}_std'] * 100:.2f}"


def scalar(summary: dict, key: str) -> str:
    return f"{summary[f'{key}_mean']:.5f} $\\pm$ {summary[f'{key}_std']:.5f}"


def collect_alternative_attacker(results_dir: Path, seeds: list[int]) -> list[dict]:
    specs = [
        ("No Protection", "no_protection", "none"),
        ("Fixed-Noise AE", "fixed_ae", "sigma=1.0"),
        ("Proposed ANIAE", "aniae", "proposed_aniae"),
    ]
    rows = []
    for label, method, parameter in specs:
        seed_rows = []
        for seed in seeds:
            path = results_dir / f"alternative_attacker_seed{seed}.json"
            seed_rows.append(find_row(load_rows(path), method, parameter))
        summary = summarize(seed_rows, ["rank1_ir", "rank5_ir", "eer"])
        rows.append({"experiment": "alternative_attacker", "variant": label, **summary})
    return rows


def collect_core_ablation(results_dir: Path, seed_runs_dir: Path, seeds: list[int]) -> list[dict]:
    specs = [
        ("Full Proposed ANIAE", "final_results", "aniae", "adaptive"),
        ("Fixed-Noise AE", "final_results", "fixed_ae", "sigma=1.0"),
        ("ANIAE without privacy loss", "ablation", "aniae", "without_privacy_loss"),
    ]
    rows = []
    for label, source, method, parameter in specs:
        seed_rows = []
        for seed in seeds:
            if source == "final_results":
                path = seed_runs_dir / f"stochastic_seed{seed}.json"
            else:
                path = results_dir / f"core_ablation_seed{seed}.json"
            seed_rows.append(find_row(load_rows(path), method, parameter))
        summary = summarize(seed_rows, METRICS)
        rows.append({
            "experiment": "core_ablation",
            "variant": label,
            "source": source,
            **summary,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def latex_row(label: str, summary: dict, include_task: bool) -> str:
    fields = [
        label,
        pct(summary, "rank1_ir"),
        pct(summary, "rank5_ir"),
        pct(summary, "eer"),
    ]
    if include_task:
        fields.extend([
            pct(summary, "task_retention"),
            scalar(summary, "reconstruction_mse"),
        ])
    return " & ".join(fields) + r" \\"


def compact_latex_row(label: str, summary: dict) -> str:
    fields = [
        label,
        pct(summary, "rank1_ir"),
        pct(summary, "rank5_ir"),
        pct(summary, "eer"),
        pct(summary, "task_retention"),
        scalar(summary, "reconstruction_mse"),
    ]
    return " & ".join(fields) + r" \\"


def write_latex(path: Path, alt_rows: list[dict], ablation_rows: list[dict]) -> None:
    alt_map = {row["variant"]: row for row in alt_rows}
    ablation_map = {row["variant"]: row for row in ablation_rows}
    content = r"""
\subsection{Alternative Attacker Robustness}
To test evaluator dependence, we used a separate Temporal-CNN biometric evaluator for the same enrollment/probe test. This is a post-hoc robustness analysis under a different evaluator, not an exhaustive adaptive-attacker evaluation.

Table~\ref{tab:alternative_attacker} shows that raw gaze remained highly identifiable, whereas both learned protection methods reduced biometric matchability. Proposed ANIAE stayed close to random Rank-1 identification, although Fixed-Noise AE produced a slightly higher EER under this attacker.

\begin{table}[!t]
\centering
\caption{Post-hoc robustness under a separate Temporal-CNN biometric evaluator.}
\label{tab:alternative_attacker}
\scriptsize
\begin{tabular}{lrrr}
\toprule
Method & Rank-1 IR (\%) & Rank-5 IR (\%) & EER (\%) \\
\midrule
"""[1:]
    for label in ["No Protection", "Fixed-Noise AE", "Proposed ANIAE"]:
        content += latex_row(label, alt_map[label], include_task=False) + "\n"
    content += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Core Ablation}
Table~\ref{tab:core_ablation} separates the contribution of adaptive sensitivity estimation and the biometric privacy loss. Fixed-Noise AE uses the same autoencoder backbone but removes adaptive sensitivity estimation, while the no-privacy-loss variant keeps the adaptive architecture and two-stage training but sets the biometric privacy loss weight to zero. Removing the privacy loss improves reconstruction and task retention, as expected, but it also increases Rank-1 and Rank-5 identification and lowers EER relative to the full model. This pattern supports the privacy objective, while the comparison with Fixed-Noise AE shows that ANIAE mainly improves the privacy-utility trade-off rather than every privacy metric in isolation.

\begin{table}[!t]
\centering
\caption{Core ablation results under the primary enrollment/probe evaluator. Full denotes Proposed ANIAE.}
\label{tab:core_ablation}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Variant & Rank-1 IR (\%) & Rank-5 IR (\%) & EER (\%) & Task Ret. (\%) & MSE \\
\midrule
"""
    compact_labels = [
        ("Full", "Full Proposed ANIAE"),
        ("Fixed-Noise AE", "Fixed-Noise AE"),
        ("No privacy loss", "ANIAE without privacy loss"),
    ]
    for short_label, label in compact_labels:
        content += compact_latex_row(short_label, ablation_map[label]) + "\n"
    content += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize robustness and ablation experiments")
    parser.add_argument("--results_dir", type=Path, default=Path("results/robustness_checks"))
    parser.add_argument("--seed_runs_dir", type=Path, default=Path("results/final_seed_runs"))
    parser.add_argument(
        "--paper_dir",
        type=Path,
        default=None,
        help="Optional private manuscript directory for writing robustness_ablation_tables.tex.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    alt_rows = collect_alternative_attacker(args.results_dir, args.seeds)
    ablation_rows = collect_core_ablation(args.results_dir, args.seed_runs_dir, args.seeds)
    all_rows = alt_rows + ablation_rows

    with open(args.results_dir / "robustness_checks_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    write_csv(args.results_dir / "robustness_checks_summary.csv", all_rows)
    print(f"Wrote {args.results_dir / 'robustness_checks_summary.json'}")
    print(f"Wrote {args.results_dir / 'robustness_checks_summary.csv'}")
    if args.paper_dir is not None:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        write_latex(args.paper_dir / "robustness_ablation_tables.tex", alt_rows, ablation_rows)
        print(f"Wrote {args.paper_dir / 'robustness_ablation_tables.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
