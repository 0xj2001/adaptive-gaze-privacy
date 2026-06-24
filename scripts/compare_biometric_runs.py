"""Compare biometric evaluator runs and recommend the best checkpoint."""

import argparse
import json
from pathlib import Path


def load_result(run_dir: Path) -> dict:
    result_path = run_dir / "test_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing test_results.json: {result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    checkpoint_path = run_dir / "checkpoint_best.pt"
    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "rank1_ir": float(metrics.get("rank1_ir", 0.0)),
        "rank5_ir": float(metrics.get("rank5_ir", 0.0)),
        "eer": float(metrics.get("eer", 1.0)),
        "mean_rank": float(metrics.get("mean_rank", 0.0)),
        "num_eval_subjects": int(metrics.get("num_eval_subjects", 0)),
    }


def is_better(candidate: dict, incumbent: dict | None, rank1_tolerance: float) -> bool:
    if incumbent is None:
        return True
    rank1_gap = candidate["rank1_ir"] - incumbent["rank1_ir"]
    if abs(rank1_gap) >= rank1_tolerance:
        return rank1_gap > 0
    eer_gap = candidate["eer"] - incumbent["eer"]
    if abs(eer_gap) > 1e-12:
        return eer_gap < 0
    return candidate["rank5_ir"] > incumbent["rank5_ir"]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Compare biometric evaluator runs")
    parser.add_argument("run_dirs", nargs="+", help="Experiment run directories")
    parser.add_argument(
        "--rank1_tolerance",
        type=float,
        default=0.02,
        help="Rank-1 gap treated as close before using EER tie-break",
    )
    args = parser.parse_args()

    rows = [load_result(Path(path)) for path in args.run_dirs]
    winner = None
    for row in rows:
        if is_better(row, winner, args.rank1_tolerance):
            winner = row

    sorted_rows = sorted(rows, key=lambda r: (-r["rank1_ir"], r["eer"], -r["rank5_ir"]))
    print("Biometric evaluator comparison")
    print(
        f"{'run':45} {'rank1':>8} {'rank5':>8} {'eer':>8} "
        f"{'mean_rank':>10} {'subjects':>9} {'checkpoint':>12}"
    )
    for row in sorted_rows:
        checkpoint_status = "yes" if row["checkpoint"].exists() else "missing"
        print(
            f"{str(row['run_dir'])[:45]:45} "
            f"{pct(row['rank1_ir']):>8} "
            f"{pct(row['rank5_ir']):>8} "
            f"{pct(row['eer']):>8} "
            f"{row['mean_rank']:10.3f} "
            f"{row['num_eval_subjects']:9d} "
            f"{checkpoint_status:>12}"
        )

    print("\nRecommended run:")
    print(f"  {winner['run_dir']}")
    print("Recommended checkpoint:")
    print(f"  {winner['checkpoint']}")


if __name__ == "__main__":
    main()
