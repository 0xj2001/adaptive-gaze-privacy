"""Compare ANIAE evaluation JSON files and recommend a validation winner."""

import argparse
import json
import math
from pathlib import Path


MIN_TASK_RETENTION = 0.922
MAX_RANK1 = 0.009524


def load_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        row["_source"] = str(path)
    return rows


def row_value(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value


def sort_key(row: dict) -> tuple:
    task_retention = row_value(row, "task_retention", 0.0)
    rank1 = row_value(row, "rank1_ir", 1.0)
    eer = row_value(row, "eer", 0.0)
    mse = row_value(row, "reconstruction_mse", math.inf)
    return (
        task_retention >= MIN_TASK_RETENTION,
        rank1 <= MAX_RANK1,
        eer,
        -mse,
        task_retention,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ANIAE tuning evaluation results")
    parser.add_argument("paths", nargs="+", help="Evaluation JSON files to compare")
    args = parser.parse_args()

    rows = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        rows.extend(load_rows(path))

    aniae_rows = [row for row in rows if row.get("method") == "aniae"]
    if not aniae_rows:
        raise SystemExit("No ANIAE rows found in the provided JSON files.")

    ranked = sorted(aniae_rows, key=sort_key, reverse=True)
    print("ANIAE candidates:")
    for row in ranked:
        print(
            f"- {row.get('parameter')} | "
            f"rank1={format_pct(row_value(row, 'rank1_ir'))} | "
            f"rank5={format_pct(row_value(row, 'rank5_ir'))} | "
            f"eer={format_pct(row_value(row, 'eer'))} | "
            f"task_retention={format_pct(row_value(row, 'task_retention'))} | "
            f"mse={row_value(row, 'reconstruction_mse'):.6f} | "
            f"source={row.get('_source')}"
        )

    winner = ranked[0]
    print("\nRecommended winner:")
    print(json.dumps({k: v for k, v in winner.items() if not k.startswith("_")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
