"""
Inspect the subject-disjoint GazeBase round protocol.

Example:
    python scripts/inspect_round_protocol.py --config configs/round_protocol_ekyt_aniae.yaml
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.gazebase_loader import apply_round_protocol, discover_recordings


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_report(recordings: list[dict], split_name: str) -> dict:
    split_records = [r for r in recordings if r.get("split_name") == split_name]
    people = {int(r["person_id"]) for r in split_records}
    rounds = {int(r["round_id"]) for r in split_records}
    task_counts = Counter(r["task"] for r in split_records)
    session_counts = Counter(int(r["session"]) for r in split_records)

    ran_by_person: dict[int, set[int]] = defaultdict(set)
    for rec in split_records:
        if rec["task"] == "RAN":
            ran_by_person[int(rec["person_id"])].add(int(rec["session"]))
    ran_s1_s2_people = {
        person_id for person_id, sessions in ran_by_person.items() if {1, 2}.issubset(sessions)
    }

    return {
        "subjects": len(people),
        "recordings": len(split_records),
        "rounds": sorted(rounds),
        "tasks": dict(sorted(task_counts.items())),
        "sessions": {str(k): v for k, v in sorted(session_counts.items())},
        "ran_s1_s2_subjects": len(ran_s1_s2_people),
        "ran_s1_s2_coverage": len(ran_s1_s2_people) / max(1, len(people)),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect GazeBase round protocol")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    recordings = discover_recordings(data_cfg["data_dir"], data_cfg["tasks"])
    protocol_records, protocol_summary = apply_round_protocol(recordings)

    split_subjects = {
        split: {int(r["person_id"]) for r in protocol_records if r.get("split_name") == split}
        for split in ["train", "val", "test"]
    }
    overlap = {
        "train_val": len(split_subjects["train"] & split_subjects["val"]),
        "train_test": len(split_subjects["train"] & split_subjects["test"]),
        "val_test": len(split_subjects["val"] & split_subjects["test"]),
    }

    report = {
        "config": args.config,
        "data_dir": data_cfg["data_dir"],
        "protocol_summary": protocol_summary,
        "splits": {
            split: split_report(protocol_records, split)
            for split in ["train", "val", "test"]
        },
        "subject_overlap": overlap,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = []
    expected = data_cfg.get("round_protocol", {}).get("expected_subjects", {})
    for split, expected_count in expected.items():
        actual = report["splits"][split]["subjects"]
        if int(expected_count) != actual:
            failures.append(f"{split}_subjects={actual} expected={expected_count}")

    expected_rounds = {"train": [1], "val": [2], "test": [3]}
    for split, expected_split_rounds in expected_rounds.items():
        actual_rounds = report["splits"][split]["rounds"]
        if actual_rounds != expected_split_rounds:
            failures.append(f"{split}_rounds={actual_rounds} expected={expected_split_rounds}")

    if any(overlap.values()):
        failures.append(f"subject_overlap={overlap}")

    for split in ["train", "val", "test"]:
        split_info = report["splits"][split]
        if split_info["ran_s1_s2_subjects"] != split_info["subjects"]:
            failures.append(
                f"{split} RAN S1/S2 coverage "
                f"{split_info['ran_s1_s2_subjects']}/{split_info['subjects']}"
            )

    if failures:
        raise SystemExit("Round protocol inspection failed: " + "; ".join(failures))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
