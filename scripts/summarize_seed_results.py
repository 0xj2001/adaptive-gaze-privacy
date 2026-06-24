"""Summarize 3-seed protection results into publication tables and plots."""

import argparse
import csv
import html
import json
import math
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

_mpl_cache = Path("experiments/round_protocol_results/.matplotlib-cache")
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache.resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_DISPLAY = {
    "no_protection": "No Protection",
    "gaussian": "Gaussian Noise",
    "dp": "Differential Privacy",
    "fixed_ae": "Fixed-Noise AE",
    "vae_ae": "VAE/KL AE",
    "grl_ae": "GRL Adversarial AE",
    "aniae": "Proposed ANIAE",
}


MAIN_TRADEOFF_POINTS = {
    ("No Protection", "-"),
    ("Gaussian Noise", "sigma=0.05"),
    ("Differential Privacy", "epsilon=20"),
    ("Fixed-Noise AE", "sigma=1.0"),
    ("VAE/KL AE", "beta=0.01"),
    ("GRL Adversarial AE", "adversarial"),
    ("Proposed ANIAE", "Ours"),
}

METRICS = [
    "rank1_ir",
    "rank5_ir",
    "eer",
    "auc",
    "tar_at_far_1pct",
    "tar_at_far_5pct",
    "task_accuracy",
    "task_macro_f1",
    "task_retention",
    "reconstruction_mse",
    "reconstruction_mae",
    "signal_correlation",
]

PERCENT_METRICS = {
    "rank1_ir",
    "rank5_ir",
    "eer",
    "auc",
    "tar_at_far_1pct",
    "tar_at_far_5pct",
    "task_accuracy",
    "task_macro_f1",
    "task_retention",
    "signal_correlation",
}


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("results", payload.get("rows", []))
        for row in payload:
            row = dict(row)
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def display_parameter(row: dict) -> str:
    method = row.get("method")
    parameter = str(row.get("parameter", ""))
    if method == "no_protection":
        return "-"
    if method == "aniae":
        return "Ours"
    return parameter


def bootstrap_ci(values: list[float], repeats: int = 10000, seed: int = 2026) -> tuple[float, float]:
    clean = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(repeats, clean.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=1)) if clean.size > 1 else 0.0


def fmt_metric(mean: float, std: float, percent: bool = True) -> str:
    if math.isnan(mean):
        return "n/a"
    scale = 100.0 if percent else 1.0
    decimals = 2 if percent else 5
    return f"{mean * scale:.{decimals}f} ± {std * scale:.{decimals}f}"


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("method"), display_parameter(row))
        grouped[key].append(row)

    summary = []
    for (method, parameter), group_rows in sorted(grouped.items(), key=lambda item: (METHOD_DISPLAY.get(item[0][0], item[0][0]), item[0][1])):
        out = {
            "Method": METHOD_DISPLAY.get(method, method),
            "Parameter": parameter,
            "n_seeds": len({row.get("seed") for row in group_rows}),
        }
        for metric in METRICS:
            values = [float(row.get(metric, float("nan"))) for row in group_rows]
            mean, std = mean_std(values)
            ci_low, ci_high = bootstrap_ci(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci95_low"] = ci_low
            out[f"{metric}_ci95_high"] = ci_high
            percent_metric = metric in PERCENT_METRICS
            out[f"{metric}_mean_std"] = fmt_metric(mean, std, percent=percent_metric)
            ci_scale = 100.0 if percent_metric else 1.0
            ci_decimals = 2 if percent_metric else 5
            out[f"{metric}_ci95"] = (
                "n/a"
                if math.isnan(ci_low)
                else f"[{ci_low * ci_scale:.{ci_decimals}f}, {ci_high * ci_scale:.{ci_decimals}f}]"
            )
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    columns = [
        ("Method", "Method"),
        ("Parameter", "Parameter"),
        ("Rank-1 IR", "rank1_ir_mean_std"),
        ("Rank-5 IR", "rank5_ir_mean_std"),
        ("EER", "eer_mean_std"),
        ("AUC", "auc_mean_std"),
        ("TAR@FAR=1%", "tar_at_far_1pct_mean_std"),
        ("TAR@FAR=5%", "tar_at_far_5pct_mean_std"),
        ("Task Acc.", "task_accuracy_mean_std"),
        ("Task Macro-F1", "task_macro_f1_mean_std"),
        ("Task Retention", "task_retention_mean_std"),
        ("MSE", "reconstruction_mse_mean_std"),
        ("MAE", "reconstruction_mae_mean_std"),
        ("Signal Corr.", "signal_correlation_mean_std"),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(label for label, _ in columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |\n")


def excel_col(index: int) -> str:
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_xml(row_idx: int, col_idx: int, value) -> str:
    ref = f"{excel_col(col_idx)}{row_idx}"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{float(value)}</v></c>'
    text = "" if value is None else html.escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_excel(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    sheet_rows = [headers] + [[row.get(header, "") for header in headers] for row in rows]
    sheet_xml_rows = []
    for row_idx, values in enumerate(sheet_rows, start=1):
        cells = "".join(cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(values))
        sheet_xml_rows.append(f'<row r="{row_idx}">{cells}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(sheet_xml_rows)
        + '</sheetData></worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="seed_summary" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def plot_tradeoff(path: Path, rows: list[dict], *, main_only: bool = True) -> None:
    if not rows:
        return
    if main_only:
        rows = [row for row in rows if (row["Method"], row["Parameter"]) in MAIN_TRADEOFF_POINTS]

    styles = {
        "No Protection": {"color": "#d55e00", "marker": "*", "size": 120},
        "Gaussian Noise": {"color": "#0072b2", "marker": "o", "size": 50},
        "Differential Privacy": {"color": "#cc79a7", "marker": "s", "size": 50},
        "Fixed-Noise AE": {"color": "#e69f00", "marker": "^", "size": 60},
        "VAE/KL AE": {"color": "#56b4e9", "marker": "v", "size": 60},
        "GRL Adversarial AE": {"color": "#000000", "marker": "P", "size": 60},
        "Proposed ANIAE": {"color": "#009e73", "marker": "D", "size": 74},
    }
    main_legend_labels = {
        "Differential Privacy": "Laplace DP",
        "GRL Adversarial AE": "GRL AE",
    }

    if main_only:
        plot_rows = [row for row in rows if row["Method"] != "No Protection"]
        fig, ax = plt.subplots(figsize=(3.55, 2.85), dpi=240)
        legend_methods = []
        for row in plot_rows:
            x = row.get("rank1_ir_mean", float("nan")) * 100.0
            y = row.get("task_retention_mean", float("nan")) * 100.0
            xerr = row.get("rank1_ir_std", 0.0) * 100.0
            yerr = row.get("task_retention_std", 0.0) * 100.0
            if math.isnan(x) or math.isnan(y):
                continue

            method = row["Method"]
            style = styles.get(method, {"color": "#555555", "marker": "o", "size": 50})
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=0.85,
                capsize=2.2,
                markersize=math.sqrt(style["size"]),
                alpha=0.92,
                label=method,
            )
            if method not in legend_methods:
                legend_methods.append(method)

        ax.set_xlim(-0.35, 5.5)
        ax.set_ylim(15, 104)
        ax.set_xlabel("Rank-1 Identification Rate (%)")
        ax.set_ylabel("Task Retention (%)")
        ax.grid(True, alpha=0.25)
        handles = [
            ax.scatter(
                [],
                [],
                marker=styles[method]["marker"],
                s=styles[method]["size"],
                color=styles[method]["color"],
                alpha=0.92,
            )
            for method in legend_methods
        ]
        fig.legend(
            handles,
            [main_legend_labels.get(label, label) for label in legend_methods],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
            fontsize=6.5,
            columnspacing=0.95,
            handletextpad=0.45,
        )
        fig.tight_layout(rect=(0, 0.20, 1, 1))
        fig.savefig(path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=180, sharey=True)
    legend_handles = {}

    def draw(ax, zoom: bool) -> None:
        for row in rows:
            x = row.get("rank1_ir_mean", float("nan")) * 100.0
            y = row.get("task_retention_mean", float("nan")) * 100.0
            xerr = row.get("rank1_ir_std", 0.0) * 100.0
            yerr = row.get("task_retention_std", 0.0) * 100.0
            if math.isnan(x) or math.isnan(y):
                continue

            method = row["Method"]
            style = styles.get(method, {"color": "#555555", "marker": "o", "size": 50})
            handle = ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=0.7,
                capsize=2,
                markersize=math.sqrt(style["size"]),
                alpha=0.88,
                label=method,
            )
            legend_handles.setdefault(method, handle)

            if method == "No Protection" and not zoom:
                ax.annotate("No Protection", (x, y), xytext=(-68, -12), textcoords="offset points", fontsize=8)
            if method == "Proposed ANIAE":
                ax.annotate("Proposed ANIAE", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=8)

        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Rank-1 Identification Rate (%)")
        if zoom:
            ax.set_title("Protected-region zoom")
            ax.set_xlim(-0.4, 5.5)
        else:
            ax.set_title("Full range")
            ax.set_xlim(-3, 100)
        ax.set_ylim(15, 104)

    draw(axes[0], zoom=False)
    draw(axes[1], zoom=True)
    axes[0].set_ylabel("Task Retention (%)")
    title = "Representative Privacy-Utility Trade-off Across Three Seeds" if main_only else "Full Privacy-Utility Parameter Sweep"
    fig.suptitle(title, y=0.99)
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_seed_stability(path: Path, rows: list[dict]) -> None:
    proposed = [row for row in rows if row.get("method") == "aniae"]
    if not proposed:
        return
    seeds = [str(row.get("seed")) for row in proposed]
    rank1 = [float(row.get("rank1_ir", float("nan"))) * 100.0 for row in proposed]
    task = [float(row.get("task_retention", float("nan"))) * 100.0 for row in proposed]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(seeds, rank1, marker="o", label="Rank-1 IR")
    ax.plot(seeds, task, marker="s", label="Task Retention")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Metric (%)")
    ax.set_title("Proposed ANIAE Seed Stability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize seed-level protection results")
    parser.add_argument("result_files", nargs="*", type=Path)
    parser.add_argument("--input_dir", type=Path, default=Path("experiments/round_protocol_results/seed_runs"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/round_protocol_results/publication_summary"))
    args = parser.parse_args()

    result_files = args.result_files or sorted(
        path for path in args.input_dir.glob("*.json")
        if not path.name.lower().startswith("smoke")
    )
    if not result_files:
        raise SystemExit(f"No result JSON files found in {args.input_dir}")

    rows = load_rows(result_files)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(args.output_dir / "seed_summary.csv", summary)
    write_markdown(args.output_dir / "seed_summary.md", summary)
    write_excel(args.output_dir / "seed_summary.xlsx", summary)
    plot_tradeoff(args.output_dir / "privacy_utility_tradeoff.png", summary, main_only=True)
    plot_tradeoff(args.output_dir / "privacy_utility_full_sweep.png", summary, main_only=False)
    plot_seed_stability(args.output_dir / "seed_stability.png", rows)
    print(f"Summary written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
