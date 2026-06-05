"""Format fixed LIBERO baseline summaries as a paper-style result table."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any


SUITE_COLUMNS = (
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long"),
)


def _load_summary(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).expanduser().open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"Expected summary JSON object at {path}")
    return summary


def _suite_rates(summary: dict[str, Any]) -> dict[str, float]:
    suite_results = summary.get("suite_results", [])
    if not isinstance(suite_results, list):
        raise ValueError("Expected summary['suite_results'] to be a list")

    rates: dict[str, float] = {}
    for result in suite_results:
        if not isinstance(result, dict):
            continue
        suite_name = str(result.get("task_suite_name", ""))
        if "success_rate" in result:
            rates[suite_name] = float(result["success_rate"])
    return rates


def _average_rate(summary: dict[str, Any], rates: dict[str, float]) -> float:
    if "average_success_rate" in summary:
        return float(summary["average_success_rate"])
    values = [rates[key] for key, _ in SUITE_COLUMNS if key in rates]
    if not values:
        raise ValueError("No suite success rates found")
    return sum(values) / len(values)


def _episodes(summary: dict[str, Any]) -> int:
    if "total_episodes" in summary:
        return int(summary["total_episodes"])
    return sum(int(result.get("total_episodes", 0)) for result in summary.get("suite_results", []))


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.1f}%"


def _build_row(model: str, setting: str, method: str, summary_path: str, baseline_avg: float | None) -> dict[str, str]:
    summary = _load_summary(summary_path)
    rates = _suite_rates(summary)
    avg = _average_rate(summary, rates)
    row = {
        "Model": model,
        "Setting": setting,
        "Method": method,
        "Avg": _fmt_pct(avg),
        "Delta": _fmt_delta(None if baseline_avg is None else avg - baseline_avg),
        "Episodes": str(_episodes(summary)),
    }
    for suite_key, column_name in SUITE_COLUMNS:
        row[column_name] = _fmt_pct(rates[suite_key]) if suite_key in rates else "-"
    return row


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        nargs=3,
        action="append",
        metavar=("SETTING", "METHOD", "SUMMARY_JSON"),
        required=True,
        help="Add one table row from a fixed_libero_baseline_eval summary.json.",
    )
    parser.add_argument("--model", default="pi0.5", help="Model name to show in the table.")
    parser.add_argument("--delta-baseline", help="Optional FP summary.json used to compute Delta.")
    parser.add_argument("--out-md", help="Optional markdown output path.")
    parser.add_argument("--out-csv", help="Optional CSV output path.")
    args = parser.parse_args()

    baseline_avg = None
    if args.delta_baseline is not None:
        baseline_summary = _load_summary(args.delta_baseline)
        baseline_rates = _suite_rates(baseline_summary)
        baseline_avg = _average_rate(baseline_summary, baseline_rates)

    rows = [_build_row(args.model, setting, method, path, baseline_avg) for setting, method, path in args.row]
    columns = ["Model", "Setting", "Method", "Spatial", "Object", "Goal", "Long", "Avg", "Delta", "Episodes"]
    table = _markdown_table(rows, columns)
    print(table)

    if args.out_md is not None:
        out_md = pathlib.Path(args.out_md).expanduser()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table + "\n", encoding="utf-8")

    if args.out_csv is not None:
        out_csv = pathlib.Path(args.out_csv).expanduser()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
