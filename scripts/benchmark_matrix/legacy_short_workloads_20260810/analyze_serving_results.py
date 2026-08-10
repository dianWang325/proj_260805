#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PERCENTILES = (50, 90, 95, 99)
VARIANT_ORDER = {
    "baseline": 0,
    "cpp_only": 1,
    "srf_only": 2,
    "cpp_srf": 3,
}


def percentile(values: list[float], p: int) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_fields(prefix: str, values_ms: list[float]) -> dict[str, float]:
    fields = {f"{prefix}_mean_ms": mean(values_ms) if values_ms else math.nan}
    fields.update(
        {f"{prefix}_p{p}_ms": percentile(values_ms, p) for p in PERCENTILES}
    )
    return fields


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, list):
        if not value:
            raise ValueError("empty result list")
        value = value[-1]
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object, got {type(value).__name__}")
    return value


def request_rows(data: dict[str, Any]) -> list[dict[str, float | int]]:
    input_lens = data.get("input_lens", [])
    output_lens = data.get("output_lens", [])
    ttfts = data.get("ttfts", [])
    itls = data.get("itls", [])
    errors = data.get("errors", [])
    count = min(len(input_lens), len(output_lens), len(ttfts), len(itls))
    rows: list[dict[str, float | int]] = []

    for index in range(count):
        error = errors[index] if index < len(errors) else ""
        output_len = int(output_lens[index])
        if error or output_len <= 0:
            continue

        ttft_s = float(ttfts[index])
        request_itls = itls[index] or []
        e2e_s = ttft_s + sum(float(value) for value in request_itls)
        tpot_s = 0.0
        if output_len > 1:
            tpot_s = (e2e_s - ttft_s) / (output_len - 1)

        rows.append(
            {
                "input_len": int(input_lens[index]),
                "output_len": output_len,
                "ttft_ms": ttft_s * 1000.0,
                "tpot_ms": tpot_s * 1000.0,
                "e2e_ms": e2e_s * 1000.0,
            }
        )

    return rows


def summarize_requests(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    ttfts = [float(row["ttft_ms"]) for row in rows]
    tpots = [float(row["tpot_ms"]) for row in rows]
    e2els = [float(row["e2e_ms"]) for row in rows]
    result: dict[str, float | int] = {"request_count": len(rows)}
    result.update(metric_fields("ttft", ttfts))
    result.update(metric_fields("tpot", tpots))
    result.update(metric_fields("e2e", e2els))
    return result


def groups_for(
    workload: str,
    rows: list[dict[str, float | int]],
    threshold: int,
) -> Iterable[tuple[str, list[dict[str, float | int]]]]:
    yield "all", rows
    if workload == "mixed":
        yield "short", [row for row in rows if int(row["input_len"]) <= threshold]
        yield "long", [row for row in rows if int(row["input_len"]) > threshold]


def finite_mean(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values if value not in (None, "")]
    numbers = [value for value in numbers if math.isfinite(value)]
    return mean(numbers) if numbers else math.nan


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        VARIANT_ORDER.get(str(row["variant"]), 99),
        str(row["workload"]),
        str(row["request_group"]),
        int(row.get("repeat", 0)),
    )


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "variant",
        "workload",
        "request_group",
        "runs",
        "request_count",
        "failed",
        "request_throughput_rps",
        "output_throughput_tps",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "e2e_p50_ms",
        "e2e_p99_ms",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            stream.write(
                "| "
                + " | ".join(format_value(row.get(column, "")) for column in columns)
                + " |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize vllm bench serve detailed JSON results."
    )
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--threshold", type=int, default=4096)
    args = parser.parse_args()

    result_paths = sorted(
        path
        for path in args.result_root.rglob("*.json")
        if not path.name.startswith("matrix_")
    )
    if not result_paths:
        parser.error(f"no benchmark JSON files found under {args.result_root}")

    per_run: list[dict[str, Any]] = []
    for path in result_paths:
        try:
            data = load_result(path)
            variant = str(data.get("variant") or path.parent.parent.name)
            workload = str(data.get("workload") or "unknown")
            repeat = int(data.get("repeat") or 0)
            rows = request_rows(data)
            failed = int(data.get("failed") or 0)

            for request_group, grouped_rows in groups_for(
                workload, rows, args.threshold
            ):
                summary: dict[str, Any] = {
                    "variant": variant,
                    "workload": workload,
                    "request_group": request_group,
                    "repeat": repeat,
                    "source": str(path),
                    "completed": int(data.get("completed") or 0),
                    "failed": failed,
                    "duration_s": float(data.get("duration") or 0.0),
                    "request_throughput_rps": (
                        float(data.get("request_throughput") or 0.0)
                        if request_group == "all"
                        else math.nan
                    ),
                    "output_throughput_tps": (
                        float(data.get("output_throughput") or 0.0)
                        if request_group == "all"
                        else math.nan
                    ),
                    "total_token_throughput_tps": (
                        float(data.get("total_token_throughput") or 0.0)
                        if request_group == "all"
                        else math.nan
                    ),
                }
                summary.update(summarize_requests(grouped_rows))
                per_run.append(summary)
        except Exception as error:
            raise RuntimeError(f"failed to analyze {path}: {error}") from error

    per_run.sort(key=sort_key)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        key = (row["variant"], row["workload"], row["request_group"])
        grouped[key].append(row)

    metric_names = [
        "duration_s",
        "request_throughput_rps",
        "output_throughput_tps",
        "total_token_throughput_tps",
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_mean_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "e2e_mean_ms",
        "e2e_p50_ms",
        "e2e_p90_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
    ]
    matrix_summary: list[dict[str, Any]] = []
    for (variant, workload, request_group), rows in grouped.items():
        summary = {
            "variant": variant,
            "workload": workload,
            "request_group": request_group,
            "runs": len(rows),
            "request_count": sum(int(row["request_count"]) for row in rows),
            "completed": sum(int(row["completed"]) for row in rows),
            "failed": sum(int(row["failed"]) for row in rows),
        }
        summary.update(
            {name: finite_mean(row[name] for row in rows) for name in metric_names}
        )
        matrix_summary.append(summary)

    matrix_summary.sort(key=sort_key)
    per_run_path = args.result_root / "matrix_per_run.csv"
    summary_csv_path = args.result_root / "matrix_summary.csv"
    summary_md_path = args.result_root / "matrix_summary.md"
    write_csv(per_run_path, per_run)
    write_csv(summary_csv_path, matrix_summary)
    write_markdown(summary_md_path, matrix_summary)

    print(f"Analyzed {len(result_paths)} result files")
    print(f"Per-run CSV: {per_run_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary Markdown: {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
