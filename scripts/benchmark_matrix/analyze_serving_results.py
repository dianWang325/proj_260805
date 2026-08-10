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
VARIANT_ORDER = {"baseline": 0, "cpp_only": 1, "srf_only": 2, "cpp_srf": 3}


def percentile(values: list[float], p: int) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_fields(prefix: str, values_ms: list[float]) -> dict[str, float]:
    result = {f"{prefix}_mean_ms": mean(values_ms) if values_ms else math.nan}
    result.update(
        {f"{prefix}_p{p}_ms": percentile(values_ms, p) for p in PERCENTILES}
    )
    return result


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
        tpot_s = (e2e_s - ttft_s) / (output_len - 1) if output_len > 1 else 0.0
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


def summarize_requests(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    input_lens = [int(row["input_len"]) for row in rows]
    result: dict[str, float | int] = {
        "request_count": len(rows),
        "input_tokens": sum(input_lens),
        "input_len_min": min(input_lens) if input_lens else math.nan,
        "input_len_max": max(input_lens) if input_lens else math.nan,
        "input_len_mean": mean(input_lens) if input_lens else math.nan,
        "o_n2_compute_ratio": (
            mean(value * value for value in input_lens) / (mean(input_lens) ** 2)
            if input_lens
            else math.nan
        ),
    }
    for prefix, key in (("ttft", "ttft_ms"), ("tpot", "tpot_ms"), ("e2e", "e2e_ms")):
        result.update(metric_fields(prefix, [float(row[key]) for row in rows]))
    return result


def groups_for(
    workload: str, rows: list[dict[str, float | int]], threshold: int
) -> Iterable[tuple[str, list[dict[str, float | int]]]]:
    yield "all", rows
    if workload == "srf_mixed":
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
        return "" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "variant", "workload", "request_group", "runs", "request_count", "failed",
        "input_len_mean", "input_throughput_tps", "input_throughput_per_card_tps",
        "output_throughput_tps", "ttft_mean_ms", "ttft_p99_ms", "tpot_mean_ms",
        "tpot_p99_ms", "e2e_mean_ms", "e2e_p99_ms",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            stream.write(
                "| " + " | ".join(format_value(row.get(c, "")) for c in columns) + " |\n"
            )


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def acceptance_rows(
    summary: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = {
        (row["variant"], row["workload"], row["request_group"]): row
        for row in summary
    }

    def value(variant: str, workload: str, group: str, metric: str) -> float:
        row = lookup.get((variant, workload, group))
        return float(row[metric]) if row and metric in row else math.nan

    cpp_retention = ratio(
        value("cpp_only", "variable_long", "all", "input_throughput_tps"),
        value("cpp_only", "fixed_long", "all", "input_throughput_tps"),
    )
    ttft_without_srf = value("cpp_only", "srf_mixed", "short", "ttft_mean_ms")
    ttft_with_srf = value("cpp_srf", "srf_mixed", "short", "ttft_mean_ms")
    srf_gain = 1.0 - ratio(ttft_with_srf, ttft_without_srf)
    baseline_ttft = value("baseline", "srf_mixed", "short", "ttft_mean_ms")
    srf_only_ttft = value("srf_only", "srf_mixed", "short", "ttft_mean_ms")
    srf_gain_without_cpp = 1.0 - ratio(srf_only_ttft, baseline_ttft)
    combined_retention = ratio(
        value("cpp_srf", "variable_long", "all", "input_throughput_tps"),
        value("cpp_srf", "fixed_long", "all", "input_throughput_tps"),
    )
    compute_ratio = float(
        manifest.get("variable_long", {}).get(
            "o_n2_compute_ratio_vs_same_mean_fixed", math.nan
        )
    )
    failed = sum(int(row.get("failed", 0)) for row in summary if row["request_group"] == "all")

    def item(name: str, formula: str, observed: float, target: str, passed: bool) -> dict[str, Any]:
        status = "N/A" if not math.isfinite(observed) else ("PASS" if passed else "FAIL")
        return {
            "criterion": name,
            "formula": formula,
            "observed": observed,
            "target": target,
            "status": status,
        }

    rows = [
        item(
            "CPP variable/fixed input-throughput retention",
            "cpp_only.variable_long / cpp_only.fixed_long",
            cpp_retention * 100.0,
            ">=85% (provisional interpretation of approximately 85%)",
            cpp_retention >= 0.85,
        ),
        item(
            "SRF short-request mean TTFT gain with CPP enabled",
            "1 - cpp_srf.short_TTFT / cpp_only.short_TTFT",
            srf_gain * 100.0,
            ">=10% (provisional interpretation of approximately 10%)",
            srf_gain >= 0.10,
        ),
        item(
            "SRF short-request mean TTFT gain without CPP",
            "1 - srf_only.short_TTFT / baseline.short_TTFT",
            srf_gain_without_cpp * 100.0,
            ">=10% (supporting premise check)",
            srf_gain_without_cpp >= 0.10,
        ),
        item(
            "Variable dataset O(n^2) compute increase",
            "mean(input_len^2) / 65536^2 - 1",
            (compute_ratio - 1.0) * 100.0,
            "<=11%",
            compute_ratio <= 1.11,
        ),
        item(
            "CPP+SRF variable/fixed throughput decrease",
            "1 - cpp_srf.variable_long / cpp_srf.fixed_long",
            (1.0 - combined_retention) * 100.0,
            "<15%",
            combined_retention > 0.85,
        ),
        item(
            "Request failures",
            "sum(failed) across all variant/workload runs",
            float(failed),
            "=0",
            failed == 0,
        ),
    ]
    return rows


def write_acceptance_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Requirement acceptance report\n\n")
        stream.write(
            "The 85% CPP and 10% SRF statements are approximate premises. "
            "Their PASS/FAIL thresholds are provisional until confirmed by the requirement owner.\n\n"
        )
        stream.write("| criterion | observed (%) | target | status |\n")
        stream.write("| --- | ---: | --- | --- |\n")
        for row in rows:
            observed = row["observed"]
            rendered = "N/A" if not math.isfinite(observed) else f"{observed:.3f}"
            stream.write(
                f"| {row['criterion']} | {rendered} | {row['target']} | {row['status']} |\n"
            )
        stream.write("\nFormulas are also recorded in `acceptance_report.csv`.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze long-sequence vLLM matrix results.")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--threshold", type=int, default=4096)
    parser.add_argument("--device-count", type=int, default=8)
    args = parser.parse_args()
    if args.device_count < 1:
        parser.error("--device-count must be positive")

    result_paths = sorted(
        path for path in args.result_root.rglob("*.json")
        if not path.name.startswith("matrix_") and path.name != "dataset_manifest.json"
    )
    if not result_paths:
        parser.error(f"no benchmark JSON files found under {args.result_root}")

    per_run: list[dict[str, Any]] = []
    for path in result_paths:
        data = load_result(path)
        variant = str(data.get("variant") or path.parent.parent.name)
        workload = str(data.get("workload") or "unknown")
        repeat = int(data.get("repeat") or 0)
        rows = request_rows(data)
        failed = int(data.get("failed") or 0)
        duration = float(data.get("duration") or 0.0)
        for request_group, grouped_rows in groups_for(workload, rows, args.threshold):
            summary: dict[str, Any] = {
                "variant": variant,
                "workload": workload,
                "request_group": request_group,
                "repeat": repeat,
                "source": str(path),
                "completed": int(data.get("completed") or 0),
                "failed": failed,
                "duration_s": duration,
                "request_throughput_rps": float(data.get("request_throughput") or 0.0) if request_group == "all" else math.nan,
                "output_throughput_tps": float(data.get("output_throughput") or 0.0) if request_group == "all" else math.nan,
                "total_token_throughput_tps": float(data.get("total_token_throughput") or 0.0) if request_group == "all" else math.nan,
            }
            summary.update(summarize_requests(grouped_rows))
            input_tps = (
                float(summary["input_tokens"]) / duration
                if request_group == "all" and duration > 0
                else math.nan
            )
            summary["input_throughput_tps"] = input_tps
            summary["input_throughput_per_card_tps"] = input_tps / args.device_count
            per_run.append(summary)

    per_run.sort(key=sort_key)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[(row["variant"], row["workload"], row["request_group"])].append(row)

    metric_names = [
        "duration_s", "request_throughput_rps", "input_throughput_tps",
        "input_throughput_per_card_tps", "output_throughput_tps",
        "total_token_throughput_tps", "input_len_min", "input_len_max",
        "input_len_mean", "o_n2_compute_ratio",
        *[f"{name}_{kind}_ms" for name in ("ttft", "tpot", "e2e") for kind in ("mean", "p50", "p90", "p95", "p99")],
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
        summary.update({name: finite_mean(row[name] for row in rows) for name in metric_names})
        matrix_summary.append(summary)
    matrix_summary.sort(key=sort_key)

    manifest_path = args.result_root / "datasets" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    acceptance = acceptance_rows(matrix_summary, manifest)

    write_csv(args.result_root / "matrix_per_run.csv", per_run)
    write_csv(args.result_root / "matrix_summary.csv", matrix_summary)
    write_summary_markdown(args.result_root / "matrix_summary.md", matrix_summary)
    write_csv(args.result_root / "acceptance_report.csv", acceptance)
    write_acceptance_markdown(args.result_root / "acceptance_report.md", acceptance)
    print(f"Analyzed {len(result_paths)} result files")
    print(f"Summary: {args.result_root / 'matrix_summary.md'}")
    print(f"Acceptance: {args.result_root / 'acceptance_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
