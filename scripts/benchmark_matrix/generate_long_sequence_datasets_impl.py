#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllm.benchmarks.datasets.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer


def adjust_mean(
    values: np.ndarray, target_mean: int, lower: int, upper: int
) -> np.ndarray:
    result = np.clip(np.rint(values), lower, upper).astype(np.int64)
    delta = int(target_mean * len(result) - int(result.sum()))
    direction = 1 if delta > 0 else -1
    while delta:
        candidates = np.flatnonzero(
            result < upper if direction > 0 else result > lower
        )
        if not len(candidates):
            raise ValueError("cannot adjust lengths to requested mean within bounds")
        step = min(abs(delta), len(candidates))
        result[candidates[:step]] += direction
        delta -= direction * step
    return result


def variable_lengths(
    count: int,
    lower: int,
    upper: int,
    target_mean: int,
    sigma: int,
    seed: int,
) -> np.ndarray:
    if count < 4:
        raise ValueError("variable dataset requires at least four prompts")
    if not lower < target_mean < upper:
        raise ValueError("variable mean must be strictly between min and max")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=count)
    z = (z - z.mean()) / z.std()

    low_center = lower - 4 * sigma
    high_center = upper + 4 * sigma
    for _ in range(80):
        center = (low_center + high_center) / 2
        current = np.clip(center + sigma * z, lower, upper)
        if current.mean() < target_mean:
            low_center = center
        else:
            high_center = center
    values = np.clip((low_center + high_center) / 2 + sigma * z, lower, upper)
    values[np.argmin(values)] = lower
    values[np.argmax(values)] = upper
    return adjust_mean(values, target_mean, lower, upper)


def build_prompt(
    dataset: RandomDataset,
    tokenizer,
    target_total_len: int,
    index: int,
    allowed_tokens: np.ndarray,
) -> tuple[str, int]:
    special_tokens = int(tokenizer.num_special_tokens_to_add())
    content_len = target_total_len - special_tokens
    if content_len < 1:
        raise ValueError(f"target input length is too small: {target_total_len}")
    prompt, _, _ = dataset.generate_token_sequence(
        tokenizer=tokenizer,
        prefix_token_ids=[],
        prefix_len=0,
        vocab_size=tokenizer.vocab_size,
        input_len=content_len,
        offset=(index * 104729) % tokenizer.vocab_size,
        index=index,
        allowed_tokens=allowed_tokens,
    )
    actual_len = len(tokenizer(prompt).input_ids)
    if actual_len != target_total_len:
        raise RuntimeError(
            f"token round-trip mismatch for item {index}: "
            f"target={target_total_len}, actual={actual_len}"
        )
    return prompt, actual_len


def write_dataset(
    path: Path,
    lengths: list[int],
    output_len: int,
    tokenizer,
    seed: int,
) -> dict[str, float | int | str]:
    dataset = RandomDataset(random_seed=seed)
    prohibited = set(tokenizer.all_special_ids)
    allowed_tokens = np.array(
        [token for token in range(tokenizer.vocab_size) if token not in prohibited],
        dtype=np.int64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_lengths: list[int] = []
    with path.open("w", encoding="utf-8") as stream:
        for index, target_len in enumerate(lengths):
            prompt, actual_len = build_prompt(
                dataset, tokenizer, target_len, index, allowed_tokens
            )
            stream.write(
                json.dumps(
                    {"prompt": prompt, "output_tokens": output_len},
                    ensure_ascii=False,
                )
                + "\n"
            )
            actual_lengths.append(actual_len)

    mean_len = float(np.mean(actual_lengths))
    compute_ratio = float(np.mean(np.square(actual_lengths)) / (mean_len**2))
    return {
        "path": str(path),
        "count": len(actual_lengths),
        "input_min": min(actual_lengths),
        "input_max": max(actual_lengths),
        "input_mean": mean_len,
        "output_len": output_len,
        "o_n2_compute_ratio_vs_same_mean_fixed": compute_ratio,
        "o_n2_compute_increase_pct": (compute_ratio - 1.0) * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate exact-token long-sequence datasets for vLLM bench serve."
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-count", type=int, required=True)
    parser.add_argument("--variable-count", type=int, required=True)
    parser.add_argument("--srf-mixed-count", type=int, required=True)
    parser.add_argument("--fixed-input-len", type=int, default=65536)
    parser.add_argument("--variable-min-len", type=int, default=40960)
    parser.add_argument("--variable-max-len", type=int, default=81920)
    parser.add_argument("--variable-mean-len", type=int, default=65536)
    parser.add_argument("--variable-sigma", type=int, default=10240)
    parser.add_argument("--output-len", type=int, default=2560)
    parser.add_argument("--srf-short-input-len", type=int, default=1024)
    parser.add_argument("--srf-long-input-len", type=int, default=65536)
    parser.add_argument("--srf-output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if min(args.fixed_count, args.variable_count, args.srf_mixed_count) < 1:
        parser.error("dataset counts must be positive")
    if args.variable_count < 4 or args.srf_mixed_count < 4:
        parser.error("variable and SRF mixed datasets require at least four prompts")
    if min(args.output_len, args.srf_output_len) < 1:
        parser.error("output lengths must be positive")
    max_total = max(
        args.fixed_input_len, args.variable_max_len, args.srf_long_input_len
    ) + max(args.output_len, args.srf_output_len)
    if max_total > 1048576:
        parser.error("input + output exceeds model max_position_embeddings=1048576")

    tokenizer = get_tokenizer(
        args.tokenizer,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
    )
    variable = variable_lengths(
        args.variable_count,
        args.variable_min_len,
        args.variable_max_len,
        args.variable_mean_len,
        args.variable_sigma,
        args.seed,
    )
    srf_lengths = [
        args.srf_long_input_len if index % 2 == 0 else args.srf_short_input_len
        for index in range(args.srf_mixed_count)
    ]

    manifest = {
        "fixed_long": write_dataset(
            args.output_dir / "fixed_64k.jsonl",
            [args.fixed_input_len] * args.fixed_count,
            args.output_len,
            tokenizer,
            args.seed,
        ),
        "variable_long": write_dataset(
            args.output_dir / "variable_40k_80k.jsonl",
            variable.tolist(),
            args.output_len,
            tokenizer,
            args.seed + 1,
        ),
        "srf_mixed": write_dataset(
            args.output_dir / "srf_mixed.jsonl",
            srf_lengths,
            args.srf_output_len,
            tokenizer,
            args.seed + 2,
        ),
        "settings": {
            "distribution": "bounded normal sample adjusted to exact mean",
            "variable_sigma": args.variable_sigma,
            "seed": args.seed,
            "required_max_model_len": max_total,
        },
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    ratio = float(manifest["variable_long"]["o_n2_compute_ratio_vs_same_mean_fixed"])
    if ratio > 1.11 + 1e-12:
        raise RuntimeError(
            f"generated variable dataset violates O(n^2) <=11% premise: {ratio:.6f}"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
