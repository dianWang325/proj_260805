#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

import generate_long_sequence_datasets_impl as implementation


def adjust_mean(
    values: np.ndarray,
    target_mean: int,
    lower: int,
    upper: int,
    locked: set[int] | None = None,
) -> np.ndarray:
    """Adjust the exact mean without moving required range endpoints."""
    result = np.clip(np.rint(values), lower, upper).astype(np.int64)
    locked = locked or set()
    delta = int(target_mean * len(result) - int(result.sum()))
    direction = 1 if delta > 0 else -1
    while delta:
        raw_candidates = np.flatnonzero(
            result < upper if direction > 0 else result > lower
        )
        candidates = np.array(
            [index for index in raw_candidates if int(index) not in locked],
            dtype=np.int64,
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
    min_index = int(np.argmin(values))
    max_index = int(np.argmax(values))
    values[min_index] = lower
    values[max_index] = upper
    return adjust_mean(
        values,
        target_mean,
        lower,
        upper,
        locked={min_index, max_index},
    )


implementation.adjust_mean = adjust_mean
implementation.variable_lengths = variable_lengths


if __name__ == "__main__":
    raise SystemExit(implementation.main())
