# © Artur Czarnecki. All rights reserved.

"""Pure metric calculations for qualification performance certification."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from testing_support.execution_qualification.performance.models import (
    QualificationPerformanceOutcome,
    TimedWallSeconds,
)


def duplicate_execution_eliminated_count(
    legacy_logical_count: int,
    canonical_physical_count: int,
) -> int:
    if legacy_logical_count < 1 or canonical_physical_count < 1:
        raise ValueError("counts must be >= 1")
    if canonical_physical_count > legacy_logical_count:
        raise ValueError("canonical physical count cannot exceed legacy logical count")
    return legacy_logical_count - canonical_physical_count


def duplicate_execution_eliminated_percent(
    legacy_logical_count: int,
    canonical_physical_count: int,
) -> float:
    eliminated = duplicate_execution_eliminated_count(
        legacy_logical_count,
        canonical_physical_count,
    )
    if legacy_logical_count == 0:
        raise ValueError("legacy_logical_count must be > 0")
    return (eliminated / legacy_logical_count) * 100.0


def speedup_ratio(legacy_wall_seconds: float, canonical_wall_seconds: float) -> float:
    if legacy_wall_seconds <= 0 or canonical_wall_seconds <= 0:
        raise ValueError("wall times must be positive for speedup")
    return legacy_wall_seconds / canonical_wall_seconds


def wall_time_reduction_percent(
    legacy_wall_seconds: float,
    canonical_wall_seconds: float,
) -> float:
    if legacy_wall_seconds <= 0:
        raise ValueError("legacy_wall_seconds must be positive")
    return (
        (legacy_wall_seconds - canonical_wall_seconds) / legacy_wall_seconds
    ) * 100.0


def effective_concurrency(
    total_leaf_work_seconds: float,
    wall_seconds: float,
) -> float:
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    if total_leaf_work_seconds < 0:
        raise ValueError("total_leaf_work_seconds must be non-negative")
    return total_leaf_work_seconds / wall_seconds


def scheduler_parallel_efficiency_estimate(
    total_leaf_work_seconds: float,
    wall_seconds: float,
    max_parallel: int,
) -> float:
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    denominator = wall_seconds * max_parallel
    if denominator <= 0:
        raise ValueError("invalid denominator")
    return min(total_leaf_work_seconds / denominator, 1.0)


def critical_path_max_leaf_approximation(
    leaf_durations_seconds: tuple[float, ...],
) -> float:
    if not leaf_durations_seconds:
        raise ValueError("leaf_durations_seconds must be non-empty")
    return max(leaf_durations_seconds)


def optional_speedup_from_timed_walls(
    legacy_wall: TimedWallSeconds,
    canonical_wall: TimedWallSeconds,
) -> float | None:
    if legacy_wall.seconds is None or canonical_wall.seconds is None:
        return None
    return speedup_ratio(legacy_wall.seconds, canonical_wall.seconds)


def optional_reduction_from_timed_walls(
    legacy_wall: TimedWallSeconds,
    canonical_wall: TimedWallSeconds,
) -> float | None:
    if legacy_wall.seconds is None or canonical_wall.seconds is None:
        return None
    return wall_time_reduction_percent(legacy_wall.seconds, canonical_wall.seconds)


def classify_performance_outcome(
    wall_time_reduction_percent_value: float | None,
) -> QualificationPerformanceOutcome | None:
    if wall_time_reduction_percent_value is None:
        return None
    value = wall_time_reduction_percent_value
    if value >= 60.0:
        return QualificationPerformanceOutcome.EXCELLENT
    if value >= 40.0:
        return QualificationPerformanceOutcome.STRONG
    if value >= 20.0:
        return QualificationPerformanceOutcome.MODERATE
    return QualificationPerformanceOutcome.WEAK


def classify_performance_problem_solved(
    canonical_wall_seconds: float | None,
    wall_time_reduction_percent_value: float | None,
) -> str:
    if canonical_wall_seconds is None or wall_time_reduction_percent_value is None:
        return "PARTIALLY"
    under_ten_minutes = canonical_wall_seconds < 600.0
    strong_reduction = wall_time_reduction_percent_value >= 40.0
    moderate_reduction = wall_time_reduction_percent_value >= 20.0
    if under_ten_minutes and strong_reduction:
        return "YES"
    if canonical_wall_seconds <= 900.0 or moderate_reduction:
        return "PARTIALLY"
    if canonical_wall_seconds > 900.0 and wall_time_reduction_percent_value < 20.0:
        return "NO"
    return "PARTIALLY"


@dataclass(frozen=True, slots=True)
class WallSampleStatistics:
    min_seconds: float
    max_seconds: float
    mean_seconds: float
    median_seconds: float | None
    spread_percent: float

    def __post_init__(self) -> None:
        for name, value in (
            ("min_seconds", self.min_seconds),
            ("max_seconds", self.max_seconds),
            ("mean_seconds", self.mean_seconds),
            ("spread_percent", self.spread_percent),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.median_seconds is not None:
            if not math.isfinite(self.median_seconds) or self.median_seconds < 0:
                raise ValueError("median_seconds must be finite and non-negative")


def wall_sample_statistics(samples: tuple[float, ...]) -> WallSampleStatistics:
    if not samples:
        raise ValueError("samples must be non-empty")
    minimum = min(samples)
    maximum = max(samples)
    mean_value = statistics.mean(samples)
    median_value: float | None
    if len(samples) >= 3:
        median_value = statistics.median(samples)
    else:
        median_value = None
    spread = 0.0
    if mean_value > 0:
        spread = ((maximum - minimum) / mean_value) * 100.0
    return WallSampleStatistics(
        min_seconds=minimum,
        max_seconds=maximum,
        mean_seconds=mean_value,
        median_seconds=median_value,
        spread_percent=spread,
    )


def primary_wall_seconds_from_samples(
    samples: tuple[float, ...],
) -> float:
    stats = wall_sample_statistics(samples)
    if stats.median_seconds is not None:
        return stats.median_seconds
    return stats.mean_seconds
