# © Artur Czarnecki. All rights reserved.

"""Typed scale metrics for DIAG-FUNCTIONAL-SCALE-S1."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from intergrax.knowledge.contracts.validation import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    sample_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def to_json_dict(self) -> JsonObject:
        return {
            "sample_count": self.sample_count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }

    @classmethod
    def from_samples_ms(cls, samples_ms: tuple[float, ...]) -> LatencyDistribution:
        if not samples_ms:
            return cls(
                sample_count=0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
            )
        ordered = sorted(samples_ms)
        count = len(ordered)

        def percentile(p: float) -> float:
            if count == 1:
                return ordered[0]
            rank = max(0, min(count - 1, math.ceil(p * count) - 1))
            return ordered[rank]

        return cls(
            sample_count=count,
            p50_ms=percentile(0.50),
            p95_ms=percentile(0.95),
            p99_ms=percentile(0.99),
            max_ms=ordered[-1],
        )


@dataclass(frozen=True, slots=True)
class ThroughputMeasurement:
    operation_count: int
    elapsed_seconds: float

    @property
    def operations_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.operation_count / self.elapsed_seconds

    def to_json_dict(self) -> JsonObject:
        return {
            "operation_count": self.operation_count,
            "elapsed_seconds": self.elapsed_seconds,
            "operations_per_second": self.operations_per_second,
        }


@dataclass(slots=True)
class ScaleCorrectnessAccumulator:
    lost_evidence: int = 0
    unexpected_duplicate_canonical_records: int = 0
    tenant_leakage: int = 0
    run_leakage: int = 0
    task_leakage: int = 0
    unexpected_errors: int = 0
    unexpected_timeouts: int = 0
    integrity_errors: int = 0
    expected_conflicts: int = 0
    analyzer_fidelity_mismatches: int = 0

    def freeze(self) -> ScaleCorrectnessMetrics:
        return ScaleCorrectnessMetrics(
            lost_evidence=self.lost_evidence,
            unexpected_duplicate_canonical_records=self.unexpected_duplicate_canonical_records,
            tenant_leakage=self.tenant_leakage,
            run_leakage=self.run_leakage,
            task_leakage=self.task_leakage,
            unexpected_errors=self.unexpected_errors,
            unexpected_timeouts=self.unexpected_timeouts,
            integrity_errors=self.integrity_errors,
            expected_conflicts=self.expected_conflicts,
            analyzer_fidelity_mismatches=self.analyzer_fidelity_mismatches,
        )

    def all_mandatory_zero(self) -> bool:
        return self.freeze().all_mandatory_zero()


@dataclass(frozen=True, slots=True)
class ScaleCorrectnessMetrics:
    lost_evidence: int
    unexpected_duplicate_canonical_records: int
    tenant_leakage: int
    run_leakage: int
    task_leakage: int
    unexpected_errors: int
    unexpected_timeouts: int
    integrity_errors: int
    expected_conflicts: int
    analyzer_fidelity_mismatches: int

    def to_json_dict(self) -> JsonObject:
        return {
            "lost_evidence": self.lost_evidence,
            "unexpected_duplicate_canonical_records": self.unexpected_duplicate_canonical_records,
            "tenant_leakage": self.tenant_leakage,
            "run_leakage": self.run_leakage,
            "task_leakage": self.task_leakage,
            "unexpected_errors": self.unexpected_errors,
            "unexpected_timeouts": self.unexpected_timeouts,
            "integrity_errors": self.integrity_errors,
            "expected_conflicts": self.expected_conflicts,
            "analyzer_fidelity_mismatches": self.analyzer_fidelity_mismatches,
        }

    def all_mandatory_zero(self) -> bool:
        return (
            self.lost_evidence == 0
            and self.unexpected_duplicate_canonical_records == 0
            and self.tenant_leakage == 0
            and self.run_leakage == 0
            and self.task_leakage == 0
            and self.unexpected_errors == 0
            and self.unexpected_timeouts == 0
            and self.integrity_errors == 0
            and self.analyzer_fidelity_mismatches == 0
        )


@dataclass(frozen=True, slots=True)
class ScaleResourceMetrics:
    rss_before_bytes: int | None
    rss_after_bytes: int | None
    mongo_document_count: int | None
    mongo_storage_size_bytes: int | None
    cpu_core_count: int | None

    def to_json_dict(self) -> JsonObject:
        return {
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": self.rss_after_bytes,
            "mongo_document_count": self.mongo_document_count,
            "mongo_storage_size_bytes": self.mongo_storage_size_bytes,
            "cpu_core_count": self.cpu_core_count,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReadScaleCurvePoint:
    label: str
    total_evidence: int
    probe_evidence_count: int
    read_latency_ms: float

    def to_json_dict(self) -> JsonObject:
        return {
            "label": self.label,
            "total_evidence": self.total_evidence,
            "probe_evidence_count": self.probe_evidence_count,
            "read_latency_ms": self.read_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReadScaleCurve:
    points: tuple[ExecutionReadScaleCurvePoint, ...]

    def to_json_dict(self) -> JsonObject:
        return {"points": [point.to_json_dict() for point in self.points]}

    def gross_linear_growth(self) -> bool:
        if len(self.points) < 2:
            return False
        first = self.points[0]
        last = self.points[-1]
        if first.read_latency_ms <= 0 or first.total_evidence <= 0:
            return False
        evidence_ratio = last.total_evidence / first.total_evidence
        latency_ratio = last.read_latency_ms / first.read_latency_ms
        return latency_ratio >= (evidence_ratio * 0.85)


class MonotonicTimer:
    """Elapsed duration helper using monotonic clock."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start

    def elapsed_ms(self) -> float:
        return self.elapsed_seconds() * 1000.0


def latency_samples_to_json(samples: dict[str, LatencyDistribution]) -> JsonObject:
    return {key: value.to_json_dict() for key, value in samples.items()}


def throughput_to_json(samples: dict[str, ThroughputMeasurement]) -> JsonObject:
    return {key: value.to_json_dict() for key, value in samples.items()}


def metrics_json_value(payload: JsonObject) -> JsonValue:
    return payload


__all__ = [
    "ExecutionReadScaleCurve",
    "ExecutionReadScaleCurvePoint",
    "LatencyDistribution",
    "MonotonicTimer",
    "ScaleCorrectnessAccumulator",
    "ScaleCorrectnessMetrics",
    "ScaleResourceMetrics",
    "ThroughputMeasurement",
    "latency_samples_to_json",
    "metrics_json_value",
    "throughput_to_json",
]
