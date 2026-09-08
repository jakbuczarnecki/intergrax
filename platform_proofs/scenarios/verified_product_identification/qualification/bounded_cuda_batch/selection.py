"""Safe production batch selection for bounded CUDA throughput qualification."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    BATCH_32_MIN_IMPROVEMENT_FRACTION,
    MIN_VRAM_HEADROOM_FRACTION,
    BatchVariantMeasurement,
    OptionalBatch32Decision,
    ProductionBatchSelection,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.metrics import (
    compute_vram_headroom_fraction,
)


def memory_headroom_is_sufficient(
    *,
    gpu_total_memory_bytes: int,
    peak_cuda_allocated_bytes: int,
    minimum_headroom_fraction: float = MIN_VRAM_HEADROOM_FRACTION,
) -> bool:
    headroom = compute_vram_headroom_fraction(
        gpu_total_memory_bytes=gpu_total_memory_bytes,
        peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
    )
    return headroom >= minimum_headroom_fraction


def has_pathological_slowdown(
    current: BatchVariantMeasurement,
    previous_smaller: BatchVariantMeasurement | None,
) -> bool:
    if previous_smaller is None:
        return False
    if previous_smaller.records_per_second <= 0.0:
        return False
    return current.records_per_second < previous_smaller.records_per_second


def evaluate_batch_safety(
    measurement: BatchVariantMeasurement,
    *,
    previous_smaller: BatchVariantMeasurement | None,
    minimum_headroom_fraction: float = MIN_VRAM_HEADROOM_FRACTION,
) -> tuple[bool, str]:
    if measurement.cuda_oom:
        return False, "CUDA OOM observed"
    if not memory_headroom_is_sufficient(
        gpu_total_memory_bytes=measurement.gpu_total_memory_bytes,
        peak_cuda_allocated_bytes=measurement.peak_cuda_allocated_bytes,
        minimum_headroom_fraction=minimum_headroom_fraction,
    ):
        headroom = measurement.vram_headroom_fraction
        return (
            False,
            f"VRAM headroom {headroom:.1%} below required {minimum_headroom_fraction:.0%}",
        )
    if has_pathological_slowdown(measurement, previous_smaller):
        return (
            False,
            (
                f"throughput {measurement.records_per_second:.2f} rec/s regressed versus "
                f"batch {previous_smaller.batch_size} "
                f"({previous_smaller.records_per_second:.2f} rec/s)"
            ),
        )
    return True, "stable CUDA execution with sufficient VRAM headroom"


def select_production_batch_candidate(
    measurements: tuple[BatchVariantMeasurement, ...],
) -> ProductionBatchSelection:
    safe_measurements = [item for item in measurements if item.safe]
    if not safe_measurements:
        return ProductionBatchSelection(
            batch_size=None,
            rationale="no safe batch variant completed qualification",
        )
    winner = max(safe_measurements, key=lambda item: item.records_per_second)
    return ProductionBatchSelection(
        batch_size=winner.batch_size,
        rationale=(
            f"batch {winner.batch_size} is the highest stable throughput among safe "
            f"variants ({winner.records_per_second:.2f} records/s, "
            f"{winner.vram_headroom_fraction:.1%} VRAM headroom)"
        ),
    )


def should_run_optional_batch_32(
    *,
    batch_8: BatchVariantMeasurement | None,
    batch_16: BatchVariantMeasurement,
    minimum_headroom_fraction: float = MIN_VRAM_HEADROOM_FRACTION,
    minimum_improvement_fraction: float = BATCH_32_MIN_IMPROVEMENT_FRACTION,
) -> OptionalBatch32Decision:
    if batch_16.cuda_oom:
        return OptionalBatch32Decision(
            executed=False,
            reason="batch 16 encountered CUDA OOM",
        )
    if not batch_16.safe:
        return OptionalBatch32Decision(
            executed=False,
            reason=f"batch 16 failed safety gate: {batch_16.safety_detail}",
        )
    if not memory_headroom_is_sufficient(
        gpu_total_memory_bytes=batch_16.gpu_total_memory_bytes,
        peak_cuda_allocated_bytes=batch_16.peak_cuda_allocated_bytes,
        minimum_headroom_fraction=minimum_headroom_fraction,
    ):
        return OptionalBatch32Decision(
            executed=False,
            reason="batch 16 peak VRAM does not leave 20% GPU-memory headroom",
        )
    if batch_8 is None or batch_8.records_per_second <= 0.0:
        return OptionalBatch32Decision(
            executed=False,
            reason="batch 8 baseline unavailable for throughput comparison",
        )
    improvement = (
        batch_16.records_per_second - batch_8.records_per_second
    ) / batch_8.records_per_second
    if improvement < minimum_improvement_fraction:
        return OptionalBatch32Decision(
            executed=False,
            reason=(
                f"batch 16 throughput improvement versus batch 8 is {improvement:.1%}; "
                f"requires >= {minimum_improvement_fraction:.0%}"
            ),
        )
    if has_pathological_slowdown(batch_16, batch_8):
        return OptionalBatch32Decision(
            executed=False,
            reason="batch 16 shows latency collapse relative to batch 8",
        )
    return OptionalBatch32Decision(
        executed=True,
        reason="batch 16 passed all optional batch-32 gating conditions",
    )
