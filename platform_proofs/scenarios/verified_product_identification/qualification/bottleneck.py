"""Bottleneck analysis from measured materialization timings."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    BottleneckBreakdown,
)


def analyze_bottleneck(
    *,
    derive_seconds: float,
    embedding_seconds: float,
    artifact_write_seconds: float,
) -> BottleneckBreakdown:
    total = derive_seconds + embedding_seconds + artifact_write_seconds
    if total <= 0:
        return BottleneckBreakdown(
            derive_share=0.0,
            embedding_share=0.0,
            artifact_write_share=0.0,
            dominant_stage="unknown",
            parallelization_recommendation="insufficient timing evidence",
        )
    derive_share = derive_seconds / total
    embedding_share = embedding_seconds / total
    artifact_write_share = artifact_write_seconds / total
    shares = {
        "derive": derive_share,
        "embedding": embedding_share,
        "artifact_write": artifact_write_share,
    }
    dominant_stage = max(shares, key=shares.get)
    if embedding_share >= 0.90:
        recommendation = (
            "embedding dominates (>90%); defer I/O parallelization to 5C4C unless "
            "GPU utilization profiling shows substantial idle time"
        )
    elif artifact_write_share >= 0.20:
        recommendation = "artifact write is material; evaluate writer concurrency in 5C4C"
    elif derive_share >= 0.20:
        recommendation = "derive is material relative to embedding; profile CPU derivation scaling in 5C4C"
    else:
        recommendation = "balanced pipeline; prioritize embedding throughput tuning first"
    return BottleneckBreakdown(
        derive_share=derive_share,
        embedding_share=embedding_share,
        artifact_write_share=artifact_write_share,
        dominant_stage=dominant_stage,
        parallelization_recommendation=recommendation,
    )
