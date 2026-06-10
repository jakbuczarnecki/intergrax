# © Artur Czarnecki. All rights reserved.

"""Live architecture health metrics wiring (AUDIT-IDEAL-1.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture import build_catalog_capability_graph, compute_architecture_metrics
from intergrax.runtime.architecture.architecture_metrics_pipeline import (
    ArchitectureMetricsPipelineReport,
    ArchitectureMetricsSnapshot,
    build_metrics_pipeline_report,
)


@dataclass(frozen=True, slots=True)
class ArchitectureHealthWiring:
    enabled: bool
    pipeline_report: ArchitectureMetricsPipelineReport | None


def resolve_architecture_health_wiring(
    env: ApplicationEnvironmentProfile,
) -> ArchitectureHealthWiring:
    """Emit live architecture health metrics on product hosts."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ArchitectureHealthWiring(enabled=False, pipeline_report=None)
    if not env.governance_profile.architecture_health_metrics_enabled:
        return ArchitectureHealthWiring(enabled=False, pipeline_report=None)

    graph = build_catalog_capability_graph()
    metrics = compute_architecture_metrics(graph)
    snapshot = ArchitectureMetricsSnapshot(
        snapshot_id=f"{env.profile_id}-health",
        collected_at=datetime.now(UTC),
        report=metrics,
    )
    pipeline = build_metrics_pipeline_report(snapshots=[snapshot])
    return ArchitectureHealthWiring(enabled=True, pipeline_report=pipeline)
