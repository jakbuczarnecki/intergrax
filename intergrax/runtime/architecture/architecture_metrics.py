# © Artur Czarnecki. All rights reserved.

"""Architecture health metrics contracts and baseline computation (Phase V-AM.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pydantic import BaseModel, Field, field_validator

from intergrax.runtime.architecture.architecture_coverage import compute_architecture_coverage
from intergrax.runtime.architecture.capability_graph import CapabilityEdgeType, CapabilityGraph


class ArchitectureMetricThresholds(BaseModel):
    modularity_score_min: float = 0.50
    dependency_health_score_min: float = 0.80
    observability_coverage_min: float = 0.80
    governance_coverage_min: float = 0.80
    architecture_debt_index_max: float = 0.50


class ArchitectureMetricsSummary(BaseModel):
    modularity_score: float
    dependency_health_score: float
    observability_coverage: float
    governance_coverage: float
    architecture_debt_index: float
    nodes_total: int
    edges_total: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator(
        "modularity_score",
        "dependency_health_score",
        "observability_coverage",
        "governance_coverage",
        "architecture_debt_index",
    )
    @classmethod
    def _validate_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("Metric values must be in range [0.0, 1.0]")
        return value


class ArchitectureMetricsReport(BaseModel):
    schema_version: str = "1.0.0"
    mode: str = "report-only"
    thresholds: ArchitectureMetricThresholds = Field(default_factory=ArchitectureMetricThresholds)
    summary: ArchitectureMetricsSummary
    notes: list[str] = Field(default_factory=list)


def compute_architecture_metrics(graph: CapabilityGraph) -> ArchitectureMetricsReport:
    """
    Compute baseline architecture health metrics from capability graph structure.

    Phase V-AM.1 starts in report-only mode; thresholds are informative.
    """
    nodes_total = len(graph.nodes)
    edges_total = len(graph.edges)

    if nodes_total == 0:
        summary = ArchitectureMetricsSummary(
            modularity_score=0.0,
            dependency_health_score=0.0,
            observability_coverage=0.0,
            governance_coverage=0.0,
            architecture_debt_index=1.0,
            nodes_total=0,
            edges_total=0,
        )
        return ArchitectureMetricsReport(
            summary=summary,
            notes=["Capability graph is empty; metrics are pessimistic by definition."],
        )

    avg_degree = float(edges_total) / float(nodes_total)
    # Lower average degree usually means lower coupling.
    modularity_score = max(0.0, min(1.0, 1.0 / (1.0 + avg_degree)))

    dependency_edges = sum(1 for edge in graph.edges if edge.edge_type == CapabilityEdgeType.DEPENDS_ON)

    dependency_health_score = 1.0 if dependency_edges > 0 else 0.0
    coverage = compute_architecture_coverage(graph)
    observability_coverage = coverage.summary.observability_coverage
    governance_coverage = coverage.summary.governance_coverage

    debt_penalty = (
        (1.0 - modularity_score) * 0.35
        + (1.0 - dependency_health_score) * 0.25
        + (1.0 - observability_coverage) * 0.20
        + (1.0 - governance_coverage) * 0.20
    )
    architecture_debt_index = max(0.0, min(1.0, debt_penalty))

    summary = ArchitectureMetricsSummary(
        modularity_score=modularity_score,
        dependency_health_score=dependency_health_score,
        observability_coverage=observability_coverage,
        governance_coverage=governance_coverage,
        architecture_debt_index=architecture_debt_index,
        nodes_total=nodes_total,
        edges_total=edges_total,
    )
    return ArchitectureMetricsReport(
        summary=summary,
        notes=[
            "Report-only baseline for Phase V-AM.1.",
            "Thresholds are informative until hard enforcement is enabled.",
        ],
    )
