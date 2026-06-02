from __future__ import annotations

from intergrax.runtime.architecture.architecture_metrics import compute_architecture_metrics
from intergrax.runtime.architecture.architecture_metrics_pipeline import (
    ArchitectureMetricsSnapshot,
    MetricsTrendDirection,
    build_metrics_pipeline_report,
)
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)


def _graph_with_governance() -> CapabilityGraph:
    return CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
            CapabilityNode(node_id="agent:research", node_type=CapabilityNodeType.AGENT),
            CapabilityNode(node_id="policy:runtime", node_type=CapabilityNodeType.POLICY),
            CapabilityNode(node_id="evaluation:runtime", node_type=CapabilityNodeType.EVALUATION),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="tool:rag.retrieve",
                target_node_id="integration:sqlite",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="agent:research",
                target_node_id="tool:rag.retrieve",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="agent:research",
                target_node_id="policy:runtime",
                edge_type=CapabilityEdgeType.CONSTRAINED_BY,
            ),
            CapabilityEdge(
                source_node_id="evaluation:runtime",
                target_node_id="agent:research",
                edge_type=CapabilityEdgeType.EVALUATES,
            ),
        ],
    )


def test_metrics_pipeline_gate_fails_when_thresholds_are_not_met() -> None:
    report = compute_architecture_metrics(_graph_with_governance())
    pipeline = build_metrics_pipeline_report(
        snapshots=[ArchitectureMetricsSnapshot(snapshot_id="current", report=report)]
    )
    assert pipeline.gate_result.passed is False
    assert pipeline.gate_result.reasons


def test_metrics_pipeline_gate_passes_with_relaxed_thresholds() -> None:
    report = compute_architecture_metrics(_graph_with_governance())
    report.thresholds.modularity_score_min = 0.20
    report.thresholds.dependency_health_score_min = 0.20
    report.thresholds.observability_coverage_min = 0.10
    report.thresholds.governance_coverage_min = 0.10
    report.thresholds.architecture_debt_index_max = 1.0
    pipeline = build_metrics_pipeline_report(
        snapshots=[ArchitectureMetricsSnapshot(snapshot_id="current", report=report)]
    )
    assert pipeline.gate_result.passed is True


def test_metrics_pipeline_returns_trend_for_two_snapshots() -> None:
    previous = ArchitectureMetricsSnapshot(
        snapshot_id="previous",
        report=compute_architecture_metrics(_graph_with_governance()),
    )
    current = ArchitectureMetricsSnapshot(
        snapshot_id="current",
        report=compute_architecture_metrics(_graph_with_governance()),
    )
    pipeline = build_metrics_pipeline_report(snapshots=[previous, current])
    assert pipeline.trend is not None
    assert pipeline.trend.modularity_trend == MetricsTrendDirection.STABLE
