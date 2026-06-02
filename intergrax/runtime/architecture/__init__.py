# © Artur Czarnecki. All rights reserved.

"""Phase V architecture hardening contracts and report builders."""

from intergrax.runtime.architecture.agent_certification import (
    AgentCertificationEvidence,
    AgentCertificationEvaluation,
    AgentCertificationGate,
    AgentCertificationOwner,
    GateCheckStatus,
    evaluate_agent_certification,
)
from intergrax.runtime.architecture.architecture_metrics import (
    ArchitectureMetricThresholds,
    ArchitectureMetricsReport,
    ArchitectureMetricsSummary,
    compute_architecture_metrics,
)
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityGraphVersion,
    CapabilityNode,
    CapabilityNodeType,
    build_catalog_capability_graph,
)
from intergrax.runtime.architecture.capability_graph_compatibility import (
    CapabilityCompatibilityReport,
    CompatibilityIssue,
    CompatibilitySeverity,
    evaluate_capability_graph_compatibility,
)
from intergrax.runtime.architecture.capability_graph_lineage import (
    CapabilityImpactRecord,
    CapabilityImpactReport,
    CapabilityLineageRecord,
    CapabilityLineageReport,
    build_capability_impact_report,
    build_capability_lineage_report,
)

__all__ = [
    "AgentCertificationEvidence",
    "AgentCertificationEvaluation",
    "AgentCertificationGate",
    "AgentCertificationOwner",
    "ArchitectureMetricThresholds",
    "ArchitectureMetricsReport",
    "ArchitectureMetricsSummary",
    "CapabilityEdge",
    "CapabilityEdgeType",
    "CapabilityGraph",
    "CapabilityGraphVersion",
    "CapabilityCompatibilityReport",
    "CapabilityImpactRecord",
    "CapabilityImpactReport",
    "CapabilityLineageRecord",
    "CapabilityLineageReport",
    "CapabilityNode",
    "CapabilityNodeType",
    "CompatibilityIssue",
    "CompatibilitySeverity",
    "GateCheckStatus",
    "evaluate_agent_certification",
    "build_catalog_capability_graph",
    "build_capability_impact_report",
    "build_capability_lineage_report",
    "compute_architecture_metrics",
    "evaluate_capability_graph_compatibility",
]
