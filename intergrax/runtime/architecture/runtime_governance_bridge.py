# © Artur Czarnecki. All rights reserved.

"""Runtime wiring for Phase V architecture governance contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveGovernanceReport,
    AdaptiveLoopProposal,
    evaluate_adaptive_governance,
    evaluate_bounded_adaptive_loop,
)
from intergrax.runtime.architecture.graph_provenance import GraphTraceFieldBundle, build_graph_provenance_trace
from intergrax.runtime.architecture.graph_rag import GraphRagEdge, GraphRagNode
from intergrax.runtime.architecture.multi_agent_coordination import (
    PatternSelectionMatrixReport,
    PlanningConstraints,
    select_coordination_pattern,
)


class RuntimeGovernanceTraceMetadata(BaseModel):
    coordination_pattern: str = ""
    adaptive_governance_passed: bool = True
    graph_trace_id: str = ""
    reasons: list[str] = Field(default_factory=list)


class RuntimeArchitectureGovernanceBridge:
    """Typed bridge used by Nexus runtime to emit architecture governance metadata."""

    def select_coordination_pattern(
        self,
        constraints: PlanningConstraints,
    ) -> PatternSelectionMatrixReport:
        return select_coordination_pattern(constraints=constraints)

    def evaluate_adaptive_proposal(self, proposal: AdaptiveLoopProposal) -> AdaptiveGovernanceReport:
        return evaluate_adaptive_governance([proposal])

    def build_graph_trace_bundle(
        self,
        *,
        trace_id: str,
        graph_id: str,
        nodes: list[GraphRagNode],
        edges: list[GraphRagEdge],
        target_node_id: str,
    ) -> GraphTraceFieldBundle:
        return build_graph_provenance_trace(
            trace_id=trace_id,
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            target_node_id=target_node_id,
        )

    def build_trace_metadata(
        self,
        *,
        constraints: PlanningConstraints | None = None,
        adaptive_proposal: AdaptiveLoopProposal | None = None,
    ) -> RuntimeGovernanceTraceMetadata:
        reasons: list[str] = []
        pattern_name = ""
        if constraints is not None:
            selection = self.select_coordination_pattern(constraints)
            pattern_name = selection.decision.selected_pattern.value
        adaptive_passed = True
        if adaptive_proposal is not None:
            gate = evaluate_bounded_adaptive_loop(adaptive_proposal)
            adaptive_passed = gate.passed
            if gate.reasons:
                reasons.extend(gate.reasons)
        return RuntimeGovernanceTraceMetadata(
            coordination_pattern=pattern_name,
            adaptive_governance_passed=adaptive_passed,
            reasons=reasons,
        )
