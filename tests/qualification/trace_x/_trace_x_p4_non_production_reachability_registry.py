# © Artur Czarnecki. All rights reserved.

"""Explicit TRACE-X-P4-R5 NON_PRODUCTION model-call reachability registry."""

from __future__ import annotations

from typing import Final

from tests.qualification.trace_x._trace_x_p4_production_composition_types import (
    NonProductionReachabilityReason,
    RegisteredNonProductionModelReachability,
)

NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY: Final[tuple[RegisteredNonProductionModelReachability, ...]] = (
    RegisteredNonProductionModelReachability(
        path='agents/model_routing_qualifier/steps/model_routing_job.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='No sanctioned Tier-0/1 Nexus GraphExecutor composition root references this agent job surface',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='agents/tool_selection_qualifier/steps/tool_selection_job.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='No sanctioned Tier-0/1 Nexus GraphExecutor composition root references this agent job surface',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='agents/tool_selection_qualifier/steps/tool_selection_job.py',
        method='generate_with_tools',
        reason=NonProductionReachabilityReason.TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='No sanctioned Tier-0/1 Nexus GraphExecutor composition root references this agent job surface',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='agents/web_search_qualifier/source_selection/llm_selector.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='No sanctioned Tier-0/1 Nexus GraphExecutor composition root references this agent job surface',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='agents/web_search_qualifier/steps/web_search_job.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='No sanctioned Tier-0/1 Nexus GraphExecutor composition root references this agent job surface',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='applications/local_workspace_application/workspaces/ask_answer_assembler.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.TIER3_APPLICATION_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='Tier-3 workspace assembler; not materialize_runtime_config sanctioned Nexus primary root',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='applications/local_workspace_application/workspaces/hybrid_ask_answer_assembler.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.TIER3_APPLICATION_NOT_SANCTIONED_COMPOSITION_ROOT,
        reachability_proof='Tier-3 workspace assembler; not materialize_runtime_config sanctioned Nexus primary root',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/agents/authoring/llm_router.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/codecraft/llm_codegen_adapter.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/memory/strategies/defaults/llm_extraction.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/rag/contextual/chunk_enricher.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/rag/graph/indexer/community_report_graph_indexer.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/rag/graph/indexer/llm_graph_indexer.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/rag/query/query_expander.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/rag/retrieval/query_refiner.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/rag/routing/llm_tier_classifier.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/execution/inference.py',
        method='generate_structured',
        reason=NonProductionReachabilityReason.INTERNAL_INFERENCE_BACKEND_UNREACHABLE,
        reachability_proof='Zero production GOVERNED_INFERENCE_EXECUTOR_CALL and zero STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW; factory only inside inference_composition',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/llm/model_call_runtime_evidence_adapter.py',
        method='stream_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/llm/model_call_runtime_evidence_adapter.py',
        method='stream_with_tools',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/nexus/llm_task_classifier.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/nexus/planning/nexus_plan_bridge.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/nexus/tools/tool_planning_service.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/nexus/tools/tool_planning_service.py',
        method='generate_with_tools',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/organization/organization_profile_instructions_service.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/runtime/user_profile/user_profile_instructions_service.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/supervisor/supervisor.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/tools/providers/eval/judge.py',
        method='generate_structured',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/websearch/service/websearch_answerer.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
    RegisteredNonProductionModelReachability(
        path='intergrax/websearch/service/websearch_context_generator.py',
        method='generate_messages',
        reason=NonProductionReachabilityReason.AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION,
        reachability_proof='No sanctioned production composition root supplies adapter consumer chain to this module',
        evidence_nodeid="test_trace_x_p4_r5_closed_world.py::test_txp4r5_q08_non_production_surfaces_have_reachability_proof",
    ),
)

__all__ = ["NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY"]
