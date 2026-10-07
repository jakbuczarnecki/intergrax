# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R7 transitive production reachability certification."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.trace_x._trace_x_p4_model_surface_registry import MODEL_CALL_SURFACE_REGISTRY
from tests.qualification.trace_x._trace_x_p4_non_production_reachability_registry import (
    NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
)
from tests.qualification.trace_x._trace_x_p4_production_composition_registry import (
    PRODUCTION_COMPOSITION_SITE_REGISTRY,
)
from tests.qualification.trace_x._trace_x_p4_r6_reachability_analysis import (
    ReachabilityEvaluationContext,
    build_production_reachability_graph,
    build_synthetic_inference_executor_production_edge,
    compare_mechanical_reachability_to_expectations,
    evaluate_all_non_production_reachability,
    evaluate_non_production_model_surface_reachability,
)
from tests.qualification.trace_x._trace_x_p4_r6_reachability_types import (
    CompositionEdge,
    ModelConsumerSurface,
    ReachabilityVerdict,
)
from tests.qualification.trace_x._trace_x_p4_support import TRACE_X_P4_R7_START_HEAD

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_SANCTIONED_ROOT = "intergrax/applications/_shared/runtime_config_bridge.py"
_INTERMEDIATE_WIRING = "intergrax/runtime/execution/orchestration.py"
_AGENTS_NON_PRODUCTION = ModelConsumerSurface(
    path="agents/model_routing_qualifier/steps/model_routing_job.py",
    method="generate_messages",
)
_APPLICATIONS_NON_PRODUCTION = ModelConsumerSurface(
    path="applications/local_workspace_application/workspaces/ask_answer_assembler.py",
    method="generate_messages",
)
_STREAM_SURFACE = ModelConsumerSurface(
    path="intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
    method="stream_messages",
)
_CONFORMANCE_STREAM_CALLER = "intergrax/llm_adapters/_shared/conformance.py"


def _default_context() -> ReachabilityEvaluationContext:
    return ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
    )


def test_txp4r7_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R7_START_HEAD, "HEAD"],
    )


def test_txp4r7_q02_transitive_import_closure_exceeds_seed_set() -> None:
    graph = build_production_reachability_graph(_default_context())
    assert graph.sanctioned_seed_modules
    assert len(graph.import_closure_modules) > len(graph.sanctioned_seed_modules)
    assert graph.sanctioned_seed_modules.issubset(graph.import_closure_modules)


def test_txp4r7_q03_mechanical_reachability_matches_non_production_expectations() -> None:
    mechanical = evaluate_all_non_production_reachability(
        MODEL_CALL_SURFACE_REGISTRY,
        _default_context(),
    )
    graph = build_production_reachability_graph(_default_context())
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
        graph=graph,
    )
    assert parity.ok
    assert len(mechanical) == 29
    assert all(
        result.verdict != ReachabilityVerdict.PRODUCTION_REACHABLE for result in mechanical
    )


def test_txp4r7_q04_r7_n1_two_hop_inference_executor_contradicts_registry() -> None:
    edges = frozenset(
        {
            CompositionEdge(
                source_module_path=_SANCTIONED_ROOT,
                target_module_path=_INTERMEDIATE_WIRING,
                edge_kind="synthetic:intermediate_wiring",
            ),
            build_synthetic_inference_executor_production_edge(_INTERMEDIATE_WIRING),
        },
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=edges,
    )
    parity = compare_mechanical_reachability_to_expectations(
        evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context),
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert (
        "intergrax/runtime/execution/inference.py",
        "generate_structured",
    ) in parity.contradictions
    assert not parity.ok


def test_txp4r7_q05_r7_n2_agents_path_not_semantic_shortcut() -> None:
    edges = frozenset(
        {
            CompositionEdge(
                source_module_path=_SANCTIONED_ROOT,
                target_module_path=_INTERMEDIATE_WIRING,
                edge_kind="synthetic:intermediate_wiring",
            ),
            CompositionEdge(
                source_module_path=_INTERMEDIATE_WIRING,
                target_module_path=_AGENTS_NON_PRODUCTION.path,
                edge_kind="synthetic:agents_consumer",
            ),
        },
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=edges,
    )
    mechanical = evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context)
    agent = next(row for row in mechanical if row.surface == _AGENTS_NON_PRODUCTION)
    assert agent.verdict == ReachabilityVerdict.PRODUCTION_REACHABLE
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert _AGENTS_NON_PRODUCTION.key in parity.contradictions


def test_txp4r7_q06_r7_n3_applications_path_not_semantic_shortcut() -> None:
    edges = frozenset(
        {
            CompositionEdge(
                source_module_path=_SANCTIONED_ROOT,
                target_module_path=_INTERMEDIATE_WIRING,
                edge_kind="synthetic:intermediate_wiring",
            ),
            CompositionEdge(
                source_module_path=_INTERMEDIATE_WIRING,
                target_module_path=_APPLICATIONS_NON_PRODUCTION.path,
                edge_kind="synthetic:applications_consumer",
            ),
        },
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=edges,
    )
    mechanical = evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context)
    app_row = next(row for row in mechanical if row.surface == _APPLICATIONS_NON_PRODUCTION)
    assert app_row.verdict == ReachabilityVerdict.PRODUCTION_REACHABLE
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert _APPLICATIONS_NON_PRODUCTION.key in parity.contradictions


def test_txp4r7_q07_r7_n4_transitive_stream_invocation_production_reachable() -> None:
    edges = frozenset(
        {
            CompositionEdge(
                source_module_path=_SANCTIONED_ROOT,
                target_module_path=_INTERMEDIATE_WIRING,
                edge_kind="synthetic:intermediate_wiring",
            ),
            CompositionEdge(
                source_module_path=_INTERMEDIATE_WIRING,
                target_module_path=_CONFORMANCE_STREAM_CALLER,
                edge_kind="synthetic:stream_caller",
            ),
        },
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=edges,
    )
    graph = build_production_reachability_graph(context)
    result = evaluate_non_production_model_surface_reachability(
        _STREAM_SURFACE,
        model_registry=MODEL_CALL_SURFACE_REGISTRY,
        context=context,
        all_non_production_surfaces=frozenset({_STREAM_SURFACE}),
        graph=graph,
    )
    assert result.verdict == ReachabilityVerdict.PRODUCTION_REACHABLE


def test_txp4r7_q08_r7_n5_unresolved_edge_fail_closed() -> None:
    edges = frozenset(
        {
            CompositionEdge(
                source_module_path=_SANCTIONED_ROOT,
                target_module_path=_AGENTS_NON_PRODUCTION.path,
                edge_kind="unresolved:ambiguous_wiring",
            ),
        },
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=edges,
    )
    graph = build_production_reachability_graph(context)
    result = evaluate_non_production_model_surface_reachability(
        _AGENTS_NON_PRODUCTION,
        model_registry=MODEL_CALL_SURFACE_REGISTRY,
        context=context,
        all_non_production_surfaces=frozenset({_AGENTS_NON_PRODUCTION}),
        graph=graph,
    )
    assert result.verdict == ReachabilityVerdict.UNRESOLVED
    parity = compare_mechanical_reachability_to_expectations(
        evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context),
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert _AGENTS_NON_PRODUCTION.key in parity.unresolved
    assert not parity.ok


def test_txp4r7_q09_graph_snapshot_has_composition_edges() -> None:
    graph = build_production_reachability_graph(_default_context())
    assert graph.composition_edges
    assert graph.production_reachable_modules
