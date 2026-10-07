# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R8 reachability resolver soundness and fail-closed closure."""

from __future__ import annotations

import subprocess
from pathlib import Path

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
    qualify_discover_composition_edges,
)
from tests.qualification.trace_x._trace_x_p4_r6_reachability_types import (
    CompositionEdge,
    ReachabilityVerdict,
)
from tests.qualification.trace_x._trace_x_p4_support import TRACE_X_P4_R8_START_HEAD

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_SANCTIONED_ROOT = "intergrax/applications/_shared/runtime_config_bridge.py"
_INTERMEDIATE_WIRING = "intergrax/runtime/execution/orchestration.py"
_FIXTURE_CALLS_UNBOUND = "tests/qualification/trace_x/r8_fixtures/r8_mod_calls_unbound_foo.py"
_FIXTURE_DEFINES_FOO = "tests/qualification/trace_x/r8_fixtures/r8_mod_defines_foo.py"
_FIXTURE_IMPORTS_FOO = "tests/qualification/trace_x/r8_fixtures/r8_mod_imports_foo.py"
_FIXTURE_IMPORTS_ALIAS = "tests/qualification/trace_x/r8_fixtures/r8_mod_imports_foo_alias.py"
_FIXTURE_MODULE_ALIAS = "tests/qualification/trace_x/r8_fixtures/r8_mod_module_import_foo.py"
_FIXTURE_DOTTED_MODULE = "tests/qualification/trace_x/r8_fixtures/r8_mod_dotted_module_import_foo.py"
_FIXTURE_PKG_FOO = "tests/qualification/trace_x/r8_fixtures/r8_package_module_defines_foo.py"
_FIXTURE_REACHABLE_UNRESOLVED = "tests/qualification/trace_x/r8_fixtures/r8_reachable_unresolved.py"
_FIXTURE_UNREACHABLE_UNRESOLVED = "tests/qualification/trace_x/r8_fixtures/r8_unreachable_unresolved.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_context() -> ReachabilityEvaluationContext:
    return ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
    )


def test_txp4r8_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R8_START_HEAD, "HEAD"],
    )


def test_txp4r8_q02_production_reachable_unresolved_sites_zero_on_head() -> None:
    graph = build_production_reachability_graph(_default_context())
    assert graph.production_reachable_unresolved_sites == ()


def test_txp4r8_q03_reachable_parser_unresolved_fails_qualification() -> None:
    edges = frozenset(
        {
            CompositionEdge(
                source_module_path=_SANCTIONED_ROOT,
                target_module_path=_FIXTURE_REACHABLE_UNRESOLVED,
                edge_kind="synthetic:r8_reachable_fixture",
            ),
        },
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=edges,
    )
    graph = build_production_reachability_graph(context)
    assert graph.production_reachable_unresolved_sites
    parity = compare_mechanical_reachability_to_expectations(
        evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context),
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
        graph=graph,
    )
    assert parity.production_reachability_proof_incomplete
    assert not parity.ok


def test_txp4r8_q04_unreachable_parser_unresolved_does_not_block_production_proof() -> None:
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        module_source_overrides=frozenset(
            {
                (
                    _FIXTURE_UNREACHABLE_UNRESOLVED,
                    (_REPO_ROOT / _FIXTURE_UNREACHABLE_UNRESOLVED).read_text(encoding="utf-8"),
                ),
            },
        ),
    )
    graph = build_production_reachability_graph(context)
    assert graph.unresolved_sites
    assert graph.production_reachable_unresolved_sites == ()
    parity = compare_mechanical_reachability_to_expectations(
        evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context),
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
        graph=graph,
    )
    assert parity.ok


def test_txp4r8_q05_unimported_same_name_class_does_not_create_edge() -> None:
    edges, unresolved = qualify_discover_composition_edges(
        frozenset({_FIXTURE_CALLS_UNBOUND, _FIXTURE_DEFINES_FOO}),
    )
    assert not unresolved
    assert not any(
        edge.source_module_path == _FIXTURE_CALLS_UNBOUND
        and edge.target_module_path == _FIXTURE_DEFINES_FOO
        for edge in edges
    )


def test_txp4r8_q06_explicit_imported_class_creates_edge() -> None:
    edges, unresolved = qualify_discover_composition_edges(
        frozenset({_FIXTURE_IMPORTS_FOO, _FIXTURE_DEFINES_FOO}),
    )
    assert not unresolved
    assert (
        _FIXTURE_IMPORTS_FOO,
        _FIXTURE_DEFINES_FOO,
        "instantiate:Foo",
    ) in {(e.source_module_path, e.target_module_path, e.edge_kind) for e in edges}


def test_txp4r8_q07_alias_import_creates_edge() -> None:
    edges, unresolved = qualify_discover_composition_edges(
        frozenset({_FIXTURE_IMPORTS_ALIAS, _FIXTURE_DEFINES_FOO}),
    )
    assert not unresolved
    assert (
        _FIXTURE_IMPORTS_ALIAS,
        _FIXTURE_DEFINES_FOO,
        "instantiate:Bar",
    ) in {(e.source_module_path, e.target_module_path, e.edge_kind) for e in edges}


def test_txp4r8_q08_module_qualified_and_aliased_import_create_edges() -> None:
    edges_alias, unresolved_alias = qualify_discover_composition_edges(
        frozenset({_FIXTURE_MODULE_ALIAS, _FIXTURE_PKG_FOO}),
    )
    assert not unresolved_alias
    assert (
        _FIXTURE_MODULE_ALIAS,
        _FIXTURE_PKG_FOO,
        "instantiate:Foo",
    ) in {(e.source_module_path, e.target_module_path, e.edge_kind) for e in edges_alias}

    edges_dotted, unresolved_dotted = qualify_discover_composition_edges(
        frozenset({_FIXTURE_DOTTED_MODULE, _FIXTURE_PKG_FOO}),
    )
    assert not unresolved_dotted
    assert (
        _FIXTURE_DOTTED_MODULE,
        _FIXTURE_PKG_FOO,
        "instantiate:Foo",
    ) in {(e.source_module_path, e.target_module_path, e.edge_kind) for e in edges_dotted}


def test_txp4r8_q09_r7_two_hop_inference_executor_regression() -> None:
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
    graph = build_production_reachability_graph(context)
    parity = compare_mechanical_reachability_to_expectations(
        evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context),
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
        graph=graph,
    )
    assert (
        "intergrax/runtime/execution/inference.py",
        "generate_structured",
    ) in parity.contradictions
    assert not parity.ok


def test_txp4r8_q10_all_non_production_surfaces_mechanically_resolved() -> None:
    context = _default_context()
    graph = build_production_reachability_graph(context)
    mechanical = evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context)
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
        graph=graph,
    )
    assert parity.ok
    assert len(mechanical) == 29
    assert all(result.verdict != ReachabilityVerdict.UNRESOLVED for result in mechanical)
    assert all(result.verdict != ReachabilityVerdict.PRODUCTION_REACHABLE for result in mechanical)
