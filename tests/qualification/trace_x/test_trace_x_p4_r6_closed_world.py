# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R6 mechanical reachability and real composition wrap certification."""

from __future__ import annotations

import subprocess

import pytest

from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.llm.model_call_runtime_evidence_adapter import (
    ModelCallRuntimeEvidenceAdapter,
    wrap_model_call_runtime_evidence,
)
from testing_support.builder import build_runtime_request_for_tests
from tests.qualification.trace_x._trace_x_p4_model_surface_registry import MODEL_CALL_SURFACE_REGISTRY
from tests.qualification.trace_x._trace_x_p4_non_production_reachability_registry import (
    NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
)
from tests.qualification.trace_x._trace_x_p4_production_composition_registry import (
    PRODUCTION_COMPOSITION_SITE_REGISTRY,
)
from tests.qualification.trace_x._trace_x_p4_production_composition_types import (
    ProductionCompositionSiteClassification,
    ProductionCompositionSiteKind,
    RegisteredProductionCompositionSite,
)
from tests.qualification.trace_x._trace_x_p4_r6_reachability_analysis import (
    ReachabilityEvaluationContext,
    build_synthetic_inference_executor_production_edge,
    compare_mechanical_reachability_to_expectations,
    evaluate_all_non_production_reachability,
)
from tests.qualification.trace_x._trace_x_p4_r6_reachability_types import (
    CompositionEdge,
    ModelConsumerSurface,
    ReachabilityVerdict,
)
from tests.qualification.trace_x._trace_x_p4_support import (
    TRACE_X_P4_R6_START_HEAD,
    compare_production_composition_sites_to_registry,
    discovered_production_composition_site_keys,
    llm_adapter_has_canonical_p4_evidence_wrapper,
    production_composition_supplies_inference_executor,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


class _MinimalQualificationLLM(BaseLLMAdapter):
    @property
    def context_window_tokens(self) -> int:
        return 4096

    async def generate_messages(self, messages, *, temperature=0.0, max_tokens=None, run_id=None):
        return "ok"


def _default_reachability_context() -> ReachabilityEvaluationContext:
    return ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
    )


def test_txp4r6_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R6_START_HEAD, "HEAD"],
    )


def test_txp4r6_q02_production_composition_parity_unchanged() -> None:
    result = compare_production_composition_sites_to_registry(
        discovered_production_composition_site_keys(),
        PRODUCTION_COMPOSITION_SITE_REGISTRY,
    )
    assert result.ok


def test_txp4r6_q03_mechanical_reachability_matches_non_production_expectations() -> None:
    mechanical = evaluate_all_non_production_reachability(
        MODEL_CALL_SURFACE_REGISTRY,
        _default_reachability_context(),
    )
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert parity.duplicate_registry_keys == frozenset()
    assert not parity.unknown
    assert not parity.orphan
    assert not parity.contradictions
    assert not parity.unresolved
    assert parity.ok
    assert all(
        result.verdict != ReachabilityVerdict.PRODUCTION_REACHABLE for result in mechanical
    )


def test_txp4r6_q04_inference_executor_mechanical_unreachable() -> None:
    mechanical = evaluate_all_non_production_reachability(
        MODEL_CALL_SURFACE_REGISTRY,
        _default_reachability_context(),
    )
    inference = next(
        row
        for row in mechanical
        if row.surface.path == "intergrax/runtime/execution/inference.py"
        and row.surface.method == "generate_structured"
    )
    assert inference.verdict == ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT
    assert not production_composition_supplies_inference_executor(PRODUCTION_COMPOSITION_SITE_REGISTRY)


def test_txp4r6_q05_materialize_runtime_config_wraps_injected_llm_adapter() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="txp4r6.wrap")
    request = build_runtime_request_for_tests(seed="txp4r6-wrap")
    raw: LLMAdapter = _MinimalQualificationLLM()
    assert not llm_adapter_has_canonical_p4_evidence_wrapper(raw)
    config = materialize_runtime_config(
        request,
        default_reference_harness(),
        env,
        llm_adapter=raw,
    )
    assert config.llm_adapter is not None
    assert llm_adapter_has_canonical_p4_evidence_wrapper(config.llm_adapter)


def test_txp4r6_q06_materialize_runtime_config_idempotent_when_adapter_already_wrapped() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="txp4r6.idempotent")
    request = build_runtime_request_for_tests(seed="txp4r6-idempotent")
    raw: LLMAdapter = _MinimalQualificationLLM()
    wrapped = wrap_model_call_runtime_evidence(raw)
    config = materialize_runtime_config(
        request,
        default_reference_harness(),
        env,
        llm_adapter=wrapped,
    )
    assert config.llm_adapter is not None
    assert isinstance(config.llm_adapter, ModelCallRuntimeEvidenceAdapter)
    assert llm_adapter_has_canonical_p4_evidence_wrapper(config.llm_adapter)
    assert llm_adapter_has_canonical_p4_evidence_wrapper(
        wrap_model_call_runtime_evidence(config.llm_adapter),
    )


def test_txp4r6_q07_r6_n1_synthetic_inference_executor_edge_contradicts_registry() -> None:
    edge = build_synthetic_inference_executor_production_edge(
        "intergrax/runtime/nexus/execution/graph_executor.py",
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=frozenset({edge}),
    )
    mechanical = evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context)
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert (
        "intergrax/runtime/execution/inference.py",
        "generate_structured",
    ) in parity.contradictions
    assert not parity.ok


def test_txp4r6_q08_r6_n2_synthetic_auxiliary_production_link_contradicts_registry() -> None:
    auxiliary = ModelConsumerSurface(
        path="intergrax/agents/authoring/llm_router.py",
        method="generate_messages",
    )
    edge = CompositionEdge(
        source_module_path="intergrax/applications/_shared/runtime_config_bridge.py",
        target_module_path=auxiliary.path,
        edge_kind="synthetic:auxiliary_consumer",
    )
    context = ReachabilityEvaluationContext(
        production_composition_registry=PRODUCTION_COMPOSITION_SITE_REGISTRY,
        extra_composition_edges=frozenset({edge}),
    )
    mechanical = evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context)
    parity = compare_mechanical_reachability_to_expectations(
        mechanical,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert auxiliary.key in parity.contradictions
    assert not parity.ok


def test_txp4r6_q09_synthetic_inference_executor_registry_injection_flips_verdict() -> None:
    synthetic_registry = PRODUCTION_COMPOSITION_SITE_REGISTRY + (
        RegisteredProductionCompositionSite(
            path="intergrax/runtime/nexus/execution/graph_executor.py",
            site_kind=ProductionCompositionSiteKind.STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW,
            line_number=9999,
            classification=ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROUTER,
            composition_anchor="synthetic",
            inference_executor_supplied=True,
            p4_wrap_applied=None,
            evidence_nodeid="synthetic",
        ),
    )
    context = ReachabilityEvaluationContext(production_composition_registry=synthetic_registry)
    mechanical = evaluate_all_non_production_reachability(MODEL_CALL_SURFACE_REGISTRY, context)
    inference = next(
        row
        for row in mechanical
        if row.surface.path == "intergrax/runtime/execution/inference.py"
    )
    assert inference.verdict == ReachabilityVerdict.PRODUCTION_REACHABLE
