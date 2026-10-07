# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R5 production composition reachability and P4 wrap certification."""

from __future__ import annotations

import inspect
import subprocess
from dataclasses import dataclass

import pytest

from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.llm.model_call_runtime_evidence_adapter import wrap_model_call_runtime_evidence
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
from tests.qualification.trace_x._trace_x_p4_registry_types import ModelCallSurfaceClassification
from tests.qualification.trace_x._trace_x_p4_support import (
    TRACE_X_P4_R5_START_HEAD,
    compare_non_production_reachability_to_model_registry,
    compare_production_composition_sites_to_registry,
    discover_alternate_p4_attribution_seam_instantiations_ast,
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


@dataclass(frozen=True, slots=True)
class SyntheticProductionModelReachabilityLink:
    composition_root_path: str
    model_call_path: str
    model_call_method: str


def _sanctioned_production_composition_root_paths() -> frozenset[str]:
    return frozenset(
        row.path
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.classification
        in (
            ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROOT,
            ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROUTER,
        )
    )


def _non_production_model_call_keys() -> frozenset[tuple[str, str]]:
    return frozenset(
        row.key
        for row in MODEL_CALL_SURFACE_REGISTRY
        if row.classification == ModelCallSurfaceClassification.NON_PRODUCTION
    )


def production_reachability_links_are_consistent(
    links: frozenset[SyntheticProductionModelReachabilityLink],
    *,
    non_production_keys: frozenset[tuple[str, str]],
    sanctioned_roots: frozenset[str],
) -> bool:
    for link in links:
        if link.composition_root_path not in sanctioned_roots:
            continue
        key = (link.model_call_path, link.model_call_method)
        if key in non_production_keys:
            return False
    return True


def test_txp4r5_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R5_START_HEAD, "HEAD"],
    )


def test_txp4r5_q02_production_composition_registry_is_static_not_discovery_derived() -> None:
    registry_source = inspect.getsource(
        __import__(
            "tests.qualification.trace_x._trace_x_p4_production_composition_registry",
            fromlist=["PRODUCTION_COMPOSITION_SITE_REGISTRY"],
        ),
    )
    assert "discover_production_composition_sites_ast" not in registry_source


def test_txp4r5_q03_non_production_reachability_registry_is_static() -> None:
    registry_source = inspect.getsource(
        __import__(
            "tests.qualification.trace_x._trace_x_p4_non_production_reachability_registry",
            fromlist=["NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY"],
        ),
    )
    assert "discover_model_call_surfaces_ast" not in registry_source


def test_txp4r5_q04_production_composition_registry_parity() -> None:
    result = compare_production_composition_sites_to_registry(
        discovered_production_composition_site_keys(),
        PRODUCTION_COMPOSITION_SITE_REGISTRY,
    )
    assert result.duplicate_registry_keys == frozenset()
    assert not result.unknown, f"unknown composition sites: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan composition registry rows: {sorted(result.orphan)}"
    assert result.ok


def test_txp4r5_q05_production_composition_negative_sensitivity() -> None:
    baseline = compare_production_composition_sites_to_registry(
        discovered_production_composition_site_keys(),
        PRODUCTION_COMPOSITION_SITE_REGISTRY,
    )
    assert baseline.ok
    synthetic = (
        "intergrax/runtime/example_new_composition_root.py",
        ProductionCompositionSiteKind.STRATEGY_EXECUTION_ROUTER_INSTANTIATION,
        1,
    )
    augmented = frozenset(discovered_production_composition_site_keys() | {synthetic})
    failed = compare_production_composition_sites_to_registry(augmented, PRODUCTION_COMPOSITION_SITE_REGISTRY)
    assert synthetic in failed.unknown
    assert not failed.ok


def test_txp4r5_q06_inference_executor_unreachable_from_sanctioned_routers() -> None:
    assert not production_composition_supplies_inference_executor(PRODUCTION_COMPOSITION_SITE_REGISTRY)
    inference_kw_sites = [
        row
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.site_kind == ProductionCompositionSiteKind.STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW
    ]
    assert inference_kw_sites == []
    inference_row = next(
        row
        for row in MODEL_CALL_SURFACE_REGISTRY
        if row.path == "intergrax/runtime/execution/inference.py"
        and row.method == "generate_structured"
    )
    assert inference_row.classification == ModelCallSurfaceClassification.NON_PRODUCTION


def test_txp4r5_q07_build_governed_inference_executor_no_production_caller() -> None:
    call_sites = [
        row
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.site_kind == ProductionCompositionSiteKind.GOVERNED_INFERENCE_EXECUTOR_CALL
    ]
    assert call_sites == []
    factory_rows = [
        row
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.site_kind == ProductionCompositionSiteKind.GOVERNED_INFERENCE_EXECUTOR_FACTORY_DEF
    ]
    assert len(factory_rows) == 1
    assert factory_rows[0].classification == (
        ProductionCompositionSiteClassification.INTERNAL_FACTORY_NO_PRODUCTION_CALLER
    )


def test_txp4r5_q08_non_production_surfaces_have_reachability_proof() -> None:
    result = compare_non_production_reachability_to_model_registry(
        MODEL_CALL_SURFACE_REGISTRY,
        NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY,
    )
    assert result.duplicate_registry_keys == frozenset()
    assert not result.unknown
    assert not result.orphan
    assert result.ok


def sanctioned_inference_executor_injection_absent(
    registry: tuple[RegisteredProductionCompositionSite, ...],
) -> bool:
    return not production_composition_supplies_inference_executor(registry)


def test_txp4r5_q09_r5_n1_synthetic_inference_executor_injection_fails() -> None:
    assert sanctioned_inference_executor_injection_absent(PRODUCTION_COMPOSITION_SITE_REGISTRY)

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
    assert not sanctioned_inference_executor_injection_absent(synthetic_registry)


def test_txp4r5_q10_sanctioned_runtime_config_llm_adapter_wrap_propagation() -> None:
    raw: LLMAdapter = _MinimalQualificationLLM()
    assert not llm_adapter_has_canonical_p4_evidence_wrapper(raw)
    wrapped = wrap_model_call_runtime_evidence(raw)
    assert llm_adapter_has_canonical_p4_evidence_wrapper(wrapped)
    assert llm_adapter_has_canonical_p4_evidence_wrapper(wrap_model_call_runtime_evidence(wrapped))
    wrap_rows = [
        row
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.site_kind == ProductionCompositionSiteKind.SANCTIONED_LLM_ADAPTER_P4_WRAP
    ]
    assert len(wrap_rows) == 1
    assert wrap_rows[0].path == "intergrax/applications/_shared/runtime_config_bridge.py"


def test_txp4r5_q11_r5_n2_non_production_production_link_fails() -> None:
    sanctioned = _sanctioned_production_composition_root_paths()
    non_production = _non_production_model_call_keys()
    assert production_reachability_links_are_consistent(
        frozenset(),
        non_production_keys=non_production,
        sanctioned_roots=sanctioned,
    )
    synthetic_link = SyntheticProductionModelReachabilityLink(
        composition_root_path="intergrax/runtime/nexus/execution/graph_executor.py",
        model_call_path="intergrax/runtime/execution/inference.py",
        model_call_method="generate_structured",
    )
    assert not production_reachability_links_are_consistent(
        frozenset({synthetic_link}),
        non_production_keys=non_production,
        sanctioned_roots=sanctioned,
    )


def test_txp4r5_q12_r5_n3_raw_adapter_fails_p4_wrap_proof() -> None:
    raw: LLMAdapter = _MinimalQualificationLLM()
    assert not llm_adapter_has_canonical_p4_evidence_wrapper(raw)


def test_txp4r5_q13_r5_n4_zero_alternate_p4_attribution_seams() -> None:
    assert discover_alternate_p4_attribution_seam_instantiations_ast() == frozenset()


def test_txp4r5_q14_production_reachable_primary_surfaces_p4_protected() -> None:
    primary = [
        row
        for row in MODEL_CALL_SURFACE_REGISTRY
        if row.classification == ModelCallSurfaceClassification.CANONICAL_PRIMARY
    ]
    assert primary
    assert all(
        row.path == "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py" for row in primary
    )
    sanctioned_wrap = [
        row
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.classification == ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROOT
        and row.p4_wrap_applied is True
    ]
    assert len(sanctioned_wrap) == 1
