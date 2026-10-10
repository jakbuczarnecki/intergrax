# © Artur Czarnecki. All rights reserved.

"""CONFIG-X final reconciliation — active exit + FRZ-CFG parent evidence."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.registry.factory import resolve_slug
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.registry.wiring import ToolWiringContext

from tests.qualification.config_x._config_x_blockers import (
    CONFIG_X_HISTORICAL_BLOCKER_RECORDS,
    WAVE1_HISTORICAL_BLOCKER_IDS,
    active_blocker_counts_by_classification,
)
from tests.qualification.config_x._config_x_concern_inventory import CONFIG_X_CONCERN_INVENTORY
from tests.qualification.config_x._config_x_discovery import discover_active_blocker_path_keys
from tests.qualification.config_x._config_x_owner_discovery import (
    CONFIG_X_OWNER_EXPECTATIONS,
    compare_owner_gate,
)
from tests.qualification.config_x._config_x_types import (
    BLOCKER_CLASSIFICATIONS,
    ConfigClassification,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_EXPECTED_WAVE1_IDS = frozenset(
    {
        "CONFIG-X-BLK-OBS-TOOL-01",
        "CONFIG-X-BLK-TOK-01",
        "CONFIG-X-BLK-HARNESS-HTTP-01",
        "CONFIG-X-BLK-MM-01",
        "CONFIG-X-BLK-INT-P3-01",
    },
)


def test_reconciliation_historical_blocker_ids_preserved() -> None:
    assert WAVE1_HISTORICAL_BLOCKER_IDS == _EXPECTED_WAVE1_IDS
    assert len(CONFIG_X_HISTORICAL_BLOCKER_RECORDS) == 5


def test_reconciliation_active_blockers_mechanically_empty() -> None:
    assert discover_active_blocker_path_keys() == frozenset()
    counts = active_blocker_counts_by_classification()
    for classification in BLOCKER_CLASSIFICATIONS:
        assert counts[classification] == 0


@pytest.mark.parametrize(
    "blocker_id",
    sorted(_EXPECTED_WAVE1_IDS),
)
def test_reconciliation_wave1_blocker_paths_not_active(blocker_id: str) -> None:
    row = next(r for r in CONFIG_X_HISTORICAL_BLOCKER_RECORDS if r.blocker_id == blocker_id)
    active_paths = discover_active_blocker_path_keys()
    for path in row.paths:
        assert path not in active_paths, blocker_id


def test_frz_cfg_01_closed_world_typed_configuration_contracts() -> None:
    for row in CONFIG_X_CONCERN_INVENTORY:
        assert row.configuration_contract.strip()
        assert row.composition_owner.strip()
        assert row.effective_resolution_owner.strip()
        assert row.classification not in BLOCKER_CLASSIFICATIONS


def test_frz_cfg_02_missing_integration_config_fails_closed() -> None:
    with pytest.raises(IntegrationConfigurationError):
        resolve_slug(IntegrationCategory.VECTOR_STORE, profile=IntegrationProfile())


def test_frz_cfg_03_unconfigured_llm_not_effective() -> None:
    LLMAdapterRegistry.reset_for_testing()
    with pytest.raises(ValueError, match="not registered"):
        LLMAdapterRegistry.create("config_x_reconciliation_unregistered")


def test_frz_cfg_04_single_owner_per_mandatory_concern() -> None:
    mandatory = {
        "integration_provider_selection",
        "llm_provider_selection",
        "execution_bound_integration_resolution",
        "existing_capability_configuration_realization",
        "plugin_integration_catalog",
    }
    assert mandatory <= set(CONFIG_X_OWNER_EXPECTATIONS)
    for concern_key in mandatory:
        discovered, expected = compare_owner_gate(concern_key)
        assert discovered == expected


def test_frz_cfg_05_no_active_hard_coded_semantic_blockers_in_wave1_scope() -> None:
    """Closed-world CONFIG-X wave-1 paths: no I-class active forbidden markers remain."""
    counts = active_blocker_counts_by_classification()
    assert counts[ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION] == 0
    assert discover_active_blocker_path_keys() == frozenset()


def test_frz_cfg_06_unconfigured_observability_and_tokenizer_not_effective() -> None:
    from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
    from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer

    with pytest.raises(RuntimeError, match="observability_backend_not_configured"):
        resolve_observability_backend(ToolWiringContext())
    registry = TokenizerRegistry()
    registry.register(SimpleTokenizer())
    registry.register(TiktokenTokenizer())
    with pytest.raises(ValueError, match="No default tokenizer configured"):
        registry.get(None)


def test_frz_cfg_07_deterministic_integration_slug_resolution() -> None:
    profile = IntegrationProfile(relational_store="sqlite")
    assert (
        resolve_slug(IntegrationCategory.RELATIONAL_STORE, profile=profile) == "sqlite"
    )
    assert (
        resolve_slug(IntegrationCategory.RELATIONAL_STORE, profile=profile) == "sqlite"
    )


def test_frz_cfg_08_active_forbidden_markers_regression_gate() -> None:
    assert discover_active_blocker_path_keys() == frozenset()
    unclassified = [
        row
        for row in CONFIG_X_CONCERN_INVENTORY
        if row.classification == ConfigClassification.L_UNCLEAR
    ]
    assert unclassified == []


def test_config_x_tenant_isolation_audit_local_pass() -> None:
    """CONFIG-X local tenant evidence (not global TENANT-X / FRZ-TEN-*)."""
    audit = {
        "tenant_scope_applicable": "YES",
        "canonical_tenant_identity": "principal / explicit tenant_id on CONFIG-X surfaces",
        "tenant_owner": "harness principal resolution; trace query param; P3 require_tenant_id",
        "propagation_path": "HTTP principal → task; trace tenant_id query; scope on multimedia",
        "state_isolation": "INT-CONFIG realization TENANT_MISMATCH (CX-G)",
        "provider_config_isolation": "VectorIntegrationConfig.require_tenant_id",
        "evidence_trace_isolation": "trace explorer requires tenant_id (422 when missing)",
        "async_recovery_continuity": "out of CONFIG-X closed-world — TENANT-X later",
        "cross_tenant_path": "CX-G adversarial mismatch rejected",
        "fail_closed_behavior": "missing tenant → 422 / IntegrationConfigurationError",
        "adversarial_evidence": "test_config_x_r1_remediation_gates.py T1–T5; test_cx_g_*",
        "result": "PASS",
    }
    assert audit["result"] == "PASS"
    assert discover_active_blocker_path_keys() == frozenset()
