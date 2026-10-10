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

from tests.qualification.config_x._config_x_activation_families import (
    discover_activation_bypass_findings,
)
from tests.qualification.config_x._config_x_blockers import (
    CONFIG_X_ACTIVE_BLOCKER_RECORDS,
    CONFIG_X_HISTORICAL_BLOCKER_RECORDS,
    WAVE1_HISTORICAL_BLOCKER_IDS,
    active_blocker_counts_by_classification,
)
from tests.qualification.config_x._config_x_concern_inventory import CONFIG_X_CONCERN_INVENTORY
from tests.qualification.config_x._config_x_current_classification import (
    discover_duplicate_configuration_authority_paths,
    sweep_concern_classification_evidence,
)
from tests.qualification.config_x._config_x_discovery import discover_active_blocker_path_keys
from tests.qualification.config_x._config_x_owner_discovery import (
    CONFIG_X_OWNER_EXPECTATIONS,
    compare_owner_gate,
)
from tests.qualification.config_x._config_x_semantic_production_scan import (
    discover_named_constant_semantic_blind_spot_paths,
    discover_semantic_i_blocker_paths,
    discover_unclassified_provider_surface_paths,
    frz_cfg_05_named_constant_blind_spot_count,
    frz_cfg_05_semantic_i_blocker_count,
    frz_cfg_05_unclassified_provider_surface_count,
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

_FRZ_CFG_PASS_CANDIDATE = "PASS CANDIDATE"


def test_reconciliation_historical_blocker_ids_preserved() -> None:
    assert WAVE1_HISTORICAL_BLOCKER_IDS == _EXPECTED_WAVE1_IDS
    assert len(CONFIG_X_HISTORICAL_BLOCKER_RECORDS) == 5


def test_reconciliation_active_blockers_mechanically_empty() -> None:
    assert discover_active_blocker_path_keys() == frozenset()
    counts = active_blocker_counts_by_classification()
    for classification in BLOCKER_CLASSIFICATIONS:
        assert counts[classification] == 0
    assert len(CONFIG_X_ACTIVE_BLOCKER_RECORDS) == 0


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
    evidence = sweep_concern_classification_evidence()
    assert len(evidence) == len(CONFIG_X_CONCERN_INVENTORY) == 54
    for row, item in zip(CONFIG_X_CONCERN_INVENTORY, evidence, strict=True):
        assert row.configuration_contract.strip()
        assert row.composition_owner.strip()
        assert row.effective_resolution_owner.strip()
        assert item.mechanical_classification not in BLOCKER_CLASSIFICATIONS
        assert item.concern_id == row.concern_id


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
        "integration_typed_resolution_delegate",
        "llm_provider_selection",
        "execution_bound_integration_resolution",
        "existing_capability_configuration_realization",
        "plugin_integration_catalog",
    }
    assert mandatory <= set(CONFIG_X_OWNER_EXPECTATIONS)
    for concern_key in mandatory:
        discovered, expected = compare_owner_gate(concern_key)
        assert discovered == expected


def test_frz_cfg_05_semantic_production_selection_closed_world_zero() -> None:
    assert frz_cfg_05_semantic_i_blocker_count() == 0
    assert discover_semantic_i_blocker_paths() == frozenset()
    assert frz_cfg_05_unclassified_provider_surface_count() == 0
    assert discover_unclassified_provider_surface_paths() == frozenset()
    assert frz_cfg_05_named_constant_blind_spot_count() == 0
    assert discover_named_constant_semantic_blind_spot_paths() == frozenset()
    assert _FRZ_CFG_PASS_CANDIDATE


def test_frz_cfg_06_activation_families_no_bypasses() -> None:
    assert discover_activation_bypass_findings() == ()
    assert _FRZ_CFG_PASS_CANDIDATE


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


def test_frz_cfg_07_explicit_slug_precedence_over_profile_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.integrations.contracts.base import IntegrationEntry
    from intergrax.integrations.registry.catalog import clear_catalog, register_integration

    clear_catalog()

    def _factory(**_kwargs: object) -> object:
        return object()

    register_integration(
        IntegrationEntry(
            slug="config_x_sqlite",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=_factory,
        ),
    )
    register_integration(
        IntegrationEntry(
            slug="config_x_postgresql",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=_factory,
        ),
    )
    monkeypatch.setenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", "config_x_postgresql")
    profile = IntegrationProfile(relational_store="config_x_sqlite")
    assert (
        resolve_slug(
            IntegrationCategory.RELATIONAL_STORE,
            slug="config_x_sqlite",
            profile=profile,
        )
        == "config_x_sqlite"
    )
    assert (
        resolve_slug(IntegrationCategory.RELATIONAL_STORE, profile=profile)
        == "config_x_sqlite"
    )


def test_frz_cfg_07_ambient_env_cannot_override_explicit_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.integrations.contracts.base import IntegrationEntry
    from intergrax.integrations.registry.catalog import clear_catalog, register_integration

    clear_catalog()

    def _factory(**_kwargs: object) -> object:
        return object()

    register_integration(
        IntegrationEntry(
            slug="config_x_profile_sqlite",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=_factory,
        ),
    )
    register_integration(
        IntegrationEntry(
            slug="config_x_env_postgresql",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=_factory,
        ),
    )
    monkeypatch.setenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", "config_x_env_postgresql")
    profile = IntegrationProfile(relational_store="config_x_profile_sqlite")
    assert (
        resolve_slug(IntegrationCategory.RELATIONAL_STORE, profile=profile)
        == "config_x_profile_sqlite"
    )


def test_frz_cfg_08_regression_protects_current_closed_world_classification() -> None:
    assert discover_active_blocker_path_keys() == frozenset()
    assert discover_duplicate_configuration_authority_paths() == frozenset()
    unclassified = [
        item
        for item in sweep_concern_classification_evidence()
        if item.mechanical_classification == ConfigClassification.L_UNCLEAR
    ]
    assert unclassified == []
    assert len(CONFIG_X_ACTIVE_BLOCKER_RECORDS) == 0


def test_config_x_tenant_isolation_audit_runs_real_local_gates() -> None:
    from tests.qualification.config_x import test_config_x_adversarial_cx_gates as cx_adv
    from tests.qualification.config_x import test_config_x_r1_remediation_gates as cx_r1

    cx_r1.test_t1_harness_async_run_requires_resolved_tenant_parameter()
    cx_r1.test_t2_trace_explorer_missing_tenant_rejected()
    cx_r1.test_t4_image_smart_loader_requires_explicit_tenant()
    cx_r1.test_t5_vector_integration_config_missing_tenant_fails_closed()
    cx_adv.test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards()
