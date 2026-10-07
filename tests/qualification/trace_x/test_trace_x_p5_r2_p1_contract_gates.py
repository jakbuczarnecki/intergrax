# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P1 mechanical contract qualification gates."""

from __future__ import annotations

import importlib
import inspect
import subprocess

import pytest

from tests.qualification.trace_x._trace_x_p5_r2_p1_support import (
    TRACE_X_P5_R2_P1_R1_START_HEAD,
    TRACE_X_P5_R2_P1_START_HEAD,
    a0_low_mapping_violations,
    category_value_fallback_violations,
    configured_binding_duplication_violations,
    forbidden_any_in_p1_modules,
    neutral_provenance_import_violations,
    production_caller_migration_violations,
    reader_surface_violations,
    semantic_enum_runtime_isinstance_violations,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REQUIRED_SYMBOLS = (
    (
        "intergrax.integrations.contracts.existing_capability_configuration_opportunity",
        (
            "ConfigurationOpportunityRef",
            "ExistingCapabilityConfigurationOpportunityFacts",
            "ExistingCapabilityConfigurationOpportunity",
            "ExistingCapabilityConfigurationOpportunityReadPort",
            "ExistingCapabilityConfigurationOpportunityProvider",
            "ExistingCapabilityConfigurationMutationRiskPolicy",
        ),
    ),
    (
        "intergrax.integrations.contracts.execution_integration_configuration",
        (
            "EffectiveIntegrationIdentity",
            "IntegrationMaterializationKind",
            "ExecutionIntegrationConfigurationAdoption",
            "validate_configured_adoption_match",
        ),
    ),
    (
        "intergrax.contracts.execution_integration_configuration_provenance",
        (
            "IntegrationConfigurationSubject",
            "ConfiguredIntegrationProvenanceSlice",
            "ExecutionIntegrationConfigurationProvenanceMode",
            "ExecutionIntegrationConfigurationProvenance",
            "ExecutionIntegrationConfigurationProvenanceReadStatus",
            "ExecutionIntegrationConfigurationProvenanceReader",
        ),
    ),
)


def test_txp5r2p1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P1_START_HEAD, "HEAD"],
    )


def test_txp5r2p1_r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P1_R1_START_HEAD, "HEAD"],
    )


def test_txp5r2p1_r1_q02_semantic_enum_runtime_isinstance_gates() -> None:
    violations = semantic_enum_runtime_isinstance_violations()
    assert not violations, violations


def test_txp5r2p1_q02_required_contract_symbols_exist() -> None:
    for module_name, symbols in _REQUIRED_SYMBOLS:
        module = importlib.import_module(module_name)
        for symbol in symbols:
            assert hasattr(module, symbol), f"missing {module_name}.{symbol}"


def test_txp5r2p1_q03_neutral_provenance_no_runtime_imports() -> None:
    violations = neutral_provenance_import_violations()
    assert not violations, violations


def test_txp5r2p1_q04_p1_modules_no_any() -> None:
    violations = forbidden_any_in_p1_modules()
    assert not violations, violations


def test_txp5r2p1_q05_no_category_value_fallback() -> None:
    violations = category_value_fallback_violations()
    assert not violations, violations


def test_txp5r2p1_q06_no_a0_low_mapping() -> None:
    violations = a0_low_mapping_violations()
    assert not violations, violations


def test_txp5r2p1_q07_reader_no_latest_or_write() -> None:
    violations = reader_surface_violations()
    assert not violations, violations


def test_txp5r2p1_q08_configured_binding_not_duplicated() -> None:
    violations = configured_binding_duplication_violations()
    assert not violations, violations


def test_txp5r2p1_q09_no_production_caller_migration() -> None:
    violations = production_caller_migration_violations()
    assert not violations, violations


def test_txp5r2p1_q10_read_port_exact_semantics() -> None:
    from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
        ExistingCapabilityConfigurationOpportunityReadPort,
    )

    method = ExistingCapabilityConfigurationOpportunityReadPort.read_exact
    assert method.__name__ == "read_exact"


def test_txp5r2p1_q11_provider_spi_discover_facts_only() -> None:
    from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
        ExistingCapabilityConfigurationOpportunityProvider,
    )

    sig = inspect.signature(
        ExistingCapabilityConfigurationOpportunityProvider.discover_opportunity_facts
    )
    assert "tenant_id" in sig.parameters


def test_txp5r2p1_q12_configured_capability_binding_reused() -> None:
    from intergrax.integrations.contracts.execution_integration_configuration import (
        ExecutionIntegrationConfigurationAdoption,
    )

    hints = ExecutionIntegrationConfigurationAdoption.__annotations__
    assert "configured_binding" in hints
