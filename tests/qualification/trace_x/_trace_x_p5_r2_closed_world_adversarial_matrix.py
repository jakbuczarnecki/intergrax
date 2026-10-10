# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2 adversarial E2E bundle matrix (E2E-A … E2E-H)."""

from __future__ import annotations

from typing import Final

from tests.qualification.trace_x._trace_x_p5_r2_closed_world_types import AdversarialBundleRow


def _e2e(
    bundle_id: str,
    scenario: str,
    test_module: str,
    test_id: str,
) -> AdversarialBundleRow:
    return AdversarialBundleRow(
        bundle_id=bundle_id,
        scenario=scenario,
        test_module=test_module,
        test_id=test_id,
        status="PASS",
    )


P5_CLOSED_WORLD_ADVERSARIAL_MATRIX: Final[tuple[AdversarialBundleRow, ...]] = (
    _e2e(
        "E2E-A",
        "configured binding → Execution → pin → spine → I/O → reconstruct",
        "tests/unit/applications/test_uca6c_marketplace_qualified_execution_composition.py",
        "test_production_marketplace_configured_path_execute_pin_spine_io_reconstruct",
    ),
    _e2e(
        "E2E-B",
        "pin committed / ACK lost → deterministic recovery",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_pin_ambiguous_outcome_contract.py",
        "test_ambiguous_pin_lost_acknowledgement_retry_reuses_stored_staging",
    ),
    _e2e(
        "E2E-C",
        "spine persistence failure → zero business I/O",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_case_c_spine_failure_blocks_io_then_recovery_allows_io",
    ),
    _e2e(
        "E2E-D",
        "crash after spine before I/O → retry + one semantic spine event",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_case_d_idempotent_spine_then_single_io",
    ),
    _e2e(
        "E2E-E",
        "current configuration mutation → historical reconstruction unchanged",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_historical_restart_ignores_changed_current_configuration_state",
    ),
    _e2e(
        "E2E-F",
        "tenant attack → rejected",
        "tests/unit/applications/test_uca6c_marketplace_qualified_execution_composition.py",
        "test_tenant_mismatch_blocks_materialization_pin_and_io",
    ),
    _e2e(
        "E2E-G",
        "missing/corrupt evidence → fail closed",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_required_provenance_missing_fails_closed",
    ),
    _e2e(
        "E2E-H",
        "unsupported configured category/path → explicit rejection",
        "tests/unit/applications/test_uca6c_marketplace_qualified_execution_composition.py",
        "test_production_marketplace_configured_adopted_unsupported_integration_category_rejects_before_io",
    ),
)

P5_SEMANTIC_OWNER_MATRIX: Final[tuple[tuple[str, str, int], ...]] = (
    ("configuration opportunity owner", "Integrations existing-capability opportunity read (WorkerConfiguredCapabilityFulfillmentService)", 1),
    ("configured adoption owner", "ExecutionIntegrationConfigurationAdoption explicit construction at fulfillment boundary", 1),
    ("execution target owner", "Marketplace configured target + subject builder", 1),
    ("intent repository", "QualifiedMarketplaceToolExecutionIntentRepository (marketplace qualified path)", 1),
    ("Execution admission owner", "Execution-bound capability execution intake/dispatch", 1),
    ("configured provider resolver", "ExecutionBoundIntegrationResolution", 1),
    ("PinRecord owner", "ExecutionIntegrationConfigurationPinningStore (+ persistence wiring)", 1),
    ("requirement staging owner", "execution_integration_configuration_requirement_fact + pin reconciliation staging", 1),
    ("requirement emitter", "integration_configuration_provenance_requirement_recorder", 1),
    ("RuntimeEventBus", "canonical runtime event persistence/delivery (SQLite production spine)", 1),
    ("reconstructor", "ExecutionReconstructor", 1),
    ("historical provenance reader", "PinningStoreExecutionIntegrationConfigurationProvenanceReader", 1),
)
