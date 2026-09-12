"""Unit tests for ERL-QUAL-004 PostgreSQL provisioning adapter (no live database)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.context import (
    ProvisioningContext,
    ProvisioningExecutionContext,
    ScenarioIdentity,
    ScenarioVariantSelection,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.failures import (
    ProvisioningFailureCode,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    load_scenario_package,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materialization import (
    resolve_external_reality_fields,
    row_payloads,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.provisioner import (
    PostgreSqlScenarioProvisioner,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    MissingScenarioVariantError,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_ROOT = _REPO_ROOT / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
_DATASET_ROOT = _SCENARIO_ROOT / "dataset"


def _manifest_variant_ids() -> tuple[str, ...]:
    manifest = json.loads((_DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return tuple(entry["variant_id"] for entry in manifest["variants"])


def _context(variant_id: str) -> ProvisioningContext:
    return ProvisioningContext(
        identity=ScenarioIdentity(
            qualification_id="ERL-QUAL-004",
            scenario_slug="enterprise_payment_uncertainty_recovery",
        ),
        variant=ScenarioVariantSelection(variant_id=variant_id),
        execution=ProvisioningExecutionContext(
            run_id="unit-test",
            dataset_package_root=_DATASET_ROOT,
        ),
    )


def test_dataset_loader_rejects_missing_variant() -> None:
    with pytest.raises(MissingScenarioVariantError):
        load_scenario_package(
            dataset_package_root=_DATASET_ROOT,
            qualification_id="ERL-QUAL-004",
            scenario_slug="enterprise_payment_uncertainty_recovery",
            variant_id="not-a-real-variant",
        )


@pytest.mark.parametrize("variant_id", _manifest_variant_ids())
def test_dataset_loader_loads_canonical_variants(variant_id: str) -> None:
    package = load_scenario_package(
        dataset_package_root=_DATASET_ROOT,
        qualification_id="ERL-QUAL-004",
        scenario_slug="enterprise_payment_uncertainty_recovery",
        variant_id=variant_id,
    )
    assert package.resolution.variant.variant_id == variant_id
    payloads = row_payloads(package)
    assert payloads["commerce.application_knowledge"]["known_status"] == "UNKNOWN"
    assert payloads["commerce.application_knowledge"]["uncertainty_explicit"] is True


@pytest.mark.parametrize(
    ("variant_id", "terminal_outcome"),
    (
        ("payment_completed_after_unknown", "PAYMENT_COMPLETED"),
        ("payment_failed_after_unknown", "PAYMENT_FAILED"),
        ("payment_truth_unavailable", "TRUTH_INDETERMINATE"),
    ),
)
def test_external_reality_mapping_is_data_driven(variant_id: str, terminal_outcome: str) -> None:
    package = load_scenario_package(
        dataset_package_root=_DATASET_ROOT,
        qualification_id="ERL-QUAL-004",
        scenario_slug="enterprise_payment_uncertainty_recovery",
        variant_id=variant_id,
    )
    fields = resolve_external_reality_fields(package.variant_document)
    assert fields["terminal_outcome"] == terminal_outcome
    assert row_payloads(package)["external_sor.external_reality"]["terminal_outcome"] == terminal_outcome


def test_prepare_reports_invalid_dataset_without_database() -> None:
    outcome = PostgreSqlScenarioProvisioner().prepare(_context("does-not-exist"))
    assert outcome.status is ProvisioningOutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code is ProvisioningFailureCode.MISSING_SCENARIO_VARIANT


def test_prepare_reports_database_unavailable_when_lab_not_running() -> None:
    missing_env = _SCENARIO_ROOT / "infrastructure/config/postgres.env.missing-unit"
    outcome = PostgreSqlScenarioProvisioner(env_file=missing_env).prepare(
        _context("payment_completed_after_unknown")
    )
    assert outcome.status is ProvisioningOutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code is ProvisioningFailureCode.PROVISIONING_UNAVAILABLE
