"""Integration tests for ERL-QUAL-004 PostgreSQL provisioning against the lab container."""

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
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.lifecycle import (
    ProvisioningLifecycleCoordinator,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.connection import (
    postgres_lab_available,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.provisioner import (
    PostgreSqlScenarioProvisioner,
)

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DATASET_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/dataset"
)


def _manifest_variant_ids() -> tuple[str, ...]:
    manifest = json.loads((_DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return tuple(entry["variant_id"] for entry in manifest["variants"])


def _context(variant_id: str, run_id: str) -> ProvisioningContext:
    return ProvisioningContext(
        identity=ScenarioIdentity(
            qualification_id="ERL-QUAL-004",
            scenario_slug="enterprise_payment_uncertainty_recovery",
        ),
        variant=ScenarioVariantSelection(variant_id=variant_id),
        execution=ProvisioningExecutionContext(
            run_id=run_id,
            dataset_package_root=_DATASET_ROOT,
        ),
    )


@pytest.mark.skipif(not postgres_lab_available(), reason="ERL-QUAL-004 PostgreSQL lab not running")
@pytest.mark.parametrize("variant_id", _manifest_variant_ids())
def test_postgres_provisioner_lifecycle_per_variant(variant_id: str) -> None:
    coordinator = ProvisioningLifecycleCoordinator()
    result = coordinator.run(
        PostgreSqlScenarioProvisioner(),
        _context(variant_id=variant_id, run_id=f"integration-{variant_id}"),
    )
    assert result.status is ProvisioningOutcomeStatus.SUCCEEDED
    assert result.session is not None
    assert result.session.variant_id == variant_id
    assert result.state_availability is not None
    assert result.state_availability.state_ready is True
