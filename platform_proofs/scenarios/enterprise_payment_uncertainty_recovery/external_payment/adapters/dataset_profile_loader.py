"""Load variant execution profiles from the canonical scenario dataset."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.profiles import (
    VariantExecutionProfile,
    build_execution_profile,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    resolve_sor_truth_fields,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    load_scenario_package,
)

_SCENARIO_DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"


def load_variant_execution_profile(variant_id: str) -> VariantExecutionProfile:
    package = load_scenario_package(
        dataset_package_root=_SCENARIO_DATASET_ROOT,
        qualification_id="ERL-QUAL-004",
        scenario_slug="enterprise_payment_uncertainty_recovery",
        variant_id=variant_id,
    )
    external_effect = package.shared["external_effect"]
    communication_event = package.shared["communication_event"]
    sor_truth = resolve_sor_truth_fields(package.variant_document)

    return build_execution_profile(
        variant_id=variant_id,
        variant_document=package.variant_document,
        external_effect=external_effect,
        communication_event=communication_event,
        sor_truth=sor_truth,
    )
