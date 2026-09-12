# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UNKNOWN admission coordinated with external effect contract declarations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    UnknownUncertaintyPosture,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyStateRecord
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import (
    admit_external_effect_unknown,
)


class ExternalEffectUnknownAdmission(BaseModel):
    """UNKNOWN episode opened with contract-derived posture (no reconcile execution)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    unknown_posture: UnknownUncertaintyPosture


def admit_external_effect_unknown_with_contract(
    *,
    correlation_id: str,
    contract: ExternalEffectContract,
) -> ExternalEffectUnknownAdmission:
    """Admit UNKNOWN and attach declared safety posture for downstream ERL phases."""
    return ExternalEffectUnknownAdmission(
        state=admit_external_effect_unknown(correlation_id=correlation_id),
        contract_id=contract.contract_id,
        unknown_posture=evaluate_unknown_uncertainty_posture(contract),
    )


__all__ = [
    "ExternalEffectUnknownAdmission",
    "admit_external_effect_unknown_with_contract",
]
