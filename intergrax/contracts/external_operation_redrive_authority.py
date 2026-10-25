# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TL8 redrive authority — effect-class eligibility (G-A02 / OD-R4-G-A02)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    contract_declares_idempotency,
    contract_declares_reconciliation,
    validate_external_effect_contract,
)


class ExternalOperationRedriveEffectClass(StrEnum):
    """Gate A recovery eligibility; not a parallel policy engine."""

    CLASS_A_PROVIDER_ATOMIC_FENCE = "CLASS_A_PROVIDER_ATOMIC_FENCE"
    CLASS_B_IDEMPOTENT_OR_RECONCILED = "CLASS_B_IDEMPOTENT_OR_RECONCILED"
    CLASS_C_FAIL_CLOSED = "CLASS_C_FAIL_CLOSED"


class ExternalOperationRedriveAuthorityError(RuntimeError):
    """Redrive authority transfer cannot proceed."""


class ExternalOperationRedriveEffectClassRejectedError(
    ExternalOperationRedriveAuthorityError,
):
    """Effect class is not eligible for automated positive redrive."""


class ExternalOperationRedriveAdmissionRejectedError(
    ExternalOperationRedriveAuthorityError,
):
    """Reconciliation or certified-loss prerequisites not satisfied."""


class ExternalOperationRedriveCompetingAuthorityError(
    ExternalOperationRedriveAuthorityError,
):
    """Another recovery worker already obtained redrive authority."""


def resolve_authoritative_redrive_effect_class(
    effect_contract: ExternalEffectContract,
) -> ExternalOperationRedriveEffectClass:
    """Derive Gate A recovery class from a trusted ``ExternalEffectContract`` only."""
    validate_external_effect_contract(effect_contract)
    if contract_declares_idempotency(effect_contract) or contract_declares_reconciliation(
        effect_contract,
    ):
        return ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED
    return ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED


def assert_redrive_effect_class_claim_matches_authoritative(
    *,
    caller_claimed_effect_class: ExternalOperationRedriveEffectClass,
    authoritative_effect_class: ExternalOperationRedriveEffectClass,
) -> None:
    if caller_claimed_effect_class is not authoritative_effect_class:
        raise ExternalOperationRedriveEffectClassRejectedError(
            "caller effect class claim does not match authoritative trusted contract",
        )
