# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TL8 redrive authority — effect-class eligibility (G-A02 / OD-R4-G-A02)."""

from __future__ import annotations

from enum import StrEnum


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
