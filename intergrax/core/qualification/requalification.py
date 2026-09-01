# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider requalification decision semantics (PROVIDER-QUAL-6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from intergrax.core.qualification.validity import (
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationValidityInterpretation,
    QualificationValidityRecord,
    ValidityEvaluationId,
    _require_aware_instant,
    new_qualification_run_id,
)
from intergrax.core.qualification.validity_evaluation import (
    establish_current_qualification_validity,
)


class ProviderRequalificationPreparationError(Exception):
    """Raised when requalification run identity cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class ProviderRequalificationDecision:
    """Derived decision: whether a new qualification execution is required."""

    qualification_run_id: QualificationRunId
    required: bool
    reason: str | None
    based_on_validity: QualificationEvidenceValidity
    basis_validity_evaluation_id: ValidityEvaluationId
    prior_run_remains_terminal: bool
    decided_at: datetime

    def __post_init__(self) -> None:
        _require_aware_instant(self.decided_at, field_name="decided_at")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("reason must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class ProviderRequalificationRunIdentity:
    """Minted identity for a subsequent qualification execution (decision != execution)."""

    prior_qualification_run_id: QualificationRunId
    new_qualification_run_id: QualificationRunId
    decision: ProviderRequalificationDecision

    def __post_init__(self) -> None:
        if self.prior_qualification_run_id == self.new_qualification_run_id:
            raise ValueError(
                "new qualification_run_id must differ from prior qualification_run_id",
            )
        if self.decision.qualification_run_id != self.prior_qualification_run_id:
            raise ValueError(
                "decision qualification_run_id must match prior_qualification_run_id",
            )


def determine_provider_requalification_requirement(
    interpretation: QualificationValidityInterpretation,
    *,
    decided_at: datetime,
) -> ProviderRequalificationDecision:
    """
    Derive whether a new qualification run is required from the latest validity view.

    CURRENT → not required. STALE or REVOKED → required (fresh evidence).
    REVOKED runs remain terminally revoked; requalification only mints new evidence.
    """
    if not isinstance(interpretation, QualificationValidityInterpretation):
        raise TypeError("interpretation must be QualificationValidityInterpretation")
    _require_aware_instant(decided_at, field_name="decided_at")

    latest = interpretation.latest_record
    validity = interpretation.validity

    if validity is QualificationEvidenceValidity.CURRENT:
        return ProviderRequalificationDecision(
            qualification_run_id=interpretation.qualification_run_id,
            required=False,
            reason=None,
            based_on_validity=validity,
            basis_validity_evaluation_id=latest.validity_evaluation_id,
            prior_run_remains_terminal=False,
            decided_at=decided_at,
        )

    required = validity in (
        QualificationEvidenceValidity.STALE,
        QualificationEvidenceValidity.REVOKED,
    )
    return ProviderRequalificationDecision(
        qualification_run_id=interpretation.qualification_run_id,
        required=required,
        reason=latest.reason,
        based_on_validity=validity,
        basis_validity_evaluation_id=latest.validity_evaluation_id,
        prior_run_remains_terminal=validity is QualificationEvidenceValidity.REVOKED,
        decided_at=decided_at,
    )


def establish_provider_requalification_requirement(
    qualification_run_id: QualificationRunId | str,
    records: Sequence[QualificationValidityRecord],
    *,
    decided_at: datetime,
) -> ProviderRequalificationDecision:
    """Fail-closed requalification decision from append-only validity history."""
    interpretation = establish_current_qualification_validity(
        qualification_run_id,
        records,
    )
    return determine_provider_requalification_requirement(
        interpretation,
        decided_at=decided_at,
    )


def prepare_provider_requalification_run_identity(
    decision: ProviderRequalificationDecision,
) -> ProviderRequalificationRunIdentity:
    """
    Mint a new qualification_run_id for a required requalification.

    Does not execute qualification or mutate the prior run. Execution belongs to
    PROVIDER-QUAL-7 shared runner when available.
    """
    if not isinstance(decision, ProviderRequalificationDecision):
        raise TypeError("decision must be ProviderRequalificationDecision")
    if not decision.required:
        raise ProviderRequalificationPreparationError(
            "requalification is not required for this qualification run",
        )

    new_run_id = new_qualification_run_id()
    if new_run_id == decision.qualification_run_id:
        raise ProviderRequalificationPreparationError(
            "minted qualification_run_id collided with prior qualification_run_id",
        )

    return ProviderRequalificationRunIdentity(
        prior_qualification_run_id=decision.qualification_run_id,
        new_qualification_run_id=new_run_id,
        decision=decision,
    )
