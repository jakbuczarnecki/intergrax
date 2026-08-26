# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic strategy reconciliation key extraction (DIAG-5D)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemGroupingBasis,
    DeterministicProblemSignature,
    ProblemGroupingBasisKind,
    ProblemGroupingCandidate,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyVersion,
)


class ProblemReconciliationKeyKind(StrEnum):
    """Typed discriminator for strategy-specific recurrence evidence."""

    DETERMINISTIC = "deterministic"


@dataclass(frozen=True, slots=True)
class DeterministicProblemReconciliationKey:
    """
    Conservative recurrence evidence for deterministic structural grouping.

    NOT ``ProblemId`` — only auditable evidence used to find the same tracked
    Problem across grouping invocations.
    """

    tenant_id: str
    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    signature: DeterministicProblemSignature

    @property
    def kind(self) -> ProblemReconciliationKeyKind:
        return ProblemReconciliationKeyKind.DETERMINISTIC

    def index_token(self) -> str:
        return "|".join(
            (
                self.kind.value,
                self.tenant_id,
                str(self.strategy_id),
                str(self.strategy_version),
                repr(self.signature),
            )
        )


def extract_deterministic_reconciliation_key(
    candidate: ProblemGroupingCandidate,
    *,
    tenant_id: str,
) -> DeterministicProblemReconciliationKey:
    """Derive the deterministic recurrence key from a validated candidate."""
    basis = candidate.provenance.basis
    if type(basis) is not DeterministicProblemGroupingBasis:
        raise TypeError(
            "deterministic reconciliation requires DeterministicProblemGroupingBasis"
        )

    for member in candidate.members:
        if member.tenant_id != tenant_id:
            raise ValueError("candidate member tenant_id does not match tenant scope")

    return DeterministicProblemReconciliationKey(
        tenant_id=tenant_id,
        strategy_id=candidate.provenance.strategy_id,
        strategy_version=candidate.provenance.strategy_version,
        signature=basis.signature,
    )


def reconciliation_key_basis_kind(
    key: DeterministicProblemReconciliationKey,
) -> ProblemGroupingBasisKind:
    return ProblemGroupingBasisKind.DETERMINISTIC
