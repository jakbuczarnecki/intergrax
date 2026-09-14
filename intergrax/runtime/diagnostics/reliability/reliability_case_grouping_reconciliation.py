# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Reconciliation keys for ERL reliability-case grouping (ERL-DIAG-001C)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.diagnostics.deterministic_problem_reconciliation import (
    ProblemReconciliationKeyKind,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingBasisKind,
    ProblemGroupingCandidate,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyVersion,
)


@dataclass(frozen=True, slots=True)
class ReliabilityCaseProblemGroupingBasis:
    """Grouping evidence: one operator Problem per tenant + plugin grouping subject token."""

    grouping_subject_index_token: str

    @property
    def kind(self) -> ProblemGroupingBasisKind:
        return ProblemGroupingBasisKind.RELIABILITY_CASE


@dataclass(frozen=True, slots=True)
class ReliabilityCaseProblemReconciliationKey:
    """Recurrence evidence for default ERL case grouping — not ``ProblemId``."""

    tenant_id: str
    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    grouping_subject_index_token: str

    @property
    def kind(self) -> ProblemReconciliationKeyKind:
        return ProblemReconciliationKeyKind.RELIABILITY_CASE

    def index_token(self) -> str:
        return "|".join(
            (
                self.kind.value,
                self.tenant_id,
                str(self.strategy_id),
                str(self.strategy_version),
                self.grouping_subject_index_token,
            ),
        )


def extract_reliability_case_reconciliation_key(
    candidate: ProblemGroupingCandidate,
    *,
    tenant_id: str,
) -> ReliabilityCaseProblemReconciliationKey:
    basis = candidate.provenance.basis
    if type(basis) is not ReliabilityCaseProblemGroupingBasis:
        raise TypeError(
            "reliability case reconciliation requires ReliabilityCaseProblemGroupingBasis",
        )
    for member in candidate.members:
        if member.tenant_id != tenant_id:
            raise ValueError("candidate member tenant_id does not match tenant scope")
    return ReliabilityCaseProblemReconciliationKey(
        tenant_id=tenant_id,
        strategy_id=candidate.provenance.strategy_id,
        strategy_version=candidate.provenance.strategy_version,
        grouping_subject_index_token=basis.grouping_subject_index_token,
    )


class ReliabilityCaseProblemReconciliationPolicy:
    """Reconciliation policy for ``ReliabilityCaseDefaultGroupingStrategy``."""

    @property
    def supported_basis_kind(self) -> ProblemGroupingBasisKind:
        return ProblemGroupingBasisKind.RELIABILITY_CASE

    def extract_reconciliation_key(
        self,
        candidate: ProblemGroupingCandidate,
        *,
        tenant_id: str,
    ) -> ReliabilityCaseProblemReconciliationKey:
        return extract_reliability_case_reconciliation_key(candidate, tenant_id=tenant_id)


__all__ = [
    "ReliabilityCaseProblemGroupingBasis",
    "ReliabilityCaseProblemReconciliationKey",
    "ReliabilityCaseProblemReconciliationPolicy",
    "extract_reliability_case_reconciliation_key",
]
