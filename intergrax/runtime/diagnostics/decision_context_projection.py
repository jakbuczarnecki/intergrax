# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic Decision context projection from correlation evidence (DIAG R4)."""

from __future__ import annotations

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationRecord,
)
from intergrax.runtime.diagnostics.decision_context_read_models import (
    DecisionContextFact,
    DecisionContextReadStatus,
    DecisionContextUnavailableReason,
    DecisionContextView,
    RelatedDecisionContextEntry,
)

_CONTEXT_ONLY_LIMITATION = (
    "Related decisions are contextual evidence only; they are not causal attribution."
)


def project_decision_context_view(
    records: tuple[DecisionExecutionCorrelationRecord, ...],
    *,
    contextual_facts_by_decision_id: dict[str, tuple[DecisionContextFact, ...]]
    | None = None,
    decision_versions_by_id: dict[str, int] | None = None,
) -> DecisionContextView:
    """Build operator read model — no inference beyond supplied evidence."""
    if contextual_facts_by_decision_id is None:
        contextual_facts_by_decision_id = {}
    if decision_versions_by_id is None:
        decision_versions_by_id = {}

    if not records:
        return DecisionContextView.unavailable(
            DecisionContextUnavailableReason.NO_CORRELATION_EVIDENCE,
            limitations=(_CONTEXT_ONLY_LIMITATION,),
        )

    entries: list[RelatedDecisionContextEntry] = []
    for record in records:
        decision_key = str(record.decision_id)
        version = decision_versions_by_id.get(decision_key, 1)
        facts = contextual_facts_by_decision_id.get(decision_key, ())
        entries.append(
            RelatedDecisionContextEntry(
                decision_id=record.decision_id,
                decision_version=version,
                decision_attempt_id=record.decision_attempt_id,
                execution_id=record.execution_id,
                correlation_kind=record.correlation_kind,
                contextual_facts=facts,
            ),
        )

    return DecisionContextView(
        read_status=DecisionContextReadStatus.AVAILABLE,
        related_decisions=tuple(entries),
        unavailable_reason=None,
        limitations=(_CONTEXT_ONLY_LIMITATION,),
    )
