# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed errors for human decision persistence deserialization."""

from __future__ import annotations

from pydantic import ValidationError

from intergrax.contracts.human_approver import HumanApproverEvidence

__all__ = [
    "HumanDecisionApproverProvenanceError",
    "deserialize_persisted_human_approver_evidence",
]


class HumanDecisionApproverProvenanceError(ValueError):
    """Persisted human decision row lacks or corrupts canonical approver provenance."""

    def __init__(
        self,
        message: str,
        *,
        decision_id: str,
        tenant_id: str,
    ) -> None:
        super().__init__(message)
        self.decision_id = decision_id
        self.tenant_id = tenant_id


def deserialize_persisted_human_approver_evidence(
    approver_raw: str | None,
    *,
    decision_id: str,
    tenant_id: str,
) -> HumanApproverEvidence:
    """
    Canonical read-path for persisted approver JSON.

    Missing or empty provenance is never promoted into synthetic HumanApproverEvidence.
    """
    if not approver_raw or not str(approver_raw).strip():
        raise HumanDecisionApproverProvenanceError(
            "human decision approver provenance missing in persistence row",
            decision_id=decision_id,
            tenant_id=tenant_id,
        )
    try:
        approver = HumanApproverEvidence.model_validate_json(approver_raw)
    except ValidationError as exc:
        raise HumanDecisionApproverProvenanceError(
            "human decision approver provenance invalid in persistence row",
            decision_id=decision_id,
            tenant_id=tenant_id,
        ) from exc
    if approver.tenant_id != tenant_id:
        raise HumanDecisionApproverProvenanceError(
            "human decision approver tenant does not match persistence row tenant",
            decision_id=decision_id,
            tenant_id=tenant_id,
        )
    return approver
