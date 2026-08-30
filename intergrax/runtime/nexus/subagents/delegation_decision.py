# © Artur Czarnecki. All rights reserved.

"""Delegation rationale in DecisionRecord (IDEAL-10.5)."""

from __future__ import annotations

from intergrax.contracts.uaep_decision_record import DecisionRecord
from intergrax.contracts.delegation import DelegationSpec


def decision_record_for_delegation(
    spec: DelegationSpec,
    *,
    trace_id: str = "",
    run_id: str = "",
    tenant_id: str = "",
    task_id: str = "",
    parent_agent_id: str = "",
) -> DecisionRecord:
    return DecisionRecord(
        trace_id=trace_id,
        run_id=run_id,
        tenant_id=tenant_id,
        task_id=task_id,
        agent_id=parent_agent_id,
        decision_type="delegation",
        rationale=spec.objective or "delegated subtask",
        delegation_target=spec.child_agent_id,
        delegation_rationale=spec.objective or "",
        delegation_scopes=tuple(spec.permission_scopes or ()),
    )
