# © Artur Czarnecki. All rights reserved.

"""Capacity scale-up approval queue (ECP-PROD.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from intergrax.runtime.capacity.contracts import ScalingActionPlan

ApprovalStatus = Literal["pending", "approved", "denied"]


@dataclass(frozen=True, slots=True)
class CapacityApprovalRecord:
    plan_id: str
    plan: ScalingActionPlan
    status: ApprovalStatus
    requested_at: datetime


@dataclass
class CapacityApprovalQueue:
    """In-process approval queue for HITL-gated scale-up plans."""

    _records: dict[str, CapacityApprovalRecord] = field(default_factory=dict)
    _approved_plan_ids: list[str] = field(default_factory=list)

    def submit(self, plan: ScalingActionPlan) -> str:
        if plan.evaluation_status != "hitl_required":
            msg = f"plan {plan.plan_id} is not hitl_required"
            raise ValueError(msg)
        self._records[plan.plan_id] = CapacityApprovalRecord(
            plan_id=plan.plan_id,
            plan=plan,
            status="pending",
            requested_at=datetime.now(timezone.utc),
        )
        return plan.plan_id

    def approve(self, plan_id: str) -> ScalingActionPlan | None:
        record = self._records.get(plan_id)
        if record is None or record.status != "pending":
            return None
        approved = record.plan.model_copy(update={"evaluation_status": "planned"})
        self._records[plan_id] = CapacityApprovalRecord(
            plan_id=plan_id,
            plan=approved,
            status="approved",
            requested_at=record.requested_at,
        )
        self._approved_plan_ids.append(plan_id)
        return approved

    def deny(self, plan_id: str) -> bool:
        record = self._records.get(plan_id)
        if record is None or record.status != "pending":
            return False
        self._records[plan_id] = CapacityApprovalRecord(
            plan_id=plan_id,
            plan=record.plan,
            status="denied",
            requested_at=record.requested_at,
        )
        return True

    def list_pending(self) -> list[CapacityApprovalRecord]:
        return [record for record in self._records.values() if record.status == "pending"]

    def drain_approved(self) -> list[ScalingActionPlan]:
        plans: list[ScalingActionPlan] = []
        for plan_id in self._approved_plan_ids:
            record = self._records.get(plan_id)
            if record is not None and record.status == "approved":
                plans.append(record.plan)
        self._approved_plan_ids.clear()
        return plans
