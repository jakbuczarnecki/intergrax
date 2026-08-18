# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanDecisionRecord,
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.tools.providers.hitl.contracts import (
    HitlGetDecisionInput,
    HitlListForTaskInput,
    HitlListPendingInput,
    HitlSubmitResponseInput,
    HitlSummarizeQueueInput,
)
from intergrax.tools.providers.hitl.service import (
    hitl_get_decision,
    hitl_list_for_task,
    hitl_list_pending,
    hitl_submit_response,
    hitl_summarize_queue,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def hitl_ctx(tmp_path: Path) -> ToolWiringContext:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    record = build_human_decision_record(
        task_id="task-1",
        tenant_id="tenant-1",
        user_id="user-1",
        verdict=HumanResponseVerdict.ESCALATE,
        response_text="needs review",
        escalation_target=EscalationTarget.HUMAN_OPERATOR,
    )
    store.record(record)
    return ToolWiringContext(human_decision_store=store)


def test_hitl_list_pending(hitl_ctx: ToolWiringContext) -> None:
    out = hitl_list_pending(hitl_ctx, HitlListPendingInput(tenant_id="tenant-1"))
    assert out.used is True
    assert out.total == 1
    assert out.decisions[0].verdict == "escalate"


def test_hitl_get_decision(hitl_ctx: ToolWiringContext) -> None:
    pending = hitl_list_pending(hitl_ctx, HitlListPendingInput(tenant_id="tenant-1"))
    decision_id = pending.decisions[0].decision_id
    out = hitl_get_decision(
        hitl_ctx,
        HitlGetDecisionInput(tenant_id="tenant-1", decision_id=decision_id),
    )
    assert out.used is True
    assert out.found is True
    assert out.decision is not None
    assert out.decision.task_id == "task-1"


def test_hitl_summarize_queue(hitl_ctx: ToolWiringContext) -> None:
    out = hitl_summarize_queue(hitl_ctx, HitlSummarizeQueueInput(tenant_id="tenant-1"))
    assert out.used is True
    assert out.pending_escalations == 1
    assert out.counts_by_verdict["escalate"] == 1


def test_hitl_not_configured() -> None:
    with pytest.raises(RuntimeError, match="human_decision_store_not_configured"):
        hitl_list_pending(ToolWiringContext(), HitlListPendingInput(tenant_id="tenant-1"))


def test_hitl_submit_response(hitl_ctx: ToolWiringContext) -> None:
    out = hitl_submit_response(
        hitl_ctx,
        HitlSubmitResponseInput(
            tenant_id="tenant-1",
            task_id="task-2",
            user_id="operator-1",
            verdict="approve",
            response_text="looks good",
        ),
    )
    assert out.used is True
    assert out.recorded is True
    assert out.decision is not None
    assert out.decision.verdict == "approve"


def test_hitl_submit_response_uses_vendor_neutral_factory() -> None:
    store = InMemoryHumanDecisionPersistence()
    ctx = ToolWiringContext(human_decision_store=store)
    out = hitl_submit_response(
        ctx,
        HitlSubmitResponseInput(
            tenant_id="tenant-1",
            task_id="task-3",
            user_id="operator-1",
            verdict="approve",
            response_text="approved without sqlite",
        ),
    )
    assert out.used is True
    assert out.recorded is True
    assert out.decision is not None
    listed = store.list_for_task("task-3", "tenant-1")
    assert len(listed) == 1
    assert isinstance(listed[0], HumanDecisionRecord)
    assert listed[0].verdict is HumanResponseVerdict.APPROVE


def test_hitl_list_for_task(hitl_ctx: ToolWiringContext) -> None:
    hitl_submit_response(
        hitl_ctx,
        HitlSubmitResponseInput(
            tenant_id="tenant-1",
            task_id="task-2",
            verdict="reject",
            response_text="no",
        ),
    )
    out = hitl_list_for_task(hitl_ctx, HitlListForTaskInput(tenant_id="tenant-1", task_id="task-2"))
    assert out.used is True
    assert out.total == 1
    assert out.decisions[0].verdict == "reject"
