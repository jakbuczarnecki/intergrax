# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.human.models import EscalationTarget, HumanResponseVerdict
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.tools.providers.hitl.contracts import (
    HitlGetDecisionInput,
    HitlListPendingInput,
    HitlSummarizeQueueInput,
)
from intergrax.tools.providers.hitl.service import hitl_get_decision, hitl_list_pending, hitl_summarize_queue
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def hitl_ctx(tmp_path: Path) -> ToolWiringContext:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    record = SQLiteHumanDecisionStore.build_record(
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
