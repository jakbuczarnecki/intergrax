# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanDecisionRecord,
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.tools.providers.hitl.contracts import (
    HitlDecisionOutput,
    HitlGetDecisionInput,
    HitlGetDecisionOutput,
    HitlListForTaskInput,
    HitlListForTaskOutput,
    HitlListPendingInput,
    HitlListPendingOutput,
    HitlSubmitResponseInput,
    HitlSubmitResponseOutput,
    HitlSummarizeQueueInput,
    HitlSummarizeQueueOutput,
)
from intergrax.tools.registry.runtime_bindings import HumanDecisionStoreBinding
from intergrax.tools.registry.wiring import ToolWiringContext

HITL_LIST_PENDING_TOOL_ID = "hitl.list_pending"
HITL_GET_DECISION_TOOL_ID = "hitl.get_decision"
HITL_SUMMARIZE_QUEUE_TOOL_ID = "hitl.summarize_queue"
HITL_SUBMIT_RESPONSE_TOOL_ID = "hitl.submit_response"
HITL_LIST_FOR_TASK_TOOL_ID = "hitl.list_for_task"


def _require_store(ctx: ToolWiringContext) -> HumanDecisionStoreBinding:
    store = ctx.human_decision_store
    if store is None:
        raise RuntimeError("human_decision_store_not_configured")
    if not isinstance(store, HumanDecisionStoreBinding):
        raise RuntimeError("human_decision_store_invalid_type")
    return store


def _decision_output(record: HumanDecisionRecord) -> HitlDecisionOutput:
    return HitlDecisionOutput(
        decision_id=record.decision_id,
        task_id=record.task_id,
        tenant_id=record.tenant_id,
        user_id=record.user_id,
        human_request_id=record.human_request_id,
        verdict=record.verdict.value,
        response_text=record.response_text,
        escalation_level=record.escalation_level,
        escalation_target=record.escalation_target.value if record.escalation_target else "",
        agent_id=record.agent_id or "",
        run_id=record.run_id or "",
        notes=record.notes,
        created_at_utc=record.created_at_utc,
    )


def hitl_list_pending(ctx: ToolWiringContext, params: HitlListPendingInput) -> HitlListPendingOutput:
    store = _require_store(ctx)
    records = [
        item
        for item in store.list_escalations(params.tenant_id.strip(), limit=params.limit)
        if isinstance(item, HumanDecisionRecord)
    ]
    decisions = [_decision_output(item) for item in records]
    return HitlListPendingOutput(
        used=True,
        decisions=decisions,
        total=len(decisions),
        reason="ok",
    )


def hitl_get_decision(ctx: ToolWiringContext, params: HitlGetDecisionInput) -> HitlGetDecisionOutput:
    store = _require_store(ctx)
    record = store.get_decision(params.decision_id.strip(), params.tenant_id.strip())
    if record is None or not isinstance(record, HumanDecisionRecord):
        return HitlGetDecisionOutput(
            used=True,
            found=False,
            reason="decision_not_found",
        )
    return HitlGetDecisionOutput(
        used=True,
        found=True,
        decision=_decision_output(record),
        reason="ok",
    )


def hitl_summarize_queue(ctx: ToolWiringContext, params: HitlSummarizeQueueInput) -> HitlSummarizeQueueOutput:
    store = _require_store(ctx)
    counts = dict(store.summarize_queue(params.tenant_id.strip()))
    pending = int(counts.get(HumanResponseVerdict.ESCALATE.value, 0))
    return HitlSummarizeQueueOutput(
        used=True,
        counts_by_verdict=counts,
        pending_escalations=pending,
        reason="ok",
    )


def _parse_verdict(raw: str) -> HumanResponseVerdict:
    try:
        return HumanResponseVerdict(raw.strip().lower())
    except ValueError as exc:
        raise ValueError(f"invalid_verdict:{raw}") from exc


def _parse_escalation_target(raw: str) -> EscalationTarget | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return EscalationTarget(value)
    except ValueError as exc:
        raise ValueError(f"invalid_escalation_target:{raw}") from exc


def hitl_submit_response(ctx: ToolWiringContext, params: HitlSubmitResponseInput) -> HitlSubmitResponseOutput:
    store = _require_store(ctx)
    try:
        verdict = _parse_verdict(params.verdict)
        escalation_target = _parse_escalation_target(params.escalation_target)
    except ValueError as exc:
        return HitlSubmitResponseOutput(used=True, recorded=False, reason=str(exc))

    record = build_human_decision_record(
        task_id=params.task_id.strip(),
        tenant_id=params.tenant_id.strip(),
        user_id=params.user_id.strip(),
        verdict=verdict,
        response_text=params.response_text,
        human_request_id=params.human_request_id.strip(),
        escalation_level=params.escalation_level,
        escalation_target=escalation_target,
        agent_id=params.agent_id.strip() or None,
        run_id=params.run_id.strip() or None,
        notes=params.notes,
    )
    stored = store.record(record)
    if not isinstance(stored, HumanDecisionRecord):
        return HitlSubmitResponseOutput(used=True, recorded=False, reason="record_failed")
    return HitlSubmitResponseOutput(
        used=True,
        recorded=True,
        decision=_decision_output(stored),
        reason="ok",
    )


def hitl_list_for_task(ctx: ToolWiringContext, params: HitlListForTaskInput) -> HitlListForTaskOutput:
    store = _require_store(ctx)
    records = [
        item
        for item in store.list_for_task(params.task_id.strip(), params.tenant_id.strip())
        if isinstance(item, HumanDecisionRecord)
    ]
    decisions = [_decision_output(item) for item in records]
    return HitlListForTaskOutput(
        used=True,
        decisions=decisions,
        total=len(decisions),
        reason="ok",
    )
