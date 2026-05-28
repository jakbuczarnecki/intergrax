# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HumanRequest v2 helpers and deadline registration (Phase G.4, §42.10.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from intergrax.contracts.agent_decision import AgentDecisionType, HumanRequest
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.task.task import Task
from intergrax.utils.time_provider import SystemTimeProvider


def compute_expires_at_utc(
    *,
    created_at: datetime,
    timeout_seconds: Optional[int],
) -> Optional[str]:
    if timeout_seconds is None:
        return None
    return (created_at + timedelta(seconds=timeout_seconds)).isoformat()


def human_request_event_payload(
    human_request: HumanRequest,
    *,
    created_at_utc: Optional[str] = None,
    expires_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize HumanRequest for RuntimeEvent / notification payloads."""
    payload = human_request.model_dump(mode="json")
    if created_at_utc is not None:
        payload["created_at_utc"] = created_at_utc
    if expires_at_utc is not None:
        payload["expires_at_utc"] = expires_at_utc
    elif created_at_utc is not None and human_request.timeout_seconds is not None:
        created = datetime.fromisoformat(created_at_utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        payload["expires_at_utc"] = compute_expires_at_utc(
            created_at=created,
            timeout_seconds=human_request.timeout_seconds,
        )
    return payload


def human_request_notification_extra(task: Task) -> Dict[str, Any]:
    gov = task.runtime.governance
    human_request = gov.human_request
    if human_request is None:
        return {}
    extra: Dict[str, Any] = {
        "human_request_id": human_request.request_id,
        "urgency": human_request.urgency.value,
        "timeout_seconds": human_request.timeout_seconds,
        "default_on_timeout": (
            human_request.default_on_timeout.value
            if human_request.default_on_timeout is not None
            else None
        ),
    }
    if gov.human_request_created_at:
        extra["human_request_created_at"] = gov.human_request_created_at
    if gov.human_request_expires_at:
        extra["expires_at_utc"] = gov.human_request_expires_at
    return extra


class HumanTimeoutCoordinator:
    """
    Registers human-request deadlines on tasks.

    Timeout enforcement is deferred to Phase J.4 (scheduler); G.4 only records
    deadline metadata for notifications, checkpoints and future auto-resume.
    """

    @staticmethod
    def attach_to_task(task: Task, human_request: HumanRequest) -> None:
        created_at = SystemTimeProvider.utc_now()
        gov = task.runtime.governance
        gov.human_request = human_request
        gov.human_request_created_at = created_at.isoformat()
        gov.human_request_expires_at = compute_expires_at_utc(
            created_at=created_at,
            timeout_seconds=human_request.timeout_seconds,
        )
        task.sync_metadata()

    @staticmethod
    def attach_from_execution(task: Task, execution: AgentExecutionResult) -> None:
        if execution.human_request is not None:
            HumanTimeoutCoordinator.attach_to_task(task, execution.human_request)

    @staticmethod
    def is_expired(task: Task, *, now: Optional[datetime] = None) -> bool:
        expires_raw = task.runtime.governance.human_request_expires_at
        if not expires_raw:
            return False
        current = now or SystemTimeProvider.utc_now()
        expires_at = datetime.fromisoformat(expires_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return current >= expires_at

    @staticmethod
    def planned_timeout_action(task: Task) -> Optional[AgentDecisionType]:
        human_request = task.runtime.governance.human_request
        if human_request is None:
            return None
        return human_request.default_on_timeout
