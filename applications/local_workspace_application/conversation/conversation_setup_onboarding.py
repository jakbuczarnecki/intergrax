# © Artur Czarnecki. All rights reserved.

"""First-run Slack UX rendering from setup snapshot only (LKW-PRODUCT-3D)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from local_workspace_application.workspaces.workspace_setup_snapshot_service import (
    SetupNextActionV1,
    SetupPhaseV1,
    WorkspaceSetupSnapshotV1,
)

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_WORKSPACE_NAME_CHARS = 100
_MAX_QUESTION_CHARS = 500
_MAX_OPERATION_COUNTER_CHARS = 120

_WELCOME_HEADER = (
    "Welcome to LKW. I can help you ask questions grounded in your own work knowledge."
)
_WELCOME_NO_WORKSPACE_BODY = (
    "To get started, create a workspace or choose one from your list.\n"
    "You can say something like:\n"
    "• create workspace Contracts\n"
    "• show my workspaces\n"
    "• switch to workspace 2"
)
_ADD_SOURCE_BODY = (
    "Add your first knowledge by sending a supported file attachment in this conversation."
)
_SYNCING_BODY = "Your knowledge is being prepared."
_CONFIGURING_BODY = "A connected source is still being prepared."
_READY_BODY = "Your knowledge is ready. You can ask questions about it normally."
_READY_CANNOT_ASK_BODY = (
    "Your knowledge is ready, but asking is temporarily unavailable. "
    "Please try again shortly."
)
_ATTENTION_HEADER = "Something needs your attention before you can continue."
_RETRY_OR_FIX_BODY = (
    "Review the issue below and try the suggested recovery option when available."
)
_WAIT_FOR_SYNC_BODY = (
    "Preparation is still in progress. Send another message here when you want an update."
)

_ATTENTION_ERROR_MESSAGES: dict[str, str] = {
    "sync_failed": "Knowledge preparation failed. You can try syncing again.",
    "index_failed": "Indexing could not finish. You can try syncing again.",
    "runtime_unavailable": "A live source is temporarily unavailable.",
    "detach_blocked": "This source cannot be removed safely right now.",
    "configuration_invalid": "The source configuration needs to be fixed.",
}

_ATTENTION_ACTION_MESSAGES: dict[str, str] = {
    "sync": "Try syncing the source again.",
    "retry_sync": "Try syncing the source again.",
    "enable": "Try enabling the source again.",
    "disable": "You can disable the source if you need to stop using it.",
    "detach": "You can detach the source if you want to remove it from this workspace.",
    "resume_detach": "You can resume detaching the source.",
}


def _safe_text(value: object, *, limit: int = 500) -> str:
    text = _CONTROL_RE.sub(" ", str(value or ""))
    text = " ".join(text.split()).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _safe_workspace_name(value: object) -> str:
    name = _safe_text(value, limit=_MAX_WORKSPACE_NAME_CHARS)
    return name or "Workspace"


class ConversationSetupOnboardingPresenter:
    """Render product-safe first-run guidance from snapshot fields only."""

    def render_welcome(
        self,
        workspaces: Sequence[object],
    ) -> str:
        lines = [_WELCOME_HEADER, "", _WELCOME_NO_WORKSPACE_BODY]
        names = [
            _safe_workspace_name(getattr(item, "name", ""))
            for item in workspaces
            if _safe_workspace_name(getattr(item, "name", ""))
        ]
        if names:
            lines.extend(["", "Available workspaces:"])
            for index, name in enumerate(names[:25], start=1):
                lines.append(f"{index}. {name}")
        return "\n".join(lines)

    def render_snapshot_guidance(self, snapshot: WorkspaceSetupSnapshotV1) -> str:
        """Map snapshot next_action and phase to conversational guidance."""
        lines: list[str] = []

        if snapshot.phase is SetupPhaseV1.NO_KNOWLEDGE:
            lines.append(_ADD_SOURCE_BODY)
        elif snapshot.phase is SetupPhaseV1.SYNCING:
            lines.append(_SYNCING_BODY)
            lines.extend(_operation_counter_lines(snapshot))
        elif snapshot.phase is SetupPhaseV1.CONFIGURING:
            lines.append(_CONFIGURING_BODY)
        elif snapshot.phase is SetupPhaseV1.ATTENTION_REQUIRED:
            lines.extend(self._attention_lines(snapshot))
        elif snapshot.phase is SetupPhaseV1.READY:
            if snapshot.can_ask:
                lines.append(_READY_BODY)
                question = _safe_text(snapshot.suggested_question, limit=_MAX_QUESTION_CHARS)
                if question:
                    lines.append(f"Try asking: {question}")
            else:
                lines.append(_READY_CANNOT_ASK_BODY)

        next_action = snapshot.next_action
        if next_action is SetupNextActionV1.ADD_SOURCE and snapshot.phase is not SetupPhaseV1.NO_KNOWLEDGE:
            lines.append(_ADD_SOURCE_BODY)
        elif next_action is SetupNextActionV1.WAIT_FOR_SYNC:
            lines.append(_WAIT_FOR_SYNC_BODY)
        elif next_action is SetupNextActionV1.RETRY_OR_FIX_SOURCE:
            if snapshot.phase is not SetupPhaseV1.ATTENTION_REQUIRED:
                lines.append(_RETRY_OR_FIX_BODY)
        elif next_action is SetupNextActionV1.ASK_QUESTION:
            if snapshot.can_ask:
                question = _safe_text(snapshot.suggested_question, limit=_MAX_QUESTION_CHARS)
                if question and snapshot.phase is not SetupPhaseV1.READY:
                    lines.append(f"You can try asking: {question}")

        return "\n".join(line for line in lines if line).strip()

    def should_gate_question(self, snapshot: WorkspaceSetupSnapshotV1) -> bool:
        """True when a plain question should not reach Ask yet."""
        if snapshot.next_action is SetupNextActionV1.ASK_QUESTION and snapshot.can_ask:
            return False
        return True

    def should_append_snapshot_guidance(self, snapshot: WorkspaceSetupSnapshotV1) -> bool:
        """READY workspaces should not repeat first-run guidance on daily turns."""
        return snapshot.phase is not SetupPhaseV1.READY

    def render_ask_blocked(self, snapshot: WorkspaceSetupSnapshotV1) -> str:
        guidance = self.render_snapshot_guidance(snapshot)
        header = "I cannot answer that yet."
        if not guidance:
            return header
        return f"{header}\n\n{guidance}"

    def _attention_lines(self, snapshot: WorkspaceSetupSnapshotV1) -> list[str]:
        lines = [_ATTENTION_HEADER]
        attention = snapshot.attention
        if attention is None:
            lines.append(_RETRY_OR_FIX_BODY)
            return lines

        error_code = _safe_text(attention.error_code, limit=80).casefold()
        if error_code:
            message = _ATTENTION_ERROR_MESSAGES.get(
                error_code,
                "A connected source needs attention before you can continue.",
            )
            lines.append(message)
        else:
            lines.append("A connected source needs attention before you can continue.")

        action_lines = []
        for action in attention.available_actions:
            key = _safe_text(action, limit=40).casefold()
            if not key:
                continue
            text = _ATTENTION_ACTION_MESSAGES.get(key)
            if text and text not in action_lines:
                action_lines.append(text)
        if action_lines:
            lines.append("Available actions:")
            for item in action_lines[:5]:
                lines.append(f"• {item}")
        return lines


def _operation_counter_lines(snapshot: WorkspaceSetupSnapshotV1) -> list[str]:
    summary = snapshot.knowledge_summary
    lines: list[str] = []
    if summary.usable > 0 or summary.total > 0:
        lines.append(
            _safe_text(
                f"Prepared items: {summary.usable} usable of {summary.total} total.",
                limit=_MAX_OPERATION_COUNTER_CHARS,
            )
        )
    operation = snapshot.recent_operation
    if operation is not None:
        status = _safe_text(operation.status, limit=40)
        operation_type = _safe_text(operation.operation_type, limit=40)
        if status and operation_type:
            lines.append(
                _safe_text(
                    f"Latest operation: {operation_type} ({status}).",
                    limit=_MAX_OPERATION_COUNTER_CHARS,
                )
            )
    return lines


__all__ = [
    "ConversationSetupOnboardingPresenter",
]
