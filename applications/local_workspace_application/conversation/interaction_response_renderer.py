# © Artur Czarnecki. All rights reserved.

"""Bounded, deterministic rendering of conversational execution results."""

from __future__ import annotations

import re
from collections.abc import Mapping

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionStatus,
    ConversationInteractionExecutionResult,
)

MAX_RESPONSE_CHARS = 3_900
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_GENERIC_ERROR = "I could not complete this request. Please try again."

_ERROR_MESSAGES = {
    "conversation_context_not_found": "This conversation is not connected to an available workspace.",
    "conversation_context_not_active": "This conversation context is not active.",
    "conversation_audience_not_supported": "This conversation type is not supported for personal actions.",
    "conversation_activation_not_allowed": "This conversation is not allowed to activate the workspace.",
    "conversation_planning_failed": "I could not understand the requested workspace operation safely.",
    "conversation_plan_invalid": "I could not validate the requested workspace operation.",
    "conversation_execution_failed": "The workspace operation could not be completed.",
    "conversation_duplicate_event": "",
    "conversation_response_render_failed": _GENERIC_ERROR,
    "conversation_response_send_failed": _GENERIC_ERROR,
    "conversation_receipt_unavailable": "This request could not be safely processed. Please try again.",
    "blocked_dependency": "Not executed because a required earlier step failed.",
    "blocked_clarification": "Not executed until the clarification is answered.",
    "conversation_capability_not_allowed": "This operation is not allowed in the current conversation.",
    "workspace_not_found": "The requested workspace is no longer available.",
    "workspace_reference_ambiguous": "The workspace reference is ambiguous.",
    "active_workspace_required": "An active workspace is required for this operation.",
    "attachment_not_found": "One or more attached files are no longer available.",
    "source_candidate_not_found": "The requested source candidate is not available.",
    "source_candidate_unavailable": "Source candidates are temporarily unavailable.",
    "source_candidate_ambiguous": "The source candidate reference is ambiguous.",
    "source_reference_unsupported": "That source type is not supported.",
    "action_execution_failed": "This operation could not be completed.",
}


def _safe_text(value: object, *, limit: int = 500) -> str:
    text = _CONTROL_RE.sub(" ", str(value or ""))
    text = " ".join(text.split()).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _safe_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _error_message(code: str) -> str:
    return _ERROR_MESSAGES.get(code, _GENERIC_ERROR)


class ConversationInteractionResponseRenderer:
    """Render one safe response from one immutable execution result."""

    def render(
        self,
        result: ConversationInteractionExecutionResult,
    ) -> str:
        if result.error is not None:
            return _bounded([_error_message(result.error.code)])

        lines: list[str] = []
        completed: list[str] = []
        failures: list[str] = []
        blocked: list[str] = []

        for item in result.action_results:
            data = _mapping(item.artifact.data) if item.artifact is not None else {}
            if item.status is ConversationActionExecutionStatus.COMPLETED:
                completed.extend(self._completed_lines(item.action_type, data))
            elif item.status is ConversationActionExecutionStatus.FAILED:
                code = item.error.code if item.error is not None else "action_execution_failed"
                failures.append(f"Failed: {_error_message(code)}")
            elif item.status is ConversationActionExecutionStatus.BLOCKED_DEPENDENCY:
                blocked.append(f"Blocked: {_error_message('blocked_dependency')}")
            elif item.status is ConversationActionExecutionStatus.BLOCKED_CLARIFICATION:
                blocked.append(f"Blocked: {_error_message('blocked_clarification')}")

        clarification_lines = [
            f"Clarification needed: {_safe_text(item.question, limit=1_000)}"
            for item in result.clarifications
        ]
        lines.extend(completed)
        lines.extend(failures)
        lines.extend(blocked)
        lines.extend(clarification_lines)

        if not lines:
            return _bounded([_GENERIC_ERROR])
        if result.active_workspace_id and any(
            item.action_type == "workspace.activate"
            and item.status is ConversationActionExecutionStatus.COMPLETED
            for item in result.action_results
        ):
            # The executor supplies the safe display name in the activation artifact.
            activation = next(
                (
                    _mapping(item.artifact.data)
                    for item in result.action_results
                    if item.action_type == "workspace.activate"
                    and item.status is ConversationActionExecutionStatus.COMPLETED
                    and item.artifact is not None
                ),
                {},
            )
            name = _safe_text(activation.get("name"), limit=200)
            if name and not any(line.startswith("Active workspace:") for line in lines):
                lines.insert(0, f"Active workspace: {name}")
        return _bounded(
            lines,
            preserved=clarification_lines + failures + blocked,
        )

    @staticmethod
    def _completed_lines(
        action_type: str,
        data: Mapping[str, object],
    ) -> list[str]:
        if action_type == "workspace.create":
            name = _safe_text(data.get("name"), limit=200)
            return [f"Workspace created: {name}"] if name else ["Workspace created."]
        if action_type == "workspace.list":
            workspaces = data.get("workspaces")
            count = len(workspaces) if isinstance(workspaces, list) else 0
            return [f"Workspaces: {count}"]
        if action_type == "source.list":
            sources = data.get("sources")
            count = len(sources) if isinstance(sources, list) else 0
            return [f"Sources: {count}"]
        if action_type == "source_candidate.list":
            candidates = data.get("candidates")
            count = len(candidates) if isinstance(candidates, list) else 0
            return [f"Source candidates: {count}"]
        if action_type == "source_candidate.attach":
            label = _safe_text(data.get("label"), limit=200)
            return [f"Source attached: {label}"] if label else ["Source attached."]
        if action_type == "knowledge.add_attachments":
            attachments = data.get("attachments")
            count = len(attachments) if isinstance(attachments, list) else 0
            return [f"Attachments accepted: {count}"]
        if action_type == "knowledge.add_sources":
            sources = data.get("sources")
            count = len(sources) if isinstance(sources, list) else 0
            return [f"Sources added: {count}"]
        if action_type == "workspace.ask":
            answer = _safe_text(data.get("answer"), limit=3_000)
            status = _safe_text(data.get("status"), limit=80).casefold()
            if status == "insufficient_evidence" or not answer:
                return ["I could not find enough verified information to answer reliably."]
            lines = [f"Question answered: {answer}"]
            citations = data.get("citations")
            if isinstance(citations, list):
                labels = []
                for citation in citations:
                    citation_map = _mapping(citation)
                    label = _safe_text(
                        citation_map.get("file_name")
                        or citation_map.get("label")
                        or citation_map.get("source_name"),
                        limit=200,
                    )
                    if label and label not in labels:
                        labels.append(label)
                if labels:
                    lines.append("Citations: " + ", ".join(labels[:5]))
            return lines
        return [f"Completed: {_safe_text(action_type, limit=100)}"]


def _bounded(lines: list[str], *, preserved: list[str] | None = None) -> str:
    normalized: list[str] = []
    for line in lines:
        if line:
            safe_line = _safe_text(line, limit=MAX_RESPONSE_CHARS)
            if safe_line:
                normalized.append(safe_line)

    preserved_lines = []
    for line in preserved or []:
        if line:
            safe_line = _safe_text(line, limit=MAX_RESPONSE_CHARS)
            if safe_line and safe_line not in preserved_lines:
                preserved_lines.append(safe_line)
    preserved_values = set(preserved_lines)
    remaining_lines = [
        line for line in normalized if line not in preserved_values
    ]

    output: list[str] = []
    used = 0
    truncated = False

    for line in [*preserved_lines, *remaining_lines]:
        separator_length = 1 if output else 0
        available = MAX_RESPONSE_CHARS - used - separator_length
        if available <= 0:
            break
        if len(line) <= available:
            output.append(line)
            used += separator_length + len(line)
            continue

        if available == 1:
            output.append("…")
        else:
            output.append(line[: available - 1].rstrip() + "…")
        truncated = True
        break

    result = "\n".join(output).strip()
    if not result:
        result = _GENERIC_ERROR[:MAX_RESPONSE_CHARS] or "…"
    if truncated and len(result) > MAX_RESPONSE_CHARS:
        result = result[:MAX_RESPONSE_CHARS]
    return result[:MAX_RESPONSE_CHARS] or "…"


__all__ = [
    "ConversationInteractionResponseRenderer",
    "MAX_RESPONSE_CHARS",
]
