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
    "knowledge_connection_not_found": "The requested knowledge connection is not available.",
    "knowledge_connection_not_active": "The requested knowledge connection is not active.",
    "knowledge_resource_discovery_unavailable": "Remote resources are temporarily unavailable.",
    "knowledge_resource_not_found": "The requested remote resource is not available.",
    "knowledge_plugin_configuration_unavailable": "Knowledge integrations are temporarily unavailable.",
    "citation_context_not_found": "I do not have a recent grounded answer with citations to inspect.",
    "citation_ordinal_invalid": "That citation reference is not available.",
    "citation_not_available": "That citation is no longer available.",
    "document_not_found": "That source document is no longer available.",
    "document_forbidden": "You do not have access to that source document.",
    "document_inspect_unavailable": "Source inspection is temporarily unavailable.",
    "attachment_too_large": "One or more attachments are too large.",
    "attachment_unsupported": "One or more attachments are not supported.",
    "intake_rejected": "The knowledge intake request could not be accepted.",
    "ingestion_failed": "Knowledge preparation failed. Please try again.",
    "ask_unavailable": "Asking questions is temporarily unavailable.",
    "host_unavailable": "The workspace service is temporarily unavailable.",
    "insufficient_evidence": "I could not find enough verified information to answer reliably.",
    "destructive_confirmation_invalid": "That confirmation is not valid. Please request the action again.",
    "destructive_confirmation_expired": "That confirmation has expired. Please request the action again.",
    "destructive_confirmation_stale": "The target changed before confirmation. Please request the action again.",
    "destructive_confirmation_required": "Confirmation is required before this action can run.",
    "knowledge_inventory_unavailable": "Knowledge inventory is temporarily unavailable.",
    "knowledge_target_not_found": "The requested knowledge source was not found.",
    "knowledge_target_ambiguous": "The knowledge source reference is ambiguous.",
    "knowledge_operation_not_available": "That operation is not available for this source right now.",
    "knowledge_operation_conflict": "The source changed before the operation could finish. Please try again.",
}


def _safe_text(value: object, *, limit: int = 500) -> str:
    text = _CONTROL_RE.sub(" ", str(value or ""))
    text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split()).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _safe_limit(value: object) -> str:
    if value is None:
        return "not declared"
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    return "not declared"


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
            if not isinstance(workspaces, list) or not workspaces:
                return ["No workspaces found."]
            lines: list[str] = []
            for index, item in enumerate(workspaces, start=1):
                workspace = _mapping(item)
                name = _safe_text(workspace.get("name"), limit=200) or "Workspace"
                active_marker = " (active)" if bool(workspace.get("is_active")) else ""
                lines.append(f"{index}. {name}{active_marker}")
            return lines
        if action_type == "source.list":
            sources = data.get("sources")
            count = len(sources) if isinstance(sources, list) else 0
            return [f"Sources: {count}"]
        if action_type == "knowledge.connections.list":
            connections = data.get("connections")
            if not isinstance(connections, list):
                return ["Configured connections: 0"]
            lines = [f"Configured connections: {len(connections)}"]
            for index, item in enumerate(connections, start=1):
                connection = _mapping(item)
                label = _safe_text(connection.get("safe_display_label"), limit=120)
                status = _safe_text(connection.get("administrative_status"), limit=40)
                modes = connection.get("available_configuration_modes")
                mode_text = ", ".join(
                    _safe_text(mode, limit=60)
                    for mode in modes
                ) if isinstance(modes, list) else "UNKNOWN"
                lines.append(f"{index}. {label} [{status}] — {mode_text}")
            return lines
        if action_type == "knowledge.resources.list":
            resources = data.get("resources")
            if not isinstance(resources, list):
                return ["Remote resources: 0"]
            lines = [f"Remote resources: {len(resources)}"]
            for index, item in enumerate(resources, start=1):
                resource = _mapping(item)
                label = _safe_text(resource.get("safe_display_label"), limit=120)
                resource_type = _safe_text(resource.get("resource_type"), limit=60)
                availability = _safe_text(resource.get("availability"), limit=40)
                modes = resource.get("configuration_modes")
                mode_text = ", ".join(
                    _safe_text(mode, limit=60)
                    for mode in modes
                ) if isinstance(modes, list) else "UNKNOWN"
                connection_ref = _safe_text(resource.get("connection_ref"), limit=100)
                lines.append(
                    f"{index}. {label} ({resource_type}) [{availability}] "
                    f"— {mode_text}; connection {connection_ref}"
                )
            return lines
        if action_type == "knowledge.capabilities.list":
            capabilities = data.get("capabilities")
            if not isinstance(capabilities, list):
                return ["Capabilities: 0"]
            lines = [f"Capabilities: {len(capabilities)}"]
            for item in capabilities:
                capability = _mapping(item)
                capability_id = _safe_text(capability.get("capability_id"), limit=120)
                read_only = bool(capability.get("read_only", False))
                bindable = bool(capability.get("bindable_read_only", False))
                scope = "resource-scoped" if capability.get("resource_scope_required") else "connection-scoped"
                limits = (
                    f"item limit: {_safe_limit(capability.get('max_result_items'))}; "
                    f"byte limit: {_safe_limit(capability.get('max_result_bytes'))}"
                )
                live = "live access eligible" if bindable else "not live-bindable"
                lines.append(
                    f"{capability_id} — {'read-only' if read_only else 'non-read-only'}, "
                    f"{scope}, {limits}, {live}"
                )
            return lines
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
            lines = [answer]
            citations = data.get("citations")
            if isinstance(citations, list):
                labels: list[str] = []
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
                    lines.append("")
                    lines.append("Sources:")
                    for index, label in enumerate(labels[:5], start=1):
                        lines.append(f"[{index}] {label}")
            return lines
        if action_type == "workspace.delete":
            status = _safe_text(data.get("status"), limit=40).casefold()
            name = _safe_text(data.get("name"), limit=200)
            if status == "confirmation_required":
                target = name or "this workspace"
                return [
                    f"Deleting workspace {target} is irreversible.",
                    "Reply with explicit confirmation to proceed.",
                ]
            if bool(data.get("deleted")):
                return [f"Workspace deleted: {name}" if name else "Workspace deleted."]
            return ["Workspace delete requested."]
        if action_type == "knowledge.inventory.list":
            workspace_name = _safe_text(data.get("workspace_name"), limit=200)
            header = f"Knowledge sources in {workspace_name}:" if workspace_name else "Knowledge sources:"
            items = data.get("items")
            if not isinstance(items, list) or not items:
                return [header, "No knowledge sources found."]
            lines = [header]
            for index, item in enumerate(items, start=1):
                entry = _mapping(item)
                label = _safe_text(entry.get("display_label"), limit=120) or f"Source {index}"
                mode = _safe_text(entry.get("mode"), limit=20)
                state = _safe_text(entry.get("lifecycle_state"), limit=40)
                parts = [f"{index}. {label} — {mode} — {state}"]
                if bool(entry.get("needs_attention")):
                    parts.append("needs attention")
                runtime = entry.get("runtime_available")
                if runtime is False:
                    parts.append("unavailable")
                last_sync = entry.get("last_successful_sync_at")
                if isinstance(last_sync, str) and last_sync.strip():
                    parts.append(f"last sync: {_safe_text(last_sync, limit=80)}")
                elif entry.get("mode") == "indexed" and last_sync is None:
                    parts.append("last sync: unknown")
                lines.append(" — ".join(parts))
            return lines
        if action_type == "knowledge.operation.execute":
            status = _safe_text(data.get("status"), limit=40).casefold()
            label = _safe_text(data.get("display_label"), limit=200)
            operation = _safe_text(data.get("operation"), limit=40)
            if status == "confirmation_required":
                target = label or "this source"
                return [
                    f"Detaching {target} is irreversible.",
                    "Reply with explicit confirmation to proceed.",
                ]
            item = _mapping(data.get("item"))
            item_label = _safe_text(item.get("display_label"), limit=200) or label
            op_text = operation.replace("_", " ") if operation else "operation"
            state = _safe_text(item.get("lifecycle_state"), limit=40)
            lines = [f"{op_text} accepted for {item_label}."]
            if state:
                lines.append(f"Current state: {state}.")
            return lines
        if action_type == "destructive.confirm":
            status = _safe_text(data.get("status"), limit=40).casefold()
            action_kind = _safe_text(data.get("action_kind"), limit=80)
            name = _safe_text(data.get("name"), limit=200)
            if status == "completed" and action_kind == "workspace.delete":
                return [f"Workspace deleted: {name}" if name else "Workspace deleted."]
            item = _mapping(data.get("item"))
            label = _safe_text(item.get("display_label"), limit=200)
            if label:
                return [f"Source detached: {label}."]
            return ["Destructive action completed."]
        if action_type == "citation.inspect":
            display_name = _safe_text(data.get("display_name"), limit=200)
            lines = []
            if display_name:
                lines.append(f"Source: {display_name}")
            location = _mapping(data.get("location"))
            page = location.get("page")
            logical_location = _safe_text(
                location.get("logical_location") or data.get("logical_location"),
                limit=200,
            )
            if isinstance(page, int) and page > 0:
                lines.append(f"Location: page {page}")
            elif logical_location:
                lines.append(f"Location: {logical_location}")
            preview = _safe_text(data.get("preview"), limit=1200)
            if preview:
                lines.append(f"Preview: {preview}")
            external_url = _safe_text(data.get("external_url"), limit=500)
            if external_url:
                lines.append(f"Open original: {external_url}")
            return lines or ["Source details are not available."]
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
