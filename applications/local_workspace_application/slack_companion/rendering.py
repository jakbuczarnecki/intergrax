# © Artur Czarnecki. All rights reserved.

"""Safe plain-text rendering of Ask results for Slack outbound messages."""

from __future__ import annotations

from local_workspace_application.slack_companion.models import (
    SlackAskHttpResponse,
    SlackWorkspaceListItem,
)

ACK_TEXT = "Checking the selected workspace…"
INSUFFICIENT_EVIDENCE_TEXT = (
    "I could not find enough verified information in the selected workspace "
    "to answer reliably."
)
GENERIC_ERROR_TEXT = "I could not complete this request. Please try again."
WORKSPACE_LIST_EMPTY_TEXT = "No available workspaces were found."
WORKSPACE_LIST_HEADER = "Available workspaces:"
WORKSPACE_SELECTED_PREFIX = "Selected workspace: "
WORKSPACE_OUT_OF_RANGE_TEXT = (
    "Workspace number is not available. Send `workspaces` to see the current list."
)
WORKSPACE_LIST_LOAD_FAILED_TEXT = (
    "I could not load the available workspaces. Please try again."
)
WORKSPACE_SELECTION_USAGE_TEXT = (
    "Use `workspace <number>` to select a workspace. "
    "Send `workspaces` to see the list."
)
SELECTED_WORKSPACE_UNAVAILABLE_TEXT = (
    "The selected workspace is no longer available. "
    "Send `workspaces` and select another one."
)
WORKSPACE_CREATED_PREFIX = "Workspace created and selected: "
WORKSPACE_CREATE_USAGE_TEXT = "Use `workspace create <name>` to create a workspace."
WORKSPACE_DELETE_CONFIRM_HEADER = "You are about to delete workspace: "
WORKSPACE_DELETE_CONFIRM_BODY = (
    "This removes its LKW index, registered sources and workspace state.\n"
    "Local source files will not be deleted.\n"
    "\n"
    "To confirm, send:\n"
    "workspace delete confirm"
)
WORKSPACE_DELETE_SUCCESS_PREFIX = "Workspace deleted: "
WORKSPACE_DELETE_SUCCESS_FOOTER = "Local source files were not changed."
WORKSPACE_DELETE_MISSING_PENDING_TEXT = (
    "There is no pending workspace deletion. "
    "Send `workspace delete <number>` first."
)
WORKSPACE_DELETE_USAGE_TEXT = (
    "Use `workspace delete <number>` to request deletion. "
    "Send `workspaces` to see the list."
)
WORKSPACE_DELETE_CANCELLED_TEXT = "Workspace deletion cancelled."
NO_WORKSPACE_AVAILABLE_TEXT = (
    "No workspace is available. "
    "Send `workspaces` to select one, or `workspace create <name>` to create one."
)

MAX_ANSWER_CHARS = 3000
MAX_SOURCE_LABELS = 5
MAX_WORKSPACE_NAME_CHARS = 100


def render_acknowledgement() -> str:
    return ACK_TEXT


def render_ask_response(response: SlackAskHttpResponse) -> str:
    if response.status == "completed":
        return _render_completed(response)
    if response.status == "insufficient_evidence":
        return _render_insufficient(response)
    return GENERIC_ERROR_TEXT


def render_error() -> str:
    return GENERIC_ERROR_TEXT


def render_workspace_list(
    workspaces: list[SlackWorkspaceListItem],
    *,
    active_workspace_id: str,
) -> str:
    """Render a safe numbered workspace list; never includes workspace/tenant IDs."""
    if not workspaces:
        return WORKSPACE_LIST_EMPTY_TEXT

    active_id = (active_workspace_id or "").strip()
    active_present = any(
        (item.workspace_id or "").strip() == active_id for item in workspaces
    )
    lines = [WORKSPACE_LIST_HEADER, ""]
    index = 0
    for item in workspaces:
        name = (item.name or "").strip()
        if not name:
            continue
        index += 1
        if active_present and (item.workspace_id or "").strip() == active_id:
            lines.append(f"{index}. {name} — active")
        else:
            lines.append(f"{index}. {name}")
    if index == 0:
        return WORKSPACE_LIST_EMPTY_TEXT
    return "\n".join(lines)


def render_workspace_selected(workspace_name: str) -> str:
    name = (workspace_name or "").strip()
    if not name:
        return WORKSPACE_LIST_EMPTY_TEXT
    return f"{WORKSPACE_SELECTED_PREFIX}{name}"


def render_workspace_out_of_range() -> str:
    return WORKSPACE_OUT_OF_RANGE_TEXT


def render_workspace_list_load_failed() -> str:
    return WORKSPACE_LIST_LOAD_FAILED_TEXT


def render_workspace_selection_usage() -> str:
    return WORKSPACE_SELECTION_USAGE_TEXT


def render_selected_workspace_unavailable() -> str:
    return SELECTED_WORKSPACE_UNAVAILABLE_TEXT


def render_workspace_created(workspace_name: str) -> str:
    name = (workspace_name or "").strip()
    if not name:
        return WORKSPACE_CREATE_USAGE_TEXT
    return f"{WORKSPACE_CREATED_PREFIX}{name}"


def render_workspace_create_usage() -> str:
    return WORKSPACE_CREATE_USAGE_TEXT


def render_workspace_delete_confirmation(workspace_name: str) -> str:
    name = (workspace_name or "").strip() or "unknown"
    return f"{WORKSPACE_DELETE_CONFIRM_HEADER}{name}\n\n{WORKSPACE_DELETE_CONFIRM_BODY}"


def render_workspace_deleted(workspace_name: str) -> str:
    name = (workspace_name or "").strip() or "unknown"
    return f"{WORKSPACE_DELETE_SUCCESS_PREFIX}{name}\n\n{WORKSPACE_DELETE_SUCCESS_FOOTER}"


def render_workspace_delete_missing_pending() -> str:
    return WORKSPACE_DELETE_MISSING_PENDING_TEXT


def render_workspace_delete_usage() -> str:
    return WORKSPACE_DELETE_USAGE_TEXT


def render_workspace_delete_cancelled() -> str:
    return WORKSPACE_DELETE_CANCELLED_TEXT


def render_no_workspace_available() -> str:
    return NO_WORKSPACE_AVAILABLE_TEXT


def safe_source_labels(response: SlackAskHttpResponse) -> list[str]:
    """Return deduplicated ``file_name`` labels in first-seen order (capped)."""
    seen: set[str] = set()
    labels: list[str] = []
    for citation in response.citations:
        label = (citation.file_name or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= MAX_SOURCE_LABELS:
            break
    return labels


def _render_completed(response: SlackAskHttpResponse) -> str:
    answer = (response.answer or "").strip()
    if not answer:
        return GENERIC_ERROR_TEXT
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[: MAX_ANSWER_CHARS - 1] + "…"
    labels = safe_source_labels(response)
    if not labels:
        return answer
    lines = [answer, "", "Sources:"]
    for index, label in enumerate(labels, start=1):
        lines.append(f"[{index}] {label}")
    omitted = _omitted_source_count(response, shown=len(labels))
    if omitted > 0:
        lines.append(f"(+{omitted} more sources)")
    return "\n".join(lines)


def _render_insufficient(response: SlackAskHttpResponse) -> str:
    labels = safe_source_labels(response)
    if not labels:
        return INSUFFICIENT_EVIDENCE_TEXT
    lines = [INSUFFICIENT_EVIDENCE_TEXT, "", "Sources:"]
    for index, label in enumerate(labels, start=1):
        lines.append(f"[{index}] {label}")
    return "\n".join(lines)


def _omitted_source_count(response: SlackAskHttpResponse, *, shown: int) -> int:
    unique: list[str] = []
    seen: set[str] = set()
    for citation in response.citations:
        label = (citation.file_name or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        unique.append(label)
    return max(0, len(unique) - shown)
