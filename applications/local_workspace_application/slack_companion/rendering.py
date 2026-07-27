# © Artur Czarnecki. All rights reserved.

"""Safe plain-text rendering of Ask results for Slack outbound messages."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from local_workspace_application.slack_companion.models import (
    SlackAskHttpResponse,
    SlackManagedFileBatchResponse,
    SlackSourceCandidateListItem,
    SlackSourceListItem,
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
SOURCE_LIST_HEADER = "Sources in the active workspace:"
SOURCE_LIST_EMPTY_TEXT = "The active workspace does not contain any sources yet."
SOURCE_LIST_LOAD_FAILED_TEXT = (
    "The source list could not be loaded. Please try again later."
)
SOURCE_WORKSPACE_UNAVAILABLE_TEXT = (
    "The active workspace is unavailable. "
    "Use `workspaces` to review available workspaces."
)
SOURCE_LIST_TRUNCATED_FOOTER = "Additional sources are not shown."

SOURCE_CANDIDATE_LIST_HEADER = "Available source candidates:"
SOURCE_CANDIDATE_LIST_EMPTY_TEXT = (
    "No source candidates are currently available for the active workspace."
)
SOURCE_CANDIDATE_LIST_LOAD_FAILED_TEXT = (
    "The source candidate list could not be loaded. Please try again later."
)
SOURCE_CANDIDATE_OUT_OF_RANGE_TEXT = (
    "Source candidate number is not available.\n"
    "Send `source candidates` to see the current list."
)
SOURCE_CANDIDATE_USAGE_TEXT = (
    "Use `source add <number>` to attach a source.\n"
    "Send `source candidates` to see the current list."
)
SOURCE_CANDIDATE_ACCEPTED_PREFIX = "Source accepted: "
SOURCE_CANDIDATE_ACCEPTED_FOOTER = "Processing continues asynchronously."
SOURCE_CANDIDATE_ALREADY_ATTACHED_TEXT = (
    "That source is already attached to the active workspace."
)
SOURCE_CANDIDATE_UNAVAILABLE_TEXT = (
    "That source candidate is not available right now."
)
SOURCE_CANDIDATE_ACCEPT_FAILED_TEXT = (
    "The source candidate could not be attached. Please try again later."
)
SOURCE_CANDIDATE_SERVICE_UNAVAILABLE_TEXT = (
    "Source candidates are temporarily unavailable. Please try again later."
)
SOURCE_CANDIDATE_LIST_FOOTER = "Use `source add <number>` to attach a source."

ATTACHMENT_FETCH_UNAVAILABLE_TEXT = (
    "File attachments are not available from this Slack connection right now."
)
ATTACHMENT_FETCH_FAILED_TEXT = (
    "The attached files could not be received from Slack. Please try again."
)
ATTACHMENT_INTAKE_FAILED_TEXT = (
    "The attached files could not be submitted for processing. Please try again."
)
ATTACHMENT_TOO_LARGE_TEXT = (
    "One or more attached files are too large to accept."
)
ATTACHMENT_ALL_FAILED_TEXT = "None of the attached files were accepted."
ATTACHMENT_PROCESSING_FOOTER = "Processing continues asynchronously."
ATTACHMENT_FILE_FALLBACK = "File"
ATTACHMENT_GENERIC_REJECT_TEXT = "File could not be accepted."

MAX_ANSWER_CHARS = 3000
MAX_SOURCE_LABELS = 5
MAX_WORKSPACE_NAME_CHARS = 100
MAX_SOURCE_LIST_ITEMS = 25
MAX_SOURCE_LIST_LABEL_CHARS = 80
MAX_SOURCE_CANDIDATE_ITEMS = 25
MAX_SOURCE_CANDIDATE_LABEL_CHARS = 80
MAX_SOURCE_CANDIDATE_DESCRIPTION_CHARS = 180
MAX_ATTACHMENT_DISPLAY_ITEMS = 10
MAX_ATTACHMENT_FILE_NAME_CHARS = 80
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_SAFE_SOURCE_CANDIDATE_LABEL_FALLBACK = "Source"

_MANAGED_FILE_ERROR_MESSAGES: dict[str, str] = {
    "managed_file_name_required": "File name is missing.",
    "managed_file_name_too_long": "File name is too long.",
    "managed_file_name_unsafe": "File name is not allowed.",
    "managed_file_extension_required": "File type is not supported.",
    "managed_file_content_type_invalid": "File type is not supported.",
    "managed_file_content_type_too_long": "File type is not supported.",
    "managed_file_body_required": "File content is missing.",
    "managed_file_empty": "Empty files are not accepted.",
    "managed_file_too_large": "File is too large.",
    "managed_file_upload_read_failed": ATTACHMENT_GENERIC_REJECT_TEXT,
    "managed_file_storage_read_failed": ATTACHMENT_GENERIC_REJECT_TEXT,
    "managed_file_storage_write_failed": ATTACHMENT_GENERIC_REJECT_TEXT,
    "managed_file_accept_failed": ATTACHMENT_GENERIC_REJECT_TEXT,
}


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


def render_source_list_empty() -> str:
    return SOURCE_LIST_EMPTY_TEXT


def render_source_list_load_failed() -> str:
    return SOURCE_LIST_LOAD_FAILED_TEXT


def render_source_workspace_unavailable() -> str:
    return SOURCE_WORKSPACE_UNAVAILABLE_TEXT


def render_source_list(sources: list[SlackSourceListItem]) -> str:
    """Render a safe numbered source list; never includes IDs, paths, or locators."""
    if not sources:
        return SOURCE_LIST_EMPTY_TEXT

    lines = [SOURCE_LIST_HEADER, ""]
    for index, item in enumerate(sources):
        if index >= MAX_SOURCE_LIST_ITEMS:
            lines.append(SOURCE_LIST_TRUNCATED_FOOTER)
            break
        label = _safe_source_display_label(item.label)
        if not label:
            label = "Source"
        shown = index + 1
        lines.append(f"{shown}. {label}")
        lines.append(f"   Type: {_format_source_type(item.source_type)}")
        lines.append(f"   Status: {_format_source_status(item.status)}")
        if _show_recursive(item.source_type):
            lines.append(f"   Recursive: {'yes' if item.recursive else 'no'}")
        lines.append(f"   Last sync: {_format_last_sync(item.last_sync_at)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_source_candidate_list(
    candidates: list[SlackSourceCandidateListItem],
) -> str:
    """Render a safe numbered Source Candidate list; never includes IDs or paths."""
    if not candidates:
        return SOURCE_CANDIDATE_LIST_EMPTY_TEXT

    lines = [SOURCE_CANDIDATE_LIST_HEADER, ""]
    for index, item in enumerate(candidates):
        if index >= MAX_SOURCE_CANDIDATE_ITEMS:
            break
        label = _safe_source_candidate_label(item.label)
        shown = index + 1
        lines.append(f"{shown}. {label}")
        description = _safe_source_candidate_description(item.description)
        if description:
            lines.append(f"   {description}")
        lines.append("")
    lines.append(SOURCE_CANDIDATE_LIST_FOOTER)
    return "\n".join(lines).rstrip() + "\n"


def render_source_candidate_list_empty() -> str:
    return SOURCE_CANDIDATE_LIST_EMPTY_TEXT


def render_source_candidate_list_load_failed() -> str:
    return SOURCE_CANDIDATE_LIST_LOAD_FAILED_TEXT


def render_source_candidate_workspace_unavailable() -> str:
    return SELECTED_WORKSPACE_UNAVAILABLE_TEXT


def render_source_candidate_out_of_range() -> str:
    return SOURCE_CANDIDATE_OUT_OF_RANGE_TEXT


def render_source_candidate_usage() -> str:
    return SOURCE_CANDIDATE_USAGE_TEXT


def render_source_candidate_accepted(label: str) -> str:
    safe = _safe_source_candidate_label(label)
    return (
        f"{SOURCE_CANDIDATE_ACCEPTED_PREFIX}{safe}\n\n"
        f"{SOURCE_CANDIDATE_ACCEPTED_FOOTER}"
    )


def render_source_candidate_already_attached() -> str:
    return SOURCE_CANDIDATE_ALREADY_ATTACHED_TEXT


def render_source_candidate_unavailable() -> str:
    return SOURCE_CANDIDATE_UNAVAILABLE_TEXT


def render_source_candidate_accept_failed() -> str:
    return SOURCE_CANDIDATE_ACCEPT_FAILED_TEXT


def render_source_candidate_service_unavailable() -> str:
    return SOURCE_CANDIDATE_SERVICE_UNAVAILABLE_TEXT


def _safe_source_candidate_label(value: str) -> str:
    cleaned = _normalize_source_candidate_text(value)
    if not cleaned:
        return _SAFE_SOURCE_CANDIDATE_LABEL_FALLBACK
    if len(cleaned) > MAX_SOURCE_CANDIDATE_LABEL_CHARS:
        return cleaned[: MAX_SOURCE_CANDIDATE_LABEL_CHARS - 1] + "…"
    return cleaned


def _safe_source_candidate_description(value: str) -> str:
    cleaned = _normalize_source_candidate_text(value)
    if not cleaned:
        return ""
    if len(cleaned) > MAX_SOURCE_CANDIDATE_DESCRIPTION_CHARS:
        return cleaned[: MAX_SOURCE_CANDIDATE_DESCRIPTION_CHARS - 1] + "…"
    return cleaned


def _normalize_source_candidate_text(value: str) -> str:
    cleaned = (value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _safe_source_display_label(value: str) -> str:
    cleaned = (value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    if len(cleaned) > MAX_SOURCE_LIST_LABEL_CHARS:
        return cleaned[: MAX_SOURCE_LIST_LABEL_CHARS - 1] + "…"
    return cleaned


def _format_source_type(source_type: str) -> str:
    raw = (source_type or "").strip()
    if not raw:
        return "source"
    return _safe_source_display_label(raw.replace("_", " ")).casefold() or "source"


def _format_source_status(status: str) -> str:
    raw = (status or "").strip()
    if not raw:
        return "unknown"
    return _safe_source_display_label(raw.replace("_", " ")).casefold() or "unknown"


def _show_recursive(source_type: str) -> bool:
    # Recursive is meaningful only for local_folder; hide for all other types.
    return (source_type or "").strip().casefold() == "local_folder"


def _format_last_sync(value: datetime | None) -> str:
    if value is None:
        return "never"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.strftime("%Y-%m-%d %H:%M UTC")


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


def _safe_attachment_display_name(value: str | None) -> str:
    cleaned = (value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ATTACHMENT_FILE_FALLBACK
    if len(cleaned) > MAX_ATTACHMENT_FILE_NAME_CHARS:
        return cleaned[: MAX_ATTACHMENT_FILE_NAME_CHARS - 1] + "…"
    return cleaned


def _managed_file_error_message(error_code: str | None) -> str:
    code = (error_code or "").strip()
    if not code:
        return ATTACHMENT_GENERIC_REJECT_TEXT
    return _MANAGED_FILE_ERROR_MESSAGES.get(code, ATTACHMENT_GENERIC_REJECT_TEXT)


def render_attachment_receiving(count: int) -> str:
    n = max(0, int(count))
    noun = "file" if n == 1 else "files"
    return f"Receiving {n} attached {noun} for the selected workspace…"


def render_attachment_too_many(limit: int) -> str:
    return (
        f"Too many files were attached. "
        f"Please send at most {max(1, int(limit))} files at a time."
    )


def render_attachment_fetch_unavailable() -> str:
    return ATTACHMENT_FETCH_UNAVAILABLE_TEXT


def render_attachment_fetch_failed(kind: str) -> str:
    normalized = (kind or "").strip()
    if normalized == "attachment_too_large":
        return ATTACHMENT_TOO_LARGE_TEXT
    if normalized == "attachment_fetch_unavailable":
        return ATTACHMENT_FETCH_UNAVAILABLE_TEXT
    return ATTACHMENT_FETCH_FAILED_TEXT


def render_attachment_intake_failed() -> str:
    return ATTACHMENT_INTAKE_FAILED_TEXT


def render_attachment_batch_response(response: SlackManagedFileBatchResponse) -> str:
    accepted = [
        item for item in response.items if (item.status or "").strip() == "accepted"
    ]
    rejected = [
        item for item in response.items if (item.status or "").strip() == "failed"
    ]
    accepted_count = int(response.accepted_count)
    failed_count = int(response.failed_count)

    if accepted_count <= 0 and failed_count > 0:
        lines = [ATTACHMENT_ALL_FAILED_TEXT]
        if rejected:
            lines.append("")
            lines.append("Rejected:")
            for index, item in enumerate(rejected[:MAX_ATTACHMENT_DISPLAY_ITEMS], start=1):
                name = _safe_attachment_display_name(item.file_name)
                reason = _managed_file_error_message(item.error_code)
                lines.append(f"{index}. {name} — {reason}")
            omitted = max(0, len(rejected) - MAX_ATTACHMENT_DISPLAY_ITEMS)
            if omitted > 0:
                lines.append(f"(+{omitted} more files omitted)")
        return "\n".join(lines)

    if failed_count <= 0:
        noun = "file" if accepted_count == 1 else "files"
        lines = [f"Accepted {accepted_count} {noun} for processing.", ""]
        for index, item in enumerate(accepted[:MAX_ATTACHMENT_DISPLAY_ITEMS], start=1):
            lines.append(f"{index}. {_safe_attachment_display_name(item.file_name)}")
        omitted = max(0, len(accepted) - MAX_ATTACHMENT_DISPLAY_ITEMS)
        if omitted > 0:
            lines.append(f"(+{omitted} more files omitted)")
        lines.append("")
        lines.append(ATTACHMENT_PROCESSING_FOOTER)
        return "\n".join(lines)

    lines = [
        f"Accepted {accepted_count} of {accepted_count + failed_count} files for processing.",
        "",
        "Accepted:",
    ]
    for index, item in enumerate(accepted[:MAX_ATTACHMENT_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {_safe_attachment_display_name(item.file_name)}")
    omitted_accepted = max(0, len(accepted) - MAX_ATTACHMENT_DISPLAY_ITEMS)
    if omitted_accepted > 0:
        lines.append(f"(+{omitted_accepted} more files omitted)")
    lines.extend(["", "Rejected:"])
    for index, item in enumerate(rejected[:MAX_ATTACHMENT_DISPLAY_ITEMS], start=1):
        name = _safe_attachment_display_name(item.file_name)
        reason = _managed_file_error_message(item.error_code)
        lines.append(f"{index}. {name} — {reason}")
    omitted_rejected = max(0, len(rejected) - MAX_ATTACHMENT_DISPLAY_ITEMS)
    if omitted_rejected > 0:
        lines.append(f"(+{omitted_rejected} more files omitted)")
    lines.extend(["", ATTACHMENT_PROCESSING_FOOTER])
    return "\n".join(lines)
