# © Artur Czarnecki. All rights reserved.

"""Safe plain-text rendering of Ask results for Slack outbound messages."""

from __future__ import annotations

from local_workspace_application.slack_companion.models import SlackAskHttpResponse

ACK_TEXT = "Checking the selected workspace…"
INSUFFICIENT_EVIDENCE_TEXT = (
    "I could not find enough verified information in the selected workspace "
    "to answer reliably."
)
GENERIC_ERROR_TEXT = "I could not complete this request. Please try again."

MAX_ANSWER_CHARS = 3000
MAX_SOURCE_LABELS = 5


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
