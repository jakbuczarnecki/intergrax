# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from local_workspace_application.slack_companion.models import (
    SlackAskCitationLabel,
    SlackAskHttpResponse,
)
from local_workspace_application.slack_companion.rendering import (
    GENERIC_ERROR_TEXT,
    INSUFFICIENT_EVIDENCE_TEXT,
    MAX_SOURCE_LABELS,
    render_ask_response,
    safe_source_labels,
)

pytestmark = pytest.mark.unit


def _response(
    *,
    status: str = "completed",
    answer: str | None = "Grounded answer",
    citations: list[SlackAskCitationLabel] | None = None,
) -> SlackAskHttpResponse:
    return SlackAskHttpResponse(
        run_id="run-1",
        workspace_id="ws-1",
        status=status,  # type: ignore[arg-type]
        question="Q?",
        answer=answer,
        citations=citations or [],
    )


def test_completed_answer_rendered_with_sources() -> None:
    text = render_ask_response(
        _response(
            citations=[
                SlackAskCitationLabel(file_name="a.pdf"),
                SlackAskCitationLabel(file_name="b.pdf"),
            ]
        )
    )
    assert "Grounded answer" in text
    assert "Sources:" in text
    assert "[1] a.pdf" in text
    assert "[2] b.pdf" in text


def test_insufficient_evidence_without_invented_answer() -> None:
    text = render_ask_response(
        _response(status="insufficient_evidence", answer="should-not-appear")
    )
    assert text.startswith(INSUFFICIENT_EVIDENCE_TEXT)
    assert "should-not-appear" not in text


def test_failed_rendered_generically() -> None:
    assert render_ask_response(_response(status="failed", answer="x")) == GENERIC_ERROR_TEXT


def test_file_name_used_source_path_and_excerpt_never_rendered() -> None:
    # Parsing from full Ask JSON keeps only file_name on the label model.
    parsed = SlackAskHttpResponse.model_validate(
        {
            "run_id": "r",
            "workspace_id": "ws",
            "status": "completed",
            "question": "Q",
            "answer": "Ans",
            "citations": [
                {
                    "file_name": "policy.pdf",
                    "source_path": "C:/Users/secret/policy.pdf",
                    "excerpt": "confidential paragraph",
                    "evidence_id": "e1",
                    "document_id": "d1",
                    "source_id": "s1",
                    "workspace_id": "ws",
                    "score": 0.99,
                    "chunk_id": "c1",
                }
            ],
        }
    )
    text = render_ask_response(parsed)
    assert "policy.pdf" in text
    assert "C:/Users/secret" not in text
    assert "confidential paragraph" not in text
    assert "evidence_id" not in text
    assert "chunk_id" not in text


def test_duplicate_source_labels_removed_and_limited() -> None:
    citations = [
        SlackAskCitationLabel(file_name="a.pdf"),
        SlackAskCitationLabel(file_name="a.pdf"),
        SlackAskCitationLabel(file_name="b.pdf"),
        SlackAskCitationLabel(file_name="c.pdf"),
        SlackAskCitationLabel(file_name="d.pdf"),
        SlackAskCitationLabel(file_name="e.pdf"),
        SlackAskCitationLabel(file_name="f.pdf"),
    ]
    labels = safe_source_labels(_response(citations=citations))
    assert labels == ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
    assert len(labels) == MAX_SOURCE_LABELS
    text = render_ask_response(_response(citations=citations))
    assert "[5] e.pdf" in text
    assert "f.pdf" not in text
    assert "+1 more sources" in text
