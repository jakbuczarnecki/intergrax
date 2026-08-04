from __future__ import annotations

from datetime import UTC, datetime

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionResult,
    ConversationActionExecutionStatus,
    ConversationExecutionArtifact,
    ConversationExecutionClarification,
    ConversationExecutionError,
    ConversationInteractionExecutionResult,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
    MAX_RESPONSE_CHARS,
)


def _result(
    *,
    actions: tuple[ConversationActionExecutionResult, ...] = (),
    clarifications: tuple[ConversationExecutionClarification, ...] = (),
    status: ConversationInteractionOverallStatus = ConversationInteractionOverallStatus.COMPLETED,
    error: ConversationExecutionError | None = None,
) -> ConversationInteractionExecutionResult:
    now = datetime.now(UTC)
    return ConversationInteractionExecutionResult(
        execution_id="execution-1",
        tenant_id="tenant-a",
        plan_version="2",
        started_at=now,
        completed_at=now,
        status=status,
        action_results=actions,
        clarifications=clarifications,
        error=error,
    )


def _action(
    *,
    action_id: str,
    action_type: str,
    status: ConversationActionExecutionStatus,
    data: dict[str, object] | None = None,
    error: ConversationExecutionError | None = None,
) -> ConversationActionExecutionResult:
    now = datetime.now(UTC)
    return ConversationActionExecutionResult(
        action_id=action_id,
        action_type=action_type,
        status=status,
        artifact=(
            ConversationExecutionArtifact(
                artifact_type=action_type,
                data=data or {},
            )
            if status is ConversationActionExecutionStatus.COMPLETED
            else None
        ),
        error=error,
        started_at=now,
        completed_at=now,
    )


def test_renderer_keeps_completed_failure_and_clarification_safe() -> None:
    text = ConversationInteractionResponseRenderer().render(
        _result(
            status=ConversationInteractionOverallStatus.PARTIALLY_COMPLETED,
            actions=(
                _action(
                    action_id="create",
                    action_type="workspace.create",
                    status=ConversationActionExecutionStatus.COMPLETED,
                    data={"name": "Project Alfa"},
                ),
                _action(
                    action_id="ask",
                    action_type="workspace.ask",
                    status=ConversationActionExecutionStatus.FAILED,
                    error=ConversationExecutionError(
                        code="active_workspace_required",
                        action_id="ask",
                    ),
                ),
                _action(
                    action_id="blocked",
                    action_type="workspace.ask",
                    status=ConversationActionExecutionStatus.BLOCKED_DEPENDENCY,
                    error=ConversationExecutionError(
                        code="blocked_dependency",
                        action_id="blocked",
                    ),
                ),
            ),
            clarifications=(
                ConversationExecutionClarification(
                    clarification_id="clarify-1",
                    question="Which workspace should I use?",
                    blocks_action_ids=("blocked",),
                ),
            ),
        )
    )

    assert "Workspace created: Project Alfa" in text
    assert "Failed:" in text
    assert "Blocked:" in text
    assert "Which workspace should I use?" in text
    assert "tenant-a" not in text
    assert "ask" not in text


def test_renderer_includes_safe_ask_citation_and_is_bounded() -> None:
    text = ConversationInteractionResponseRenderer().render(
        _result(
            actions=(
                _action(
                    action_id="ask",
                    action_type="workspace.ask",
                    status=ConversationActionExecutionStatus.COMPLETED,
                    data={
                        "answer": "A" * 10_000,
                        "status": "completed",
                        "citations": [
                            {
                                "file_name": "policy.pdf",
                                "source_path": "C:\\secret\\policy.pdf",
                                "source_id": "source-internal",
                            }
                        ],
                    },
                ),
            )
        )
    )

    assert len(text) <= MAX_RESPONSE_CHARS
    assert "Question answered:" in text
    assert "policy.pdf" in text
    assert "C:\\secret" not in text
    assert "source-internal" not in text


def test_renderer_maps_top_level_failure_without_exception_details() -> None:
    text = ConversationInteractionResponseRenderer().render(
        _result(
            status=ConversationInteractionOverallStatus.FAILED,
            error=ConversationExecutionError(code="conversation_planning_failed"),
        )
    )

    assert text == "I could not understand the requested workspace operation safely."
    assert "conversation_planning_failed" not in text
