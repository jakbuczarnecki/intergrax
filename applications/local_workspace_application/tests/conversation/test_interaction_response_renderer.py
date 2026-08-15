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
    _bounded,
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
    assert "policy.pdf" in text
    assert "Sources:" in text
    assert "[1] policy.pdf" in text
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


def test_renderer_bounds_many_long_failure_lines() -> None:
    text = ConversationInteractionResponseRenderer().render(
        _result(
            status=ConversationInteractionOverallStatus.PARTIALLY_COMPLETED,
            actions=(
                _action(
                    action_id="completed",
                    action_type="workspace.create",
                    status=ConversationActionExecutionStatus.COMPLETED,
                    data={"name": "Project Alfa"},
                ),
                *(
                    _action(
                        action_id=f"failure-{index}",
                        action_type="workspace.ask",
                        status=ConversationActionExecutionStatus.FAILED,
                        error=ConversationExecutionError(
                            code=(
                                "conversation_execution_failed"
                                if index % 2
                                else "conversation_planning_failed"
                            ),
                            action_id=f"failure-{index}",
                        ),
                    )
                    for index in range(50)
                ),
            ),
        )
    )

    assert 0 < len(text) <= MAX_RESPONSE_CHARS
    assert "Failed:" in text


def test_renderer_bounds_many_long_clarifications_and_keeps_priority() -> None:
    result = _result(
        status=ConversationInteractionOverallStatus.PARTIALLY_COMPLETED,
        clarifications=tuple(
            ConversationExecutionClarification(
                clarification_id=f"clarify-{index}",
                question=f"Question {index}: " + "x" * 1_000,
                blocks_action_ids=(),
            )
            for index in range(50)
        ),
        actions=(
            _action(
                action_id="completed",
                action_type="workspace.create",
                status=ConversationActionExecutionStatus.COMPLETED,
                data={"name": "Project Alfa"},
            ),
            _action(
                action_id="failure",
                action_type="workspace.ask",
                status=ConversationActionExecutionStatus.FAILED,
                error=ConversationExecutionError(
                    code="active_workspace_required",
                    action_id="failure",
                ),
            ),
        ),
    )

    text = ConversationInteractionResponseRenderer().render(result)

    assert 0 < len(text) <= MAX_RESPONSE_CHARS
    assert "Clarification needed:" in text


def test_renderer_mixed_content_is_bounded_and_deterministic() -> None:
    result = _result(
        status=ConversationInteractionOverallStatus.PARTIALLY_COMPLETED,
        actions=(
            _action(
                action_id="completed",
                action_type="workspace.create",
                status=ConversationActionExecutionStatus.COMPLETED,
                data={"name": "Project Alfa"},
            ),
            _action(
                action_id="failed",
                action_type="workspace.ask",
                status=ConversationActionExecutionStatus.FAILED,
                error=ConversationExecutionError(
                    code="active_workspace_required",
                    action_id="failed",
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
                clarification_id="clarify",
                question="Which workspace should I use? " + "y" * 1_900,
                blocks_action_ids=("blocked",),
            ),
        ),
    )

    first = ConversationInteractionResponseRenderer().render(result)
    second = ConversationInteractionResponseRenderer().render(result)

    assert first == second
    assert 0 < len(first) <= MAX_RESPONSE_CHARS
    assert "Clarification needed:" in first or "Failed:" in first


def test_bounded_prefix_regression_and_ellipsis_stay_within_limit() -> None:
    text = _bounded(
        ["Completed: " + "c" * MAX_RESPONSE_CHARS],
        preserved=[
            f"Clarification needed: {index} " + "q" * MAX_RESPONSE_CHARS
            for index in range(50)
        ],
    )

    assert 0 < len(text) <= MAX_RESPONSE_CHARS
    assert text.startswith("Clarification needed:")
    assert text.endswith("…")


def test_renderer_maps_first_run_errors_without_internal_leakage() -> None:
    renderer = ConversationInteractionResponseRenderer()
    for code, fragment in (
        ("citation_not_available", "no longer available"),
        ("document_forbidden", "do not have access"),
        ("ask_unavailable", "temporarily unavailable"),
        ("Traceback (most recent call last)", "could not complete"),
    ):
        if code.startswith("Traceback"):
            text = renderer.render(
                _result(
                    status=ConversationInteractionOverallStatus.FAILED,
                    actions=(
                        _action(
                            action_id="a1",
                            action_type="workspace.ask",
                            status=ConversationActionExecutionStatus.FAILED,
                            error=ConversationExecutionError(
                                code="action_execution_failed",
                                action_id="a1",
                            ),
                        ),
                    ),
                )
            )
            assert "Traceback" not in text
            continue
        text = renderer.render(
            _result(
                status=ConversationInteractionOverallStatus.FAILED,
                actions=(
                    _action(
                        action_id="a1",
                        action_type="citation.inspect",
                        status=ConversationActionExecutionStatus.FAILED,
                        error=ConversationExecutionError(code=code, action_id="a1"),
                    ),
                ),
            )
        )
        assert fragment.casefold() in text.casefold()
        assert "C:\\" not in text
        assert "Traceback" not in text
