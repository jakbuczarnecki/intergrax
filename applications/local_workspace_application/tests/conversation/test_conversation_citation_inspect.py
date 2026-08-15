# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationInteractionExecutionCommand,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_models import (
    CitationInspectPlannedAction,
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
)
from local_workspace_application.workspaces.ask_models import (
    AskCitation,
    AskCitationLocation,
    AskRunStatus,
    WorkspaceAskRun,
)
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.conversation_citation_context_service import (
    ConversationCitationContextService,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
)
from local_workspace_application.workspaces.document_inspect_service import (
    DocumentInspectService,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)


def _execution_context() -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id="tenant-a",
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="ws-1",
        principal_ref="user-1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(ConversationProductCapability),
    )


def _active_workspace() -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind.active, value=None)


def _planning_request() -> ConversationPlanningRequest:
    return ConversationPlanningRequest(message_text="show source 1")


def _seed_repository(repo: ManagedWorkspaceRepository) -> None:
    repo.put_workspace(
        Workspace(
            workspace_id="ws-1",
            tenant_id="tenant-a",
            name="Docs",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repo.put_source(
        WorkspaceSource(
            source_id="src-1",
            workspace_id="ws-1",
            tenant_id="tenant-a",
            source_type=WorkspaceSourceType.MANAGED_UPLOAD,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.READY,
            created_at=_NOW,
        )
    )
    repo.put_document_ref(
        WorkspaceDocumentReference(
            document_id="doc-1",
            tenant_id="tenant-a",
            workspace_id="ws-1",
            source_id="src-1",
            source_path="managed/report.pdf",
            file_name="report.pdf",
            content_hash="sha256:" + "a" * 64,
            indexed_at=_NOW,
        )
    )


def _seed_ask_run(ask_repo: WorkspaceAskRepository) -> WorkspaceAskRun:
    run = WorkspaceAskRun(
        run_id="run-1",
        tenant_id="tenant-a",
        workspace_id="ws-1",
        question="What are payment terms?",
        status=AskRunStatus.COMPLETED,
        answer="Net 30.",
        citations=[
            AskCitation(
                evidence_id="E1",
                document_id="doc-1",
                source_id="src-1",
                workspace_id="ws-1",
                source_path="managed/report.pdf",
                file_name="report.pdf",
                excerpt="Payment is due within 30 days.",
                location=AskCitationLocation(page=4),
            )
        ],
        created_at=_NOW,
        completed_at=_NOW,
    )
    ask_repo.put_run(run)
    return run


@pytest.fixture
def executor_bundle() -> dict[str, object]:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repository(repo)
    context_repo = ConversationContextRepository(store)
    ask_repo = WorkspaceAskRepository(store)
    _seed_ask_run(ask_repo)
    citation_context = ConversationCitationContextService(
        context_repository=context_repo,
        ask_repository=ask_repo,
        clock=lambda: _NOW,
    )
    citation_context.record_ask_run(
        context=_execution_context(),
        run_id="run-1",
        workspace_id="ws-1",
    )
    workspace_service = MagicMock()
    workspace_service.get_workspace.return_value = SimpleNamespace(workspace_id="ws-1")
    selection_service = MagicMock()
    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,
        workspace_selection_service=selection_service,
        ask_service=MagicMock(),
        document_inspect_service=DocumentInspectService(repository=repo),
        citation_context_service=citation_context,
    )
    return {
        "executor": executor,
        "citation_context": citation_context,
        "ask_repo": ask_repo,
        "repo": repo,
    }


@pytest.mark.asyncio
async def test_grounded_answer_citation_still_rendered_from_workspace_ask() -> None:
    renderer = ConversationInteractionResponseRenderer()
    from local_workspace_application.conversation.interaction_execution_models import (
        ConversationActionExecutionResult,
        ConversationActionExecutionStatus,
        ConversationExecutionArtifact,
        ConversationInteractionExecutionResult,
    )

    now = _NOW
    result = ConversationInteractionExecutionResult(
        execution_id="exec-1",
        tenant_id="tenant-a",
        plan_version="2",
        started_at=now,
        completed_at=now,
        status=ConversationInteractionOverallStatus.COMPLETED,
        action_results=(
            ConversationActionExecutionResult(
                action_id="ask-1",
                action_type="workspace.ask",
                status=ConversationActionExecutionStatus.COMPLETED,
                artifact=ConversationExecutionArtifact(
                    artifact_type="workspace.ask",
                    data={
                        "answer": "Net 30.",
                        "status": "completed",
                        "citations": [
                            {"file_name": "report.pdf"},
                        ],
                    },
                ),
                started_at=now,
                completed_at=now,
            ),
        ),
    )
    text = renderer.render(result)
    assert "Net 30." in text
    assert "[1] report.pdf" in text


@pytest.mark.asyncio
async def test_inspect_citation_by_ordinal(executor_bundle: dict[str, object]) -> None:
    executor: ConversationInteractionExecutor = executor_bundle["executor"]  # type: ignore[assignment]
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            CitationInspectPlannedAction(
                action_id="inspect-1",
                action_type="citation.inspect",
                workspace=_active_workspace(),
                citation_ordinal=1,
            ),
        ),
        response_mode="aggregate",
    )
    command = ConversationInteractionExecutionCommand(
        tenant_id="tenant-a",
        planning_request=_planning_request(),
        interaction_plan=plan,
        execution_context=_execution_context(),
        execution_id="exec-1",
    )
    result = await executor.execute(command)
    assert result.status is ConversationInteractionOverallStatus.COMPLETED
    artifact = result.action_results[0].artifact
    assert artifact is not None
    assert artifact.data["display_name"] == "report.pdf"
    assert artifact.data["preview"] == "Payment is due within 30 days."
    assert "managed/report.pdf" not in str(artifact.data)


@pytest.mark.asyncio
async def test_invalid_citation_ordinal_handled_safely(executor_bundle: dict[str, object]) -> None:
    executor: ConversationInteractionExecutor = executor_bundle["executor"]  # type: ignore[assignment]
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            CitationInspectPlannedAction(
                action_id="inspect-1",
                action_type="citation.inspect",
                workspace=_active_workspace(),
                citation_ordinal=9,
            ),
        ),
        response_mode="aggregate",
    )
    command = ConversationInteractionExecutionCommand(
        tenant_id="tenant-a",
        planning_request=_planning_request(),
        interaction_plan=plan,
        execution_context=_execution_context(),
        execution_id="exec-2",
    )
    result = await executor.execute(command)
    assert result.status is ConversationInteractionOverallStatus.FAILED
    assert result.action_results[0].error is not None
    assert result.action_results[0].error.code == "citation_ordinal_invalid"


@pytest.mark.asyncio
async def test_citation_disappeared_handled_safely(executor_bundle: dict[str, object]) -> None:
    ask_repo: WorkspaceAskRepository = executor_bundle["ask_repo"]  # type: ignore[assignment]
    ask_repo.delete_run(tenant_id="tenant-a", run_id="run-1")
    executor: ConversationInteractionExecutor = executor_bundle["executor"]  # type: ignore[assignment]
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            CitationInspectPlannedAction(
                action_id="inspect-1",
                action_type="citation.inspect",
                workspace=_active_workspace(),
                citation_ordinal=1,
            ),
        ),
        response_mode="aggregate",
    )
    command = ConversationInteractionExecutionCommand(
        tenant_id="tenant-a",
        planning_request=_planning_request(),
        interaction_plan=plan,
        execution_context=_execution_context(),
        execution_id="exec-3",
    )
    result = await executor.execute(command)
    assert result.action_results[0].error is not None
    assert result.action_results[0].error.code == "citation_not_available"


@pytest.mark.asyncio
async def test_citation_inspect_survives_service_recreation(executor_bundle: dict[str, object]) -> None:
    repo: ManagedWorkspaceRepository = executor_bundle["repo"]  # type: ignore[assignment]
    store = repo.document_store
    context_repo = ConversationContextRepository(store)
    ask_repo = WorkspaceAskRepository(store)
    citation_context = ConversationCitationContextService(
        context_repository=context_repo,
        ask_repository=ask_repo,
        clock=lambda: _NOW,
    )
    executor = ConversationInteractionExecutor(
        workspace_service=MagicMock(
            get_workspace=MagicMock(return_value=SimpleNamespace(workspace_id="ws-1"))
        ),
        workspace_selection_service=MagicMock(),
        document_inspect_service=DocumentInspectService(repository=repo),
        citation_context_service=citation_context,
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            CitationInspectPlannedAction(
                action_id="inspect-1",
                action_type="citation.inspect",
                workspace=_active_workspace(),
                citation_ordinal=1,
            ),
        ),
        response_mode="aggregate",
    )
    command = ConversationInteractionExecutionCommand(
        tenant_id="tenant-a",
        planning_request=_planning_request(),
        interaction_plan=plan,
        execution_context=_execution_context(),
        execution_id="exec-4",
    )
    result = await executor.execute(command)
    assert result.status is ConversationInteractionOverallStatus.COMPLETED


def test_renderer_escapes_unsafe_citation_metadata() -> None:
    renderer = ConversationInteractionResponseRenderer()
    from local_workspace_application.conversation.interaction_execution_models import (
        ConversationActionExecutionResult,
        ConversationActionExecutionStatus,
        ConversationExecutionArtifact,
        ConversationInteractionExecutionResult,
        ConversationInteractionOverallStatus,
    )

    now = _NOW
    result = ConversationInteractionExecutionResult(
        execution_id="exec-5",
        tenant_id="tenant-a",
        plan_version="2",
        started_at=now,
        completed_at=now,
        status=ConversationInteractionOverallStatus.COMPLETED,
        action_results=(
            ConversationActionExecutionResult(
                action_id="inspect-1",
                action_type="citation.inspect",
                status=ConversationActionExecutionStatus.COMPLETED,
                artifact=ConversationExecutionArtifact(
                    artifact_type="citation.inspect",
                    data={
                        "display_name": "<script>alert(1)</script>",
                        "preview": "line\x00break",
                        "external_url": "https://example.com/safe",
                    },
                ),
                started_at=now,
                completed_at=now,
            ),
        ),
    )
    text = renderer.render(result)
    assert "<script>" not in text
    assert "\x00" not in text
    assert "Open original: https://example.com/safe" in text
