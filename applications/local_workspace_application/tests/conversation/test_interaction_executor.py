from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionResult,
    ConversationActionExecutionStatus,
    ConversationExecutionArtifact,
    ConversationExecutionError,
    ConversationInteractionExecutionCommand,
    ConversationInteractionExecutionResult,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningAttachment,
    ConversationPlanningRequest,
    ConversationPlanningSourceCandidate,
    ConversationPlanningWorkspace,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    LocalFileReferenceExtractedObject,
    MessageTextEvidenceSpan,
    SourceCandidateAttachPlannedAction,
    SourceCandidateListPlannedAction,
    SourceListPlannedAction,
    WebUrlExtractedObject,
    WorkspaceActivatePlannedAction,
    WorkspaceAskPlannedAction,
    WorkspaceCreatePlannedAction,
    WorkspaceDeletePlannedAction,
    WorkspaceListPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.workspaces.destructive_action_confirmation import (
    HmacDestructiveActionConfirmationCodec,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)


_CONFIRM_SECRET = b"test-confirmation-secret"
_CONFIRM_CODEC = HmacDestructiveActionConfirmationCodec(secret=_CONFIRM_SECRET)

class WorkspaceServiceFake:
    def __init__(self) -> None:
        self.workspaces = {
            "w1": SimpleNamespace(workspace_id="w1", name="old", workspace_revision=1),
            "w2": SimpleNamespace(workspace_id="w2", name="current", workspace_revision=1),
        }
        self.calls: list[str] = []

    def list_workspaces(self, *, tenant_id: str) -> list[object]:
        self.calls.append(f"list:{tenant_id}")
        return list(self.workspaces.values())

    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> object | None:
        self.calls.append(f"get:{tenant_id}:{workspace_id}")
        return self.workspaces.get(workspace_id)

    def create_workspace(self, *, tenant_id: str, name: str) -> object:
        self.calls.append(f"create:{tenant_id}:{name}")
        workspace = SimpleNamespace(workspace_id="w3", name=name, workspace_revision=1)
        self.workspaces["w3"] = workspace
        return workspace

    def delete_workspace(self, *, tenant_id: str, workspace_id: str) -> bool:
        self.calls.append(f"delete:{tenant_id}:{workspace_id}")
        return self.workspaces.pop(workspace_id, None) is not None

    def list_sources(self, *, tenant_id: str, workspace_id: str) -> list[object] | None:
        self.calls.append(f"sources:{tenant_id}:{workspace_id}")
        return [
            SimpleNamespace(
                source_id="source-1",
                source_type=SimpleNamespace(value="web_url"),
                status=SimpleNamespace(value="registered"),
            )
        ]


class SelectionServiceFake:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def select_personal_workspace(
        self,
        *,
        execution_context: ConversationExecutionContextV1,
        workspace_id: str,
    ) -> object:
        self.calls.append(workspace_id)
        return SimpleNamespace(
            selected_workspace_id=workspace_id,
            previous_workspace_id=execution_context.workspace_id,
            configuration_version=2,
            changed=workspace_id != execution_context.workspace_id,
        )


class CandidateServiceFake:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_candidates(self, *, tenant_id: str, workspace_id: str) -> tuple[object, ...]:
        self.calls.append(f"list:{workspace_id}")
        return (
            SimpleNamespace(
                candidate_id="candidate-1",
                label="files",
                source_type="local_folder",
                available=True,
            ),
        )

    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        candidate_id: str,
        idempotency_key: str,
    ) -> object:
        self.calls.append(f"accept:{workspace_id}:{candidate_id}:{idempotency_key}")
        return SimpleNamespace(
            source_id="source-candidate-1",
            operation_id="operation-1",
            status="queued",
        )


class AttachmentServiceFake:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def accept_many(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        uploads: list[object],
    ) -> object:
        self.calls.append(tuple(str(item) for item in uploads))
        return SimpleNamespace(batch_id="batch-1", status="accepted")


class WebUrlServiceFake:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def accept(self, **kwargs: object) -> object:
        self.calls.append(str(kwargs["raw_url"]))
        return SimpleNamespace(source_id="web-source", operation_id="web-op", status="queued")


class LocalReferenceServiceFake:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def accept(self, **kwargs: object) -> object:
        reference = kwargs["reference"]
        self.calls.append(reference.value)
        return SimpleNamespace(source_id="local-source", operation_id="local-op", status="queued")


class AskServiceFake:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def ask(self, **kwargs: object) -> object:
        self.calls.append(str(kwargs["question"]))
        return SimpleNamespace(
            run_id="run-1",
            status=SimpleNamespace(value="completed"),
            answer="grounded answer",
            citations=[{"source_id": "source-1"}],
        )


def _context(
    *,
    tenant_id: str = "tenant-1",
    capabilities: frozenset[ConversationProductCapability] | None = None,
) -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id=tenant_id,
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="w1",
        principal_ref="principal-1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=capabilities
        or frozenset(ConversationProductCapability),
    )


def _reference(kind: WorkspaceReferenceKind, value: str | None = None) -> WorkspaceReference:
    return WorkspaceReference(kind=kind, value=value)


@pytest.mark.asyncio
async def test_executor_maps_all_ten_actions_to_injected_services() -> None:
    message = "old current new https://example.test/a C:\\data"
    url_start = message.index("https://")
    path_start = message.index("C:\\")
    objects = (
        WebUrlExtractedObject(
            object_id="object-url",
            object_type="web_url",
            value="https://example.test/a",
            evidence=MessageTextEvidenceSpan(
                source="message_text",
                start=url_start,
                end=url_start + len("https://example.test/a"),
                text="https://example.test/a",
            ),
        ),
        LocalFileReferenceExtractedObject(
            object_id="object-path",
            object_type="local_file_reference",
            reference_kind="folder",
            value="C:\\data",
            evidence=MessageTextEvidenceSpan(
                source="message_text",
                start=path_start,
                end=path_start + len("C:\\data"),
                text="C:\\data",
            ),
        ),
    )
    request = ConversationPlanningRequest(
        message_text=message,
        attachments=(
            ConversationPlanningAttachment(attachment_id="attachment-1", file_name="a.txt"),
        ),
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="w1", name="old", is_active=True),
            ConversationPlanningWorkspace(workspace_id="w2", name="current", is_active=False),
        ),
        available_source_candidates=(
            ConversationPlanningSourceCandidate(
                candidate_id="candidate-1",
                label="files",
                source_type="local_folder",
                available=True,
            ),
        ),
    )
    actions = (
        WorkspaceListPlannedAction(action_id="a1", action_type="workspace.list"),
        WorkspaceCreatePlannedAction(action_id="a2", action_type="workspace.create", name="new"),
        WorkspaceActivatePlannedAction(
            action_id="a3",
            action_type="workspace.activate",
            workspace=_reference(WorkspaceReferenceKind.created_by_action, "a2"),
            depends_on=("a2",),
        ),
        WorkspaceDeletePlannedAction(
            action_id="a4",
            action_type="workspace.delete",
            workspace=_reference(WorkspaceReferenceKind.name, "old"),
        ),
        SourceListPlannedAction(
            action_id="a5",
            action_type="source.list",
            workspace=_reference(WorkspaceReferenceKind.active),
            depends_on=("a3",),
        ),
        SourceCandidateListPlannedAction(
            action_id="a6",
            action_type="source_candidate.list",
            workspace=_reference(WorkspaceReferenceKind.active),
            depends_on=("a3",),
        ),
        SourceCandidateAttachPlannedAction(
            action_id="a7",
            action_type="source_candidate.attach",
            workspace=_reference(WorkspaceReferenceKind.active),
            candidate_reference_kind="ordinal",
            candidate_reference="1",
            depends_on=("a3",),
        ),
        KnowledgeAddAttachmentsPlannedAction(
            action_id="a8",
            action_type="knowledge.add_attachments",
            workspace=_reference(WorkspaceReferenceKind.active),
            attachment_ids=("attachment-1",),
            depends_on=("a3",),
        ),
        KnowledgeAddSourcesPlannedAction(
            action_id="a9",
            action_type="knowledge.add_sources",
            workspace=_reference(WorkspaceReferenceKind.active),
            source_object_ids=("object-url", "object-path"),
            depends_on=("a3",),
        ),
        WorkspaceAskPlannedAction(
            action_id="a10",
            action_type="workspace.ask",
            workspace=_reference(WorkspaceReferenceKind.active),
            question="What is indexed?",
            depends_on=("a3",),
        ),
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        objects=objects,
        actions=actions,
        response_mode="aggregate",
    )
    workspace_service = WorkspaceServiceFake()
    selection_service = SelectionServiceFake()
    candidate_service = CandidateServiceFake()
    attachment_service = AttachmentServiceFake()
    web_service = WebUrlServiceFake()
    local_service = LocalReferenceServiceFake()
    ask_service = AskServiceFake()
    clock_calls: list[datetime] = []

    def clock() -> datetime:
        value = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=len(clock_calls))
        clock_calls.append(value)
        return value

    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,  # type: ignore[arg-type]
        workspace_selection_service=selection_service,  # type: ignore[arg-type]
        source_candidate_service=candidate_service,
        attachment_intake_service=attachment_service,
        trusted_attachment_resolver=lambda attachment_id: f"upload:{attachment_id}",
        web_url_intake_service=web_service,
        local_reference_intake_service=local_service,
        ask_service=ask_service,  # type: ignore[arg-type]
        destructive_confirmation_codec=_CONFIRM_CODEC,
        execution_id_factory=lambda: "execution-1",
        clock=clock,
    )

    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="tenant-1",
            planning_request=request,
            interaction_plan=plan,
            execution_context=_context(),
        )
    )

    assert result.status is ConversationInteractionOverallStatus.COMPLETED
    assert result.tenant_id == "tenant-1"
    assert result.plan_version == "2"
    assert result.completed_at >= result.started_at
    assert [item.status for item in result.action_results] == [
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
        ConversationActionExecutionStatus.COMPLETED,
    ]
    assert selection_service.calls == ["w3"]
    assert ask_service.calls == ["What is indexed?"]
    assert web_service.calls == ["https://example.test/a"]
    assert local_service.calls == ["C:\\data"]
    assert result.active_workspace_id == "w3"
    assert result.created_resources[0].data["workspace_id"] == "w3"
    assert result.ask_runs[0].data["run_id"] == "run-1"
    assert [item.resolved_workspace_id for item in result.action_results] == [
        None,
        "w3",
        "w3",
        "w1",
        "w3",
        "w3",
        "w3",
        "w3",
        "w3",
        "w3",
    ]
    assert all(item.completed_at >= item.started_at for item in result.action_results)
    assert len(clock_calls) == 23
    assert result.started_at == clock_calls[0]
    assert result.completed_at == clock_calls[-1]


@pytest.mark.asyncio
async def test_executor_blocks_dependency_and_continues_independent_action() -> None:
    workspace_service = WorkspaceServiceFake()
    workspace_service.create_workspace = lambda **_: (_ for _ in ()).throw(
        RuntimeError("create failed")
    )
    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,  # type: ignore[arg-type]
        workspace_selection_service=SelectionServiceFake(),  # type: ignore[arg-type]
        execution_id_factory=lambda: "execution-2",
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceCreatePlannedAction(action_id="create", action_type="workspace.create", name="new"),
            WorkspaceListPlannedAction(
                action_id="dependent",
                action_type="workspace.list",
                depends_on=("create",),
            ),
            WorkspaceListPlannedAction(action_id="independent", action_type="workspace.list"),
        ),
        response_mode="aggregate",
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="tenant-1",
            planning_request=ConversationPlanningRequest(message_text="new"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )

    assert result.status is ConversationInteractionOverallStatus.PARTIALLY_COMPLETED
    assert result.action_results[0].status is ConversationActionExecutionStatus.FAILED
    assert (
        result.action_results[1].status
        is ConversationActionExecutionStatus.BLOCKED_DEPENDENCY
    )
    assert result.action_results[2].status is ConversationActionExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_preflight_context_mismatch_performs_no_mutation() -> None:
    workspace_service = WorkspaceServiceFake()
    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,  # type: ignore[arg-type]
        workspace_selection_service=SelectionServiceFake(),  # type: ignore[arg-type]
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(WorkspaceCreatePlannedAction(action_id="a1", action_type="workspace.create", name="new"),),
        response_mode="aggregate",
    )

    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="different-tenant",
            planning_request=ConversationPlanningRequest(message_text="new"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )

    assert result.status is ConversationInteractionOverallStatus.FAILED
    assert result.error is not None
    assert result.error.code == "conversation_execution_context_mismatch"
    assert not any(call.startswith("create:") for call in workspace_service.calls)


_STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_COMPLETED_AT = _STARTED_AT + timedelta(seconds=1)


def _artifact(artifact_type: str = "workspace.list") -> ConversationExecutionArtifact:
    return ConversationExecutionArtifact(artifact_type=artifact_type)


def _action_result(
    status: ConversationActionExecutionStatus,
    *,
    artifact: ConversationExecutionArtifact | None = None,
    error: ConversationExecutionError | None = None,
    action_id: str = "a1",
    started_at: datetime = _STARTED_AT,
    completed_at: datetime = _COMPLETED_AT,
) -> ConversationActionExecutionResult:
    return ConversationActionExecutionResult(
        action_id=action_id,
        action_type="workspace.list",
        status=status,
        artifact=artifact,
        error=error,
        started_at=started_at,
        completed_at=completed_at,
    )


@pytest.mark.parametrize(
    ("status", "artifact", "error"),
    [
        (ConversationActionExecutionStatus.COMPLETED, _artifact(), None),
        (
            ConversationActionExecutionStatus.FAILED,
            None,
            ConversationExecutionError(code="action_execution_failed", action_id="a1"),
        ),
        (
            ConversationActionExecutionStatus.BLOCKED_DEPENDENCY,
            None,
            ConversationExecutionError(code="blocked_dependency", action_id="a1"),
        ),
        (
            ConversationActionExecutionStatus.BLOCKED_CLARIFICATION,
            None,
            ConversationExecutionError(code="blocked_clarification", action_id="a1"),
        ),
        (ConversationActionExecutionStatus.SKIPPED, None, None),
    ],
)
def test_action_result_integrity_accepts_every_status(
    status: ConversationActionExecutionStatus,
    artifact: ConversationExecutionArtifact | None,
    error: ConversationExecutionError | None,
) -> None:
    result = _action_result(status, artifact=artifact, error=error)
    assert result.status is status


@pytest.mark.parametrize(
    "result_factory",
    [
        lambda: _action_result(ConversationActionExecutionStatus.COMPLETED),
        lambda: _action_result(
            ConversationActionExecutionStatus.COMPLETED,
            artifact=_artifact(),
            error=ConversationExecutionError(code="action_execution_failed", action_id="a1"),
        ),
        lambda: _action_result(ConversationActionExecutionStatus.FAILED),
        lambda: _action_result(
            ConversationActionExecutionStatus.FAILED,
            artifact=_artifact(),
            error=ConversationExecutionError(code="action_execution_failed", action_id="a1"),
        ),
        lambda: _action_result(
            ConversationActionExecutionStatus.BLOCKED_DEPENDENCY,
            error=ConversationExecutionError(code="action_execution_failed", action_id="a1"),
        ),
        lambda: _action_result(
            ConversationActionExecutionStatus.BLOCKED_CLARIFICATION,
            error=ConversationExecutionError(code="action_execution_failed", action_id="a1"),
        ),
        lambda: _action_result(
            ConversationActionExecutionStatus.FAILED,
            error=ConversationExecutionError(code="action_execution_failed", action_id="other"),
        ),
        lambda: _action_result(
            ConversationActionExecutionStatus.COMPLETED,
            artifact=_artifact(),
            started_at=_COMPLETED_AT,
            completed_at=_STARTED_AT,
        ),
        lambda: _action_result(
            ConversationActionExecutionStatus.COMPLETED,
            artifact=_artifact(),
            started_at=datetime(2026, 1, 1, 12, 0),
        ),
        lambda: _action_result(
            ConversationActionExecutionStatus.COMPLETED,
            artifact=_artifact(),
            completed_at=datetime(
                2026,
                1,
                1,
                14,
                0,
                tzinfo=timezone(timedelta(hours=2)),
            ),
        ),
    ],
)
def test_action_result_integrity_rejects_invalid_instances(result_factory: object) -> None:
    with pytest.raises(ValidationError):
        result_factory()  # type: ignore[operator]


def _aggregate(
    status: ConversationInteractionOverallStatus,
    action_results: tuple[ConversationActionExecutionResult, ...] = (),
    *,
    created_resources: tuple[ConversationExecutionArtifact, ...] = (),
    ask_runs: tuple[ConversationExecutionArtifact, ...] = (),
    error: ConversationExecutionError | None = None,
    started_at: datetime = _STARTED_AT,
    completed_at: datetime = _COMPLETED_AT,
) -> ConversationInteractionExecutionResult:
    return ConversationInteractionExecutionResult(
        execution_id="execution-1",
        tenant_id="tenant-1",
        plan_version="2",
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        action_results=action_results,
        created_resources=created_resources,
        ask_runs=ask_runs,
        error=error,
    )


def test_aggregate_integrity_accepts_canonical_outcomes() -> None:
    completed = _action_result(
        ConversationActionExecutionStatus.COMPLETED,
        artifact=_artifact(),
    )
    failed = _action_result(
        ConversationActionExecutionStatus.FAILED,
        action_id="a2",
        error=ConversationExecutionError(code="action_execution_failed", action_id="a2"),
    )
    blocked = _action_result(
        ConversationActionExecutionStatus.BLOCKED_CLARIFICATION,
        action_id="a2",
        error=ConversationExecutionError(code="blocked_clarification", action_id="a2"),
    )

    assert _aggregate(ConversationInteractionOverallStatus.COMPLETED, (completed,))
    assert _aggregate(
        ConversationInteractionOverallStatus.PARTIALLY_COMPLETED,
        (completed, failed),
    )
    assert _aggregate(
        ConversationInteractionOverallStatus.CLARIFICATION_REQUIRED,
        (completed, blocked),
    )
    assert _aggregate(ConversationInteractionOverallStatus.FAILED, (failed,))
    assert _aggregate(
        ConversationInteractionOverallStatus.FAILED,
        error=ConversationExecutionError(code="interaction_plan_invalid"),
    )


@pytest.mark.parametrize(
    "aggregate_factory",
    [
        lambda: _aggregate(
            ConversationInteractionOverallStatus.COMPLETED,
            (
                _action_result(
                    ConversationActionExecutionStatus.COMPLETED,
                    artifact=_artifact(),
                ),
                _action_result(
                    ConversationActionExecutionStatus.COMPLETED,
                    artifact=_artifact("workspace.create"),
                ),
            ),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.COMPLETED,
            (
                _action_result(
                    ConversationActionExecutionStatus.FAILED,
                    error=ConversationExecutionError(
                        code="action_execution_failed",
                        action_id="a1",
                    ),
                ),
            ),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.COMPLETED,
            (_action_result(ConversationActionExecutionStatus.COMPLETED, artifact=_artifact()),),
            error=ConversationExecutionError(code="interaction_plan_invalid"),
        ),
        lambda: _aggregate(ConversationInteractionOverallStatus.FAILED),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.FAILED,
            error=ConversationExecutionError(
                code="interaction_plan_invalid",
                action_id="a1",
            ),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.CLARIFICATION_REQUIRED,
            (_action_result(ConversationActionExecutionStatus.COMPLETED, artifact=_artifact()),),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.PARTIALLY_COMPLETED,
            (_action_result(ConversationActionExecutionStatus.COMPLETED, artifact=_artifact()),),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.COMPLETED,
            (_action_result(ConversationActionExecutionStatus.COMPLETED, artifact=_artifact()),),
            created_resources=(_artifact("workspace.create"),),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.COMPLETED,
            (_action_result(ConversationActionExecutionStatus.COMPLETED, artifact=_artifact()),),
            ask_runs=(_artifact("workspace.ask"),),
        ),
        lambda: _aggregate(
            ConversationInteractionOverallStatus.COMPLETED,
            (_action_result(ConversationActionExecutionStatus.COMPLETED, artifact=_artifact()),),
            started_at=_COMPLETED_AT,
            completed_at=_STARTED_AT,
        ),
    ],
)
def test_aggregate_integrity_rejects_invalid_instances(aggregate_factory: object) -> None:
    with pytest.raises(ValidationError):
        aggregate_factory()  # type: ignore[operator]


@pytest.mark.asyncio
async def test_preflight_result_preserves_identity_and_uses_injected_clock() -> None:
    now = datetime(2026, 2, 1, tzinfo=UTC)
    clock_values = iter((now, now + timedelta(seconds=1)))
    executor = ConversationInteractionExecutor(
        workspace_service=WorkspaceServiceFake(),  # type: ignore[arg-type]
        workspace_selection_service=SelectionServiceFake(),  # type: ignore[arg-type]
        clock=lambda: next(clock_values),
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceCreatePlannedAction(action_id="a1", action_type="workspace.create", name="new"),
        ),
        response_mode="aggregate",
    )

    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="different-tenant",
            planning_request=ConversationPlanningRequest(message_text="new"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )

    assert result.status is ConversationInteractionOverallStatus.FAILED
    assert result.action_results == ()
    assert result.tenant_id == "tenant-1"
    assert result.plan_version == "2"
    assert result.started_at == now
    assert result.completed_at == now + timedelta(seconds=1)
    assert result.started_at.tzinfo is not None
    assert result.started_at.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_failed_workspace_resolution_does_not_publish_workspace_id() -> None:
    executor = ConversationInteractionExecutor(
        workspace_service=WorkspaceServiceFake(),  # type: ignore[arg-type]
        workspace_selection_service=SelectionServiceFake(),  # type: ignore[arg-type]
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceDeletePlannedAction(
                action_id="delete",
                action_type="workspace.delete",
                workspace=_reference(WorkspaceReferenceKind.name, "missing"),
            ),
        ),
        response_mode="aggregate",
    )

    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="tenant-1",
            planning_request=ConversationPlanningRequest(message_text="delete missing"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )

    assert result.action_results[0].status is ConversationActionExecutionStatus.FAILED
    assert result.action_results[0].resolved_workspace_id is None


@pytest.mark.asyncio
async def test_clarification_blocks_only_declared_branch() -> None:
    workspace_service = WorkspaceServiceFake()
    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,  # type: ignore[arg-type]
        workspace_selection_service=SelectionServiceFake(),  # type: ignore[arg-type]
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceCreatePlannedAction(action_id="blocked", action_type="workspace.create", name="new"),
            WorkspaceListPlannedAction(
                action_id="dependent",
                action_type="workspace.list",
                depends_on=("blocked",),
            ),
            WorkspaceListPlannedAction(action_id="safe", action_type="workspace.list"),
        ),
        clarifications=(
            {
                "clarification_id": "clarify-1",
                "question": "Which workspace?",
                "blocks_action_ids": ("blocked",),
            },
        ),
        response_mode="aggregate",
    )

    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="tenant-1",
            planning_request=ConversationPlanningRequest(message_text="new"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )

    assert result.status is ConversationInteractionOverallStatus.CLARIFICATION_REQUIRED
    assert (
        result.action_results[0].status
        is ConversationActionExecutionStatus.BLOCKED_CLARIFICATION
    )
    assert (
        result.action_results[1].status
        is ConversationActionExecutionStatus.BLOCKED_DEPENDENCY
    )
    assert result.action_results[2].status is ConversationActionExecutionStatus.COMPLETED
    assert not any(call.startswith("create:") for call in workspace_service.calls)
