# © Artur Czarnecki. All rights reserved.

"""Deterministic execution of canonical ConversationInteractionPlan V2 plans."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionResult,
    ConversationExecutionArtifact,
    ConversationExecutionClarification,
    ConversationExecutionError,
    ConversationInteractionExecutionCommand,
    ConversationInteractionExecutionResult,
    ConversationInteractionOverallStatus,
    ConversationActionExecutionStatus,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ExtractedObject,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    LocalFileReferenceExtractedObject,
    PlannedAction,
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
)
from local_workspace_application.conversation.interaction_planner import (
    PlanRequestValidationError,
    validate_plan_against_request,
)
from local_workspace_application.conversation.interaction_reference_resolver import (
    ConversationInteractionReferenceResolver,
    ConversationReferenceResolutionError,
)
from local_workspace_application.workspaces.ask_service import WorkspaceAskService
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationProductCapability,
)
from local_workspace_application.workspaces.conversation_workspace_selection_service import (
    ConversationWorkspaceSelectionError,
    ConversationWorkspaceSelectionService,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.source_candidates import (
    SourceCandidateIntakeService,
)


class TrustedAttachmentResolver(Protocol):
    def __call__(self, attachment_id: str) -> object | None: ...


class CandidateIntakeService(Protocol):
    def list_candidates(self, *, tenant_id: str, workspace_id: str) -> Sequence[object]: ...

    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        candidate_id: str,
        idempotency_key: str,
    ) -> object: ...


class AttachmentIntakeService(Protocol):
    def accept_many(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        uploads: Sequence[object],
    ) -> object: ...


class WebUrlIntakeServiceProtocol(Protocol):
    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        raw_url: str,
        idempotency_key: str,
    ) -> object: ...


class LocalReferenceIntakeService(Protocol):
    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        reference: LocalFileReferenceExtractedObject,
        idempotency_key: str,
    ) -> object: ...


_ACTION_CAPABILITIES: dict[str, ConversationProductCapability] = {
    "workspace.list": ConversationProductCapability.WORKSPACE_DISCOVERY,
    "workspace.create": ConversationProductCapability.WORKSPACE_ADMINISTRATION,
    "workspace.activate": ConversationProductCapability.WORKSPACE_SELECTION,
    "workspace.delete": ConversationProductCapability.WORKSPACE_ADMINISTRATION,
    "source.list": ConversationProductCapability.SOURCE_DISCOVERY,
    "source_candidate.list": ConversationProductCapability.SOURCE_DISCOVERY,
    "source_candidate.attach": ConversationProductCapability.SOURCE_INTAKE,
    "knowledge.add_attachments": ConversationProductCapability.ATTACHMENT_INTAKE,
    "knowledge.add_sources": ConversationProductCapability.SOURCE_INTAKE,
    "workspace.ask": ConversationProductCapability.READ_ONLY_ASK,
}

_SAFE_ERROR_CODES = frozenset(
    {
        "workspace_not_found",
        "workspace_reference_ambiguous",
        "active_workspace_required",
        "workspace_selection_personal_context_required",
        "workspace_selection_not_allowed",
        "workspace_selection_conflict",
        "conversation_context_identity_mismatch",
        "conversation_context_storage_unavailable",
        "source_candidate_not_found",
        "source_candidate_ambiguous",
        "source_candidate_unavailable",
        "attachment_not_found",
        "source_object_not_found",
        "source_reference_unsupported",
        "conversation_capability_not_allowed",
        "interaction_action_unsupported",
        "action_execution_failed",
        "blocked_dependency",
        "blocked_clarification",
        "interaction_plan_invalid",
        "dependency_cycle",
        "unknown_action_dependency",
        "unknown_extracted_object",
        "unknown_attachment",
        "evidence_not_grounded",
        "conversation_execution_context_mismatch",
    }
)

_SELECTION_ERROR_CODES = frozenset(
    {
        "workspace_selection_personal_context_required",
        "workspace_selection_not_allowed",
        "workspace_not_found",
        "workspace_selection_conflict",
        "conversation_context_identity_mismatch",
        "conversation_context_storage_unavailable",
    }
)


class _PreflightFailure(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ConversationInteractionExecutor:
    """Sequential, fail-closed executor with injected application boundaries."""

    def __init__(
        self,
        *,
        workspace_service: ManagedWorkspaceService,
        workspace_selection_service: ConversationWorkspaceSelectionService,
        source_candidate_service: CandidateIntakeService | SourceCandidateIntakeService | None = None,
        attachment_intake_service: AttachmentIntakeService | None = None,
        trusted_attachment_resolver: TrustedAttachmentResolver | None = None,
        web_url_intake_service: WebUrlIntakeServiceProtocol | None = None,
        local_reference_intake_service: LocalReferenceIntakeService | None = None,
        ask_service: WorkspaceAskService | None = None,
        execution_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._workspace_service = workspace_service
        self._workspace_selection_service = workspace_selection_service
        self._source_candidate_service = source_candidate_service
        self._attachment_intake_service = attachment_intake_service
        self._trusted_attachment_resolver = trusted_attachment_resolver
        self._web_url_intake_service = web_url_intake_service
        self._local_reference_intake_service = local_reference_intake_service
        self._ask_service = ask_service
        self._execution_id_factory = execution_id_factory or (lambda: str(uuid.uuid4()))
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        command: ConversationInteractionExecutionCommand,
    ) -> ConversationInteractionExecutionResult:
        started_at = self._utc_now()
        execution_id = command.execution_id or self._execution_id_factory()
        plan = command.interaction_plan
        try:
            order, clarification_map = self._preflight(command)
        except _PreflightFailure as exc:
            return self._preflight_result(
                execution_id=execution_id,
                plan=plan,
                tenant_id=command.execution_context.tenant_id,
                started_at=started_at,
                completed_at=self._utc_now(),
                code=exc.code,
            )

        resolver = ConversationInteractionReferenceResolver(
            planning_request=command.planning_request,
            execution_context=command.execution_context,
            workspace_service=self._workspace_service,
        )
        results_by_id: dict[str, ConversationActionExecutionResult] = {}
        created_workspace_ids: dict[str, str] = {}
        resolved_workspace_ids: dict[str, str] = {}
        for action_index in order:
            action = plan.actions[action_index]
            result = await self._execute_one(
                action=action,
                command=command,
                resolver=resolver,
                results_by_id=results_by_id,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
                clarification_ids=clarification_map.get(action.action_id, ()),
                execution_id=execution_id,
            )
            results_by_id[action.action_id] = result

        action_results = tuple(
            results_by_id[action.action_id] for action in plan.actions
        )
        clarifications = tuple(
            ConversationExecutionClarification(
                clarification_id=item.clarification_id,
                question=item.question,
                blocks_action_ids=item.blocks_action_ids,
            )
            for item in plan.clarifications
        )
        artifacts = tuple(
            item.artifact for item in action_results if item.artifact is not None
        )
        created_resources = tuple(
            item.artifact
            for action, item in zip(plan.actions, action_results, strict=True)
            if action.action_type == "workspace.create" and item.artifact is not None
        )
        ask_runs = tuple(
            item.artifact
            for action, item in zip(plan.actions, action_results, strict=True)
            if action.action_type == "workspace.ask" and item.artifact is not None
        )
        completed_at = self._utc_now()
        return ConversationInteractionExecutionResult(
            execution_id=execution_id,
            tenant_id=command.execution_context.tenant_id,
            plan_version=plan.plan_version,
            started_at=started_at,
            completed_at=completed_at,
            status=self._overall_status(action_results),
            action_results=action_results,
            clarifications=clarifications,
            active_workspace_id=resolver.current_active_workspace_id,
            created_resources=created_resources,
            ask_runs=ask_runs,
            response_data=artifacts,
        )

    def _preflight(
        self,
        command: ConversationInteractionExecutionCommand,
    ) -> tuple[tuple[int, ...], dict[str, tuple[str, ...]]]:
        if command.tenant_id != command.execution_context.tenant_id:
            raise _PreflightFailure("conversation_execution_context_mismatch")

        plan = command.interaction_plan
        action_ids = [action.action_id for action in plan.actions]
        if len(action_ids) != len(set(action_ids)):
            raise _PreflightFailure("interaction_plan_invalid")
        action_id_set = set(action_ids)
        object_ids = [obj.object_id for obj in plan.objects]
        if len(object_ids) != len(set(object_ids)):
            raise _PreflightFailure("interaction_plan_invalid")
        object_id_set = set(object_ids)
        for action in plan.actions:
            for dependency in action.depends_on:
                if dependency not in action_id_set:
                    raise _PreflightFailure("unknown_action_dependency")
            if isinstance(action, KnowledgeAddSourcesPlannedAction):
                for object_id in action.source_object_ids:
                    if object_id not in object_id_set:
                        raise _PreflightFailure("unknown_extracted_object")
            if isinstance(action, KnowledgeAddAttachmentsPlannedAction):
                allowed = {
                    attachment.attachment_id
                    for attachment in command.planning_request.attachments
                }
                if any(item not in allowed for item in action.attachment_ids):
                    raise _PreflightFailure("unknown_attachment")
            if any(
                evidence_id
                not in {
                    attachment.attachment_id
                    for attachment in command.planning_request.attachments
                }
                for evidence_id in action.evidence_attachment_ids
            ):
                raise _PreflightFailure("unknown_attachment")

        clarification_map: dict[str, list[str]] = {}
        for clarification in plan.clarifications:
            for action_id in clarification.blocks_action_ids:
                if action_id not in action_id_set:
                    raise _PreflightFailure("interaction_plan_invalid")
                clarification_map.setdefault(action_id, []).append(
                    clarification.clarification_id
                )

        try:
            validate_plan_against_request(plan, command.planning_request)
        except PlanRequestValidationError as exc:
            message = str(exc)
            if "attachment" in message:
                code = "unknown_attachment"
            elif "object" in message:
                code = "unknown_extracted_object"
            elif "evidence" in message or "quote" in message:
                code = "evidence_not_grounded"
            else:
                code = "interaction_plan_invalid"
            raise _PreflightFailure(code) from exc
        except (TypeError, ValueError) as exc:
            raise _PreflightFailure("interaction_plan_invalid") from exc

        dependencies = {
            action.action_id: set(action.depends_on) for action in plan.actions
        }
        order: list[int] = []
        remaining = set(action_ids)
        while remaining:
            ready = [
                index
                for index, action in enumerate(plan.actions)
                if action.action_id in remaining
                and dependencies[action.action_id].isdisjoint(remaining)
            ]
            if not ready:
                raise _PreflightFailure("dependency_cycle")
            index = ready[0]
            remaining.remove(plan.actions[index].action_id)
            order.append(index)
        return tuple(order), {
            action_id: tuple(ids) for action_id, ids in clarification_map.items()
        }

    async def _execute_one(
        self,
        *,
        action: PlannedAction,
        command: ConversationInteractionExecutionCommand,
        resolver: ConversationInteractionReferenceResolver,
        results_by_id: Mapping[str, ConversationActionExecutionResult],
        created_workspace_ids: dict[str, str],
        resolved_workspace_ids: dict[str, str],
        clarification_ids: tuple[str, ...],
        execution_id: str,
    ) -> ConversationActionExecutionResult:
        started_at = self._utc_now()
        if clarification_ids:
            return self._result(
                action,
                ConversationActionExecutionStatus.BLOCKED_CLARIFICATION,
                started_at=started_at,
                error_code="blocked_clarification",
            )
        if any(
            results_by_id[dependency].status
            is not ConversationActionExecutionStatus.COMPLETED
            for dependency in action.depends_on
        ):
            return self._result(
                action,
                ConversationActionExecutionStatus.BLOCKED_DEPENDENCY,
                started_at=started_at,
                error_code="blocked_dependency",
            )
        capability = _ACTION_CAPABILITIES.get(action.action_type)
        if capability is None:
            return self._result(
                action,
                ConversationActionExecutionStatus.FAILED,
                started_at=started_at,
                error_code="interaction_action_unsupported",
            )
        if capability not in command.execution_context.allowed_product_capabilities:
            return self._result(
                action,
                ConversationActionExecutionStatus.FAILED,
                started_at=started_at,
                error_code="conversation_capability_not_allowed",
            )

        try:
            artifact = await self._dispatch(
                action=action,
                command=command,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
                execution_id=execution_id,
            )
        except ConversationReferenceResolutionError as exc:
            return self._result(
                action,
                ConversationActionExecutionStatus.FAILED,
                started_at=started_at,
                resolved_workspace_id=resolved_workspace_ids.get(action.action_id),
                error_code=exc.code,
            )
        except ConversationWorkspaceSelectionError as exc:
            code = exc.error_code if exc.error_code in _SELECTION_ERROR_CODES else "action_execution_failed"
            return self._result(
                action,
                ConversationActionExecutionStatus.FAILED,
                started_at=started_at,
                resolved_workspace_id=resolved_workspace_ids.get(action.action_id),
                error_code=code,
            )
        except Exception as exc:  # noqa: BLE001 - every action has a safe boundary
            code = _safe_exception_code(exc)
            return self._result(
                action,
                ConversationActionExecutionStatus.FAILED,
                started_at=started_at,
                resolved_workspace_id=resolved_workspace_ids.get(action.action_id),
                error_code=code,
            )

        if action.action_type == "workspace.create":
            workspace_id = str(artifact.data.get("workspace_id", "")).strip()
            if workspace_id:
                created_workspace_ids[action.action_id] = workspace_id
                resolved_workspace_ids[action.action_id] = workspace_id
        return self._result(
            action,
            ConversationActionExecutionStatus.COMPLETED,
            started_at=started_at,
            artifact=artifact,
            resolved_workspace_id=resolved_workspace_ids.get(action.action_id),
        )

    async def _dispatch(
        self,
        *,
        action: PlannedAction,
        command: ConversationInteractionExecutionCommand,
        resolver: ConversationInteractionReferenceResolver,
        created_workspace_ids: dict[str, str],
        resolved_workspace_ids: dict[str, str],
        execution_id: str,
    ) -> ConversationExecutionArtifact:
        tenant_id = command.execution_context.tenant_id
        if isinstance(action, WorkspaceListPlannedAction):
            workspaces = self._workspace_service.list_workspaces(tenant_id=tenant_id)
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspaces": [
                        {
                            "workspace_id": str(item.workspace_id),
                            "name": str(item.name),
                            "is_active": str(item.workspace_id)
                            == resolver.current_active_workspace_id,
                        }
                        for item in workspaces
                    ]
                },
            )
        if isinstance(action, WorkspaceCreatePlannedAction):
            workspace = self._workspace_service.create_workspace(
                tenant_id=tenant_id,
                name=action.name,
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": str(workspace.workspace_id),
                    "name": str(workspace.name),
                },
            )
        if isinstance(action, WorkspaceActivatePlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            selection = self._workspace_selection_service.select_personal_workspace(
                execution_context=command.execution_context,
                workspace_id=workspace_id,
            )
            selected_workspace_id = str(selection.selected_workspace_id)
            resolved_workspace_ids[action.action_id] = selected_workspace_id
            resolver.set_active_workspace(selected_workspace_id)
            selected_workspace = self._workspace_service.get_workspace(
                tenant_id=tenant_id,
                workspace_id=selected_workspace_id,
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": selected_workspace_id,
                    "name": _safe_attr(selected_workspace, "name"),
                    "previous_workspace_id": selection.previous_workspace_id,
                    "configuration_version": selection.configuration_version,
                    "changed": selection.changed,
                    "active": True,
                },
            )
        if isinstance(action, WorkspaceDeletePlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            deleted = self._workspace_service.delete_workspace(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if not deleted:
                raise ConversationReferenceResolutionError("workspace_not_found")
            if resolver.current_active_workspace_id == workspace_id:
                resolver.clear_active_workspace()
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={"workspace_id": workspace_id, "deleted": True},
            )
        if isinstance(action, SourceListPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            sources = self._workspace_service.list_sources(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if sources is None:
                raise ConversationReferenceResolutionError("workspace_not_found")
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "sources": [
                        {
                            "source_id": str(item.source_id),
                            "source_type": str(getattr(item.source_type, "value", item.source_type)),
                            "status": str(getattr(item.status, "value", item.status)),
                        }
                        for item in sources
                    ],
                },
            )
        if isinstance(action, SourceCandidateListPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            service = self._require_candidate_service()
            candidates = service.list_candidates(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "candidates": [_safe_candidate(item) for item in candidates],
                },
            )
        if isinstance(action, SourceCandidateAttachPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            candidate = self._resolve_candidate_snapshot(
                command.planning_request.available_source_candidates,
                action,
            )
            service = self._require_candidate_service()
            candidate_id = str(_safe_attr(candidate, "candidate_id") or "")
            candidate_label = str(_safe_attr(candidate, "label") or "")
            acceptance = service.accept(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                candidate_id=candidate_id,
                idempotency_key=f"{execution_id}:{action.action_id}",
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "candidate_id": candidate_id,
                    "label": candidate_label,
                    "source_id": _safe_attr(acceptance, "source_id"),
                    "operation_id": _safe_attr(acceptance, "operation_id"),
                    "status": _safe_attr(acceptance, "status"),
                },
            )
        if isinstance(action, KnowledgeAddAttachmentsPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            if self._trusted_attachment_resolver is None:
                raise RuntimeError("attachment_not_found")
            uploads: list[object] = []
            for attachment_id in action.attachment_ids:
                upload = self._trusted_attachment_resolver(attachment_id)
                if upload is None:
                    raise RuntimeError("attachment_not_found")
                uploads.append(upload)
            if self._attachment_intake_service is None:
                raise RuntimeError("action_execution_failed")
            batch = self._attachment_intake_service.accept_many(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                idempotency_key=f"{execution_id}:{action.action_id}",
                uploads=uploads,
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "batch_id": _safe_attr(batch, "batch_id"),
                    "status": _safe_attr(batch, "status"),
                    "attachments": list(action.attachment_ids),
                },
            )
        if isinstance(action, KnowledgeAddSourcesPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            objects = {item.object_id: item for item in command.interaction_plan.objects}
            source_results = []
            for object_id in action.source_object_ids:
                source = objects.get(object_id)
                if source is None:
                    raise RuntimeError("source_object_not_found")
                source_results.append(
                    await self._add_source(
                        source=source,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        idempotency_key=f"{execution_id}:{action.action_id}:{object_id}",
                    )
                )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={"workspace_id": workspace_id, "sources": source_results},
            )
        if isinstance(action, WorkspaceAskPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            if self._ask_service is None:
                raise RuntimeError("action_execution_failed")
            run = await self._ask_service.ask(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                question=action.question,
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "run_id": _safe_attr(run, "run_id"),
                    "status": _safe_attr(run, "status"),
                    "answer": getattr(run, "answer", None),
                    "citations": _safe_json(getattr(run, "citations", [])),
                },
            )
        raise RuntimeError("interaction_action_unsupported")

    @staticmethod
    def _resolve_workspace_id(
        *,
        action_id: str,
        reference: WorkspaceReference,
        resolver: ConversationInteractionReferenceResolver,
        created_workspace_ids: dict[str, str],
        resolved_workspace_ids: dict[str, str],
    ) -> str:
        resolved = resolver.resolve_workspace(
            reference,
            created_workspace_ids=created_workspace_ids,
        )
        workspace_id = str(resolved.workspace_id)
        resolved_workspace_ids[action_id] = workspace_id
        return workspace_id

    async def _add_source(
        self,
        *,
        source: ExtractedObject,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if isinstance(source, WebUrlExtractedObject):
            if self._web_url_intake_service is None:
                raise RuntimeError("source_reference_unsupported")
            accepted = await _maybe_await(
                self._web_url_intake_service.accept(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    raw_url=source.value,
                    idempotency_key=idempotency_key,
                )
            )
            return {
                "object_id": source.object_id,
                "source_id": _safe_attr(accepted, "source_id"),
                "operation_id": _safe_attr(accepted, "operation_id"),
                "status": _safe_attr(accepted, "status"),
            }
        if isinstance(source, LocalFileReferenceExtractedObject):
            if self._local_reference_intake_service is not None:
                accepted = await _maybe_await(
                    self._local_reference_intake_service.accept(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        reference=source,
                        idempotency_key=idempotency_key,
                    )
                )
            elif source.reference_kind == "folder":
                accepted = self._workspace_service.register_local_folder_source(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    path=source.value,
                    recursive=True,
                )
            else:
                raise RuntimeError("source_reference_unsupported")
            return {
                "object_id": source.object_id,
                "source_id": _safe_attr(accepted, "source_id"),
                "operation_id": _safe_attr(accepted, "operation_id"),
                "status": _safe_attr(accepted, "status"),
            }
        raise RuntimeError("source_reference_unsupported")

    def _resolve_candidate_snapshot(
        self,
        candidates: Sequence[object],
        action: SourceCandidateAttachPlannedAction,
    ) -> object:
        reference = action.candidate_reference.strip().casefold()
        if action.candidate_reference_kind == "ordinal":
            if not reference.isdigit():
                raise RuntimeError("source_candidate_not_found")
            index = int(reference) - 1
            if index < 0 or index >= len(candidates):
                raise RuntimeError("source_candidate_not_found")
            candidate = candidates[index]
            if not bool(getattr(candidate, "available", False)):
                raise RuntimeError("source_candidate_unavailable")
            return candidate
        matches = [
            candidate
            for candidate in candidates
            if str(getattr(candidate, "label", "")).strip().casefold() == reference
        ]
        if not matches:
            raise RuntimeError("source_candidate_not_found")
        if len(matches) > 1:
            raise RuntimeError("source_candidate_ambiguous")
        if not bool(getattr(matches[0], "available", False)):
            raise RuntimeError("source_candidate_unavailable")
        return matches[0]

    def _require_candidate_service(self) -> CandidateIntakeService:
        if self._source_candidate_service is None:
            raise RuntimeError("source_candidate_unavailable")
        return self._source_candidate_service

    def _utc_now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise ValueError("clock must return a datetime")
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("clock must return a timezone-aware UTC datetime")
        return value

    def _result(
        self,
        action: PlannedAction,
        status: ConversationActionExecutionStatus,
        *,
        started_at: datetime,
        artifact: ConversationExecutionArtifact | None = None,
        error_code: str | None = None,
        resolved_workspace_id: str | None = None,
    ) -> ConversationActionExecutionResult:
        completed_at = self._utc_now()
        return ConversationActionExecutionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            status=status,
            artifact=artifact,
            error=(
                ConversationExecutionError(code=error_code, action_id=action.action_id)
                if error_code is not None
                else None
            ),
            resolved_workspace_id=resolved_workspace_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    @staticmethod
    def _overall_status(
        results: Sequence[ConversationActionExecutionResult],
    ) -> ConversationInteractionOverallStatus:
        statuses = {item.status for item in results}
        has_completed = ConversationActionExecutionStatus.COMPLETED in statuses
        has_failed = ConversationActionExecutionStatus.FAILED in statuses
        has_clarification = (
            ConversationActionExecutionStatus.BLOCKED_CLARIFICATION in statuses
        )
        has_blocked = has_clarification or (
            ConversationActionExecutionStatus.BLOCKED_DEPENDENCY in statuses
        )
        if has_failed:
            if not has_completed:
                return ConversationInteractionOverallStatus.FAILED
            return ConversationInteractionOverallStatus.PARTIALLY_COMPLETED
        if has_clarification:
            return ConversationInteractionOverallStatus.CLARIFICATION_REQUIRED
        if has_blocked or ConversationActionExecutionStatus.SKIPPED in statuses:
            return ConversationInteractionOverallStatus.PARTIALLY_COMPLETED
        return ConversationInteractionOverallStatus.COMPLETED

    @staticmethod
    def _preflight_result(
        *,
        execution_id: str,
        plan: ConversationInteractionPlan,
        tenant_id: str,
        started_at: datetime,
        completed_at: datetime,
        code: str,
    ) -> ConversationInteractionExecutionResult:
        return ConversationInteractionExecutionResult(
            execution_id=execution_id,
            tenant_id=tenant_id,
            plan_version=plan.plan_version,
            started_at=started_at,
            completed_at=completed_at,
            status=ConversationInteractionOverallStatus.FAILED,
            clarifications=tuple(
                ConversationExecutionClarification(
                    clarification_id=item.clarification_id,
                    question=item.question,
                    blocks_action_ids=item.blocks_action_ids,
                )
                for item in plan.clarifications
            ),
            error=ConversationExecutionError(code=code),
        )


def _safe_exception_code(exc: Exception) -> str:
    normalized_aliases = {
        "candidate_not_found": "source_candidate_not_found",
        "source_candidate_configuration_invalid": "source_candidate_unavailable",
        "source_candidate_configuration_changed": "source_candidate_unavailable",
    }
    for attribute in ("error_code", "code", "reason"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value in _SAFE_ERROR_CODES:
            return value
    value = str(exc).strip()
    if value in normalized_aliases:
        return normalized_aliases[value]
    return value if value in _SAFE_ERROR_CODES else "action_execution_failed"


def _safe_attr(value: object, name: str) -> object | None:
    item = getattr(value, name, None)
    enum_value = getattr(item, "value", None)
    if enum_value is not None:
        item = enum_value
    return item if isinstance(item, (str, int, float, bool)) or item is None else str(item)


def _safe_candidate(candidate: object) -> dict[str, object]:
    return {
        "candidate_id": _safe_attr(candidate, "candidate_id"),
        "label": _safe_attr(candidate, "label"),
        "source_type": _safe_attr(candidate, "source_type"),
        "available": bool(getattr(candidate, "available", False)),
    }


def _safe_json(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _safe_json(model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _safe_json(enum_value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value
