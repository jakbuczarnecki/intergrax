# © Artur Czarnecki. All rights reserved.

"""Deterministic execution of canonical ConversationInteractionPlan V2 plans."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Protocol, cast

from pydantic import BaseModel

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
    ConversationPlanningSourceCandidate,
    ExtractedObject,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    KnowledgeCapabilitiesListPlannedAction,
    KnowledgeConnectionsListPlannedAction,
    KnowledgeResourcesListPlannedAction,
    LocalFileReferenceExtractedObject,
    PlannedAction,
    SourceCandidateAttachPlannedAction,
    SourceCandidateListPlannedAction,
    SourceListPlannedAction,
    WebUrlExtractedObject,
    WorkspaceActivatePlannedAction,
    WorkspaceAskPlannedAction,
    CitationInspectPlannedAction,
    DestructiveActionConfirmPlannedAction,
    KnowledgeInventoryFilter,
    KnowledgeInventoryListPlannedAction,
    KnowledgeOperationExecutePlannedAction,
    KnowledgeOperationKind,
    KnowledgeTargetReferenceKind,
    TenantConnectionProvidersListPlannedAction,
    TenantConnectionConnectionsListPlannedAction,
    TenantConnectionInspectPlannedAction,
    TenantConnectionBeginAuthorizationPlannedAction,
    TenantConnectionCompleteManualAuthorizationPlannedAction,
    TenantConnectionReconnectPlannedAction,
    TenantConnectionRevokePlannedAction,
    WorkspaceCreatePlannedAction,
    WorkspaceDeletePlannedAction,
    WorkspaceListPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_memory_policy import (
    interaction_plan_requires_credential_memory_redaction,
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
from local_workspace_application.workspaces.knowledge_ask_scope_models import KnowledgeAskScopeV1
from local_workspace_application.workspaces.knowledge_target_resolution import (
    resolve_knowledge_target,
)
from local_workspace_application.workspaces.conversation_citation_context_service import (
    ConversationCitationContextError,
    ConversationCitationContextService,
)
from local_workspace_application.workspaces.document_inspect_service import (
    DocumentInspectError,
    DocumentInspectService,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationProductCapability,
)
from local_workspace_application.workspaces.conversation_workspace_selection_service import (
    ConversationWorkspaceSelectionError,
    ConversationWorkspaceSelectionService,
)
from local_workspace_application.workspaces.models import (
    IntakeBatch,
    Workspace,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.knowledge_plugin_configuration_service import (
    KnowledgeCapabilitySummaryV1,
    KnowledgeConnectionSummaryV1,
    KnowledgeRemoteResourcePageV1,
    KnowledgePluginConfigurationService,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInspectionService,
    KnowledgeInventoryError,
    KnowledgeInventoryItemV1,
    KnowledgeInventorySummaryV1,
    KnowledgeInventoryV1,
    KnowledgeOperationCommandV1,
    KnowledgeOperationError,
    KnowledgeOperationV1,
    KnowledgeOperationsService,
)
from local_workspace_application.workspaces.knowledge_administration_service import (
    Sha256KnowledgeAdministrationIdempotencyKeyFactory,
)
from local_workspace_application.workspaces.destructive_action_confirmation import (
    DestructiveActionConfirmationError,
    DestructiveActionConfirmationV1,
    DestructiveActionKindV1,
    HmacDestructiveActionConfirmationCodec,
    knowledge_detach_action_kind,
    knowledge_operation_action_kind,
    tenant_connection_revoke_action_kind,
)
from local_workspace_application.workspaces.conversation_connection_auth_context_service import (
    ConversationConnectionAuthContextError,
    ConversationConnectionAuthContextService,
    TenantConnectionConversationConfig,
    parse_manual_credential_payload,
    safe_connection_payload,
)
from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    THREAD_MEMORY_CREDENTIAL_REDACTION,
)
from local_workspace_application.workspaces.tenant_connection_product_errors import (
    TenantConnectionProductError,
)
from local_workspace_application.workspaces.source_candidates import (
    SourceCandidateAcceptance,
    SourceCandidateIntakeService,
    SourceCandidateSummary,
)
from local_workspace_application.workspaces.web_url_ingestion import WebUrlAcceptance


class TrustedAttachmentResolver(Protocol):
    def __call__(self, attachment_id: str) -> object | None: ...


class CandidateIntakeService(Protocol):
    def list_candidates(
        self, *, tenant_id: str, workspace_id: str
    ) -> Sequence[SourceCandidateSummary]: ...

    def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        candidate_id: str,
        idempotency_key: str,
    ) -> SourceCandidateAcceptance: ...


class AttachmentIntakeService(Protocol):
    def accept_many(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        uploads: Sequence[object],
    ) -> IntakeBatch: ...


class WebUrlIntakeServiceProtocol(Protocol):
    async def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        raw_url: str,
        idempotency_key: str,
    ) -> WebUrlAcceptance: ...


class LocalReferenceIntakeService(Protocol):
    async def accept(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        reference: LocalFileReferenceExtractedObject,
        idempotency_key: str,
    ) -> WebUrlAcceptance: ...


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
    "citation.inspect": ConversationProductCapability.READ_ONLY_ASK,
    "knowledge.connections.list": ConversationProductCapability.KNOWLEDGE_CONFIGURATION_DISCOVERY,
    "knowledge.resources.list": ConversationProductCapability.KNOWLEDGE_CONFIGURATION_DISCOVERY,
    "knowledge.capabilities.list": ConversationProductCapability.KNOWLEDGE_CONFIGURATION_DISCOVERY,
    "knowledge.inventory.list": ConversationProductCapability.SOURCE_DISCOVERY,
    "knowledge.operation.execute": ConversationProductCapability.WORKSPACE_ADMINISTRATION,
    "destructive.confirm": ConversationProductCapability.WORKSPACE_ADMINISTRATION,
    "tenant_connection.providers.list": ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION,
    "tenant_connection.connections.list": ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION,
    "tenant_connection.connection.inspect": ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION,
    "tenant_connection.authorization.begin": ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION,
    "tenant_connection.authorization.complete_manual": (
        ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION
    ),
    "tenant_connection.connection.reconnect": ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION,
    "tenant_connection.connection.revoke": ConversationProductCapability.TENANT_CONNECTION_ADMINISTRATION,
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
        "knowledge_connection_not_found",
        "knowledge_connection_not_active",
        "knowledge_resource_discovery_unavailable",
        "knowledge_resource_not_found",
        "knowledge_resource_not_available",
        "knowledge_capability_not_found",
        "knowledge_capability_not_bindable",
        "knowledge_configuration_snapshot_stale",
        "knowledge_plugin_configuration_unavailable",
        "citation_context_not_found",
        "citation_ordinal_invalid",
        "citation_not_available",
        "document_not_found",
        "document_forbidden",
        "document_inspect_unavailable",
        "destructive_confirmation_invalid",
        "destructive_confirmation_expired",
        "destructive_confirmation_stale",
        "destructive_confirmation_required",
        "knowledge_inventory_unavailable",
        "knowledge_target_not_found",
        "knowledge_target_ambiguous",
        "knowledge_operation_not_available",
        "knowledge_operation_conflict",
        "tenant_connection_unavailable",
        "tenant_connection_provider_not_found",
        "tenant_connection_provider_ambiguous",
        "tenant_connection_not_found",
        "tenant_connection_ambiguous",
        "tenant_connection_authorization_pending_not_found",
        "connection_not_found",
        "connection_revoked",
        "connection_not_active",
        "connection_provider_unsupported",
        "connection_provider_misconfigured",
        "authorization_redirect_not_allowed",
        "authorization_transaction_not_found",
        "authorization_transaction_expired",
        "authorization_already_in_progress",
        "authorization_state_invalid",
        "credential_binding_invalid",
        "connection_already_exists",
        "connection_version_conflict",
        "connection_runtime_unavailable",
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
        document_inspect_service: DocumentInspectService | None = None,
        citation_context_service: ConversationCitationContextService | None = None,
        knowledge_plugin_configuration_service: KnowledgePluginConfigurationService | None = None,
        knowledge_inspection_service: KnowledgeInspectionService | None = None,
        knowledge_operations_service: KnowledgeOperationsService | None = None,
        destructive_confirmation_codec: HmacDestructiveActionConfirmationCodec | None = None,
        connection_auth_context_service: ConversationConnectionAuthContextService | None = None,
        tenant_connection_config: TenantConnectionConversationConfig | None = None,
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
        self._document_inspect_service = document_inspect_service
        self._citation_context_service = citation_context_service
        self._knowledge_plugin_configuration = knowledge_plugin_configuration_service
        self._knowledge_inspection = knowledge_inspection_service
        self._knowledge_operations = knowledge_operations_service
        self._destructive_confirmation = destructive_confirmation_codec
        self._connection_auth_context = connection_auth_context_service
        self._tenant_connection_config = tenant_connection_config or TenantConnectionConversationConfig()
        self._knowledge_idempotency = Sha256KnowledgeAdministrationIdempotencyKeyFactory()
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
        thread_memory_user_text = self._thread_memory_user_text(plan=plan)
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
            thread_memory_user_text=thread_memory_user_text,
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
        if isinstance(action, KnowledgeConnectionsListPlannedAction):
            service = self._require_knowledge_plugin_configuration()
            connections = cast(
                Sequence[KnowledgeConnectionSummaryV1],
                await _maybe_await(
                service.list_connections(
                    tenant_id=tenant_id,
                    execution_context=command.execution_context,
                )
                ),
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "connections": [
                        item.model_dump(mode="json") for item in connections
                    ]
                },
            )
        if isinstance(action, TenantConnectionProvidersListPlannedAction):
            return await self._execute_tenant_connection_providers_list(
                tenant_id=tenant_id,
            )
        if isinstance(action, TenantConnectionConnectionsListPlannedAction):
            return await self._execute_tenant_connection_connections_list(
                tenant_id=tenant_id,
            )
        if isinstance(action, TenantConnectionInspectPlannedAction):
            return await self._execute_tenant_connection_inspect(
                tenant_id=tenant_id,
                connection_ref=action.connection_ref,
            )
        if isinstance(action, TenantConnectionBeginAuthorizationPlannedAction):
            return await self._execute_tenant_connection_begin_authorization(
                action=action,
                tenant_id=tenant_id,
                command=command,
            )
        if isinstance(action, TenantConnectionCompleteManualAuthorizationPlannedAction):
            return await self._execute_tenant_connection_complete_manual_authorization(
                command=command,
            )
        if isinstance(action, TenantConnectionReconnectPlannedAction):
            return await self._execute_tenant_connection_reconnect(
                tenant_id=tenant_id,
                connection_ref=action.connection_ref,
                command=command,
            )
        if isinstance(action, TenantConnectionRevokePlannedAction):
            return await self._execute_tenant_connection_revoke(
                action=action,
                tenant_id=tenant_id,
                command=command,
            )
        if isinstance(action, KnowledgeResourcesListPlannedAction):
            service = self._require_knowledge_plugin_configuration()
            page = cast(
                KnowledgeRemoteResourcePageV1,
                await _maybe_await(
                service.list_remote_resources(
                    tenant_id=tenant_id,
                    execution_context=command.execution_context,
                    connection_ref=action.connection_ref,
                    source_kind=action.source_kind,
                    page_token=action.page_token,
                )
                ),
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "resources": [item.model_dump(mode="json") for item in page.resources],
                    "next_page_token": page.next_page_token,
                    "snapshot_version": page.snapshot_version,
                },
            )
        if isinstance(action, KnowledgeCapabilitiesListPlannedAction):
            service = self._require_knowledge_plugin_configuration()
            capabilities = cast(
                Sequence[KnowledgeCapabilitySummaryV1],
                await _maybe_await(
                service.list_resource_capabilities(
                    tenant_id=tenant_id,
                    execution_context=command.execution_context,
                    connection_ref=action.connection_ref,
                    remote_resource_id=action.remote_resource_id,
                )
                ),
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "capabilities": [
                        item.model_dump(mode="json") for item in capabilities
                    ]
                },
            )
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
                    "name": (
                        selected_workspace.name.strip()
                        if selected_workspace is not None
                        and selected_workspace.name.strip()
                        else "Workspace"
                    ),
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
            workspace = self._workspace_service.get_workspace(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if workspace is None:
                raise ConversationReferenceResolutionError("workspace_not_found")
            codec = self._require_destructive_confirmation_codec()
            token = codec.issue(
                DestructiveActionConfirmationV1(
                    token="",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    action_kind=DestructiveActionKindV1.WORKSPACE_DELETE,
                    target_id=workspace_id,
                    expected_state_version=workspace.workspace_revision,
                    expires_at=self._utc_now() + codec.ttl,
                )
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "name": str(workspace.name),
                    "status": "confirmation_required",
                    "action_kind": DestructiveActionKindV1.WORKSPACE_DELETE,
                    "confirmation_token": token,
                },
            )
        if isinstance(action, DestructiveActionConfirmPlannedAction):
            return await self._execute_destructive_confirm(
                action=action,
                tenant_id=tenant_id,
                resolver=resolver,
                execution_id=execution_id,
            )
        if isinstance(action, KnowledgeInventoryListPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            inventory = self._list_knowledge_inventory(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            filtered = _filter_knowledge_inventory(inventory, action.inventory_filter)
            workspace_name = _safe_workspace_name(
                self._workspace_service.get_workspace(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                ),
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "workspace_name": workspace_name,
                    "inventory_filter": action.inventory_filter.value,
                    "items": [_safe_inventory_item(item) for item in filtered.items],
                    "summary": filtered.summary.model_dump(mode="json"),
                },
            )
        if isinstance(action, KnowledgeOperationExecutePlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            return await self._execute_knowledge_operation(
                action=action,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                execution_id=execution_id,
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
                            "source_type": item.source_type.value,
                            "status": item.status.value,
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
            candidate_id = candidate.candidate_id
            candidate_label = candidate.label
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
                    "source_id": acceptance.source_id,
                    "operation_id": acceptance.operation_id,
                    "status": acceptance.status,
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
                    "batch_id": batch.batch_id,
                    "status": batch.status.value,
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
            knowledge_scope = None
            scope_labels: list[str] = []
            if action.knowledge_targets is not None:
                inventory = self._list_knowledge_inventory(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                )
                knowledge_item_ids: list[str] = []
                for target in action.knowledge_targets:
                    resolution = resolve_knowledge_target(
                        inventory.items,
                        reference_kind=target.target_reference_kind.value,
                        target_reference=target.target_reference,
                    )
                    if resolution.ambiguous:
                        raise RuntimeError("knowledge_target_ambiguous")
                    if resolution.item is None:
                        raise RuntimeError("knowledge_target_not_found")
                    knowledge_item_ids.append(resolution.item.knowledge_item_id)
                    if resolution.item.display_label:
                        scope_labels.append(resolution.item.display_label)
                knowledge_scope = KnowledgeAskScopeV1(
                    knowledge_item_ids=tuple(knowledge_item_ids)
                )
            run = await self._ask_service.ask(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                question=action.question,
                knowledge_scope=knowledge_scope,
            )
            artifact_data: dict[str, object] = {
                "run_id": run.run_id,
                "status": run.status.value,
                "answer": run.answer,
                "citations": _safe_json(run.citations),
            }
            if scope_labels:
                artifact_data["scope_labels"] = scope_labels
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data=artifact_data,
            )
        if isinstance(action, CitationInspectPlannedAction):
            workspace_id = self._resolve_workspace_id(
                action_id=action.action_id,
                reference=action.workspace,
                resolver=resolver,
                created_workspace_ids=created_workspace_ids,
                resolved_workspace_ids=resolved_workspace_ids,
            )
            if (
                self._citation_context_service is None
                or self._document_inspect_service is None
            ):
                raise RuntimeError("document_inspect_unavailable")
            try:
                resolved = self._citation_context_service.resolve_citation(
                    context=command.execution_context,
                    workspace_id=workspace_id,
                    citation_ordinal=action.citation_ordinal,
                )
            except ConversationCitationContextError as exc:
                raise RuntimeError(exc.error_code) from exc
            page = (
                resolved.citation.location.page
                if resolved.citation.location is not None
                else None
            )
            try:
                inspected = self._document_inspect_service.inspect(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    document_id=resolved.citation.document_id,
                    preview_hint=resolved.citation.excerpt or None,
                    page=page,
                    logical_location_hint=resolved.citation.file_name,
                )
            except DocumentInspectError as exc:
                raise RuntimeError(exc.error_code) from exc
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "workspace_id": workspace_id,
                    "citation_ordinal": action.citation_ordinal,
                    "run_id": resolved.run_id,
                    "document_id": inspected.document_id,
                    "display_name": inspected.display_name,
                    "source_type": inspected.source_type,
                    "source_label": inspected.source_label,
                    "logical_location": inspected.logical_location,
                    "location": (
                        {
                            "page": inspected.location.page,
                            "logical_location": inspected.location.logical_location,
                        }
                        if inspected.location is not None
                        else None
                    ),
                    "preview": inspected.preview,
                    "external_url": inspected.external_url,
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
                "source_id": accepted.source_id,
                "operation_id": accepted.operation_id,
                "status": accepted.status,
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
                return {
                    "object_id": source.object_id,
                    "source_id": accepted.source_id,
                    "operation_id": accepted.operation_id,
                    "status": accepted.status,
                }
            if source.reference_kind == "folder":
                registered = self._workspace_service.register_local_folder_source(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    path=source.value,
                    recursive=True,
                )
                return {
                    "object_id": source.object_id,
                    "source_id": registered.source_id,
                    "operation_id": "",
                    "status": registered.status.value,
                }
            raise RuntimeError("source_reference_unsupported")
        raise RuntimeError("source_reference_unsupported")

    def _resolve_candidate_snapshot(
        self,
        candidates: Sequence[ConversationPlanningSourceCandidate],
        action: SourceCandidateAttachPlannedAction,
    ) -> ConversationPlanningSourceCandidate:
        reference = action.candidate_reference.strip().casefold()
        if action.candidate_reference_kind == "ordinal":
            if not reference.isdigit():
                raise RuntimeError("source_candidate_not_found")
            index = int(reference) - 1
            if index < 0 or index >= len(candidates):
                raise RuntimeError("source_candidate_not_found")
            candidate = candidates[index]
            if not candidate.available:
                raise RuntimeError("source_candidate_unavailable")
            return candidate
        matches = [
            candidate
            for candidate in candidates
            if candidate.label.strip().casefold() == reference
        ]
        if not matches:
            raise RuntimeError("source_candidate_not_found")
        if len(matches) > 1:
            raise RuntimeError("source_candidate_ambiguous")
        if not matches[0].available:
            raise RuntimeError("source_candidate_unavailable")
        return matches[0]

    def _require_connection_auth_context_service(
        self,
    ) -> ConversationConnectionAuthContextService:
        if self._connection_auth_context is None:
            raise RuntimeError("tenant_connection_unavailable")
        return self._connection_auth_context

    def _tenant_orchestration(
        self,
        tenant_id: str,
    ):
        return self._require_connection_auth_context_service().orchestration_for(tenant_id)

    def _oauth_redirect_uri(self) -> str:
        redirect_uri = self._tenant_connection_config.oauth_redirect_uri
        if redirect_uri is None or not redirect_uri.strip():
            raise RuntimeError("authorization_redirect_not_allowed")
        return redirect_uri.strip()

    async def _execute_tenant_connection_providers_list(
        self,
        *,
        tenant_id: str,
    ) -> ConversationExecutionArtifact:
        service = self._tenant_orchestration(tenant_id)
        providers = service.list_supported_connection_providers()
        return ConversationExecutionArtifact(
            artifact_type="tenant_connection.providers.list",
            data={
                "providers": [dict(item) for item in providers],
            },
        )

    async def _execute_tenant_connection_connections_list(
        self,
        *,
        tenant_id: str,
    ) -> ConversationExecutionArtifact:
        service = self._tenant_orchestration(tenant_id)
        connections = service.list_connections()
        return ConversationExecutionArtifact(
            artifact_type="tenant_connection.connections.list",
            data={
                "connections": [
                    safe_connection_payload(connection) for connection in connections
                ],
            },
        )

    async def _execute_tenant_connection_inspect(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> ConversationExecutionArtifact:
        service = self._tenant_orchestration(tenant_id)
        try:
            connection = service.get_connection(connection_ref)
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc
        return ConversationExecutionArtifact(
            artifact_type="tenant_connection.connection.inspect",
            data={"connection": safe_connection_payload(connection)},
        )

    async def _execute_tenant_connection_begin_authorization(
        self,
        *,
        action: TenantConnectionBeginAuthorizationPlannedAction,
        tenant_id: str,
        command: ConversationInteractionExecutionCommand,
    ) -> ConversationExecutionArtifact:
        service = self._tenant_orchestration(tenant_id)
        redirect_uri = self._oauth_redirect_uri()
        try:
            result = service.begin_connection_authorization(
                provider_id=action.provider_id,
                redirect_uri=redirect_uri,
            )
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc

        auth_context = self._require_connection_auth_context_service()
        auth_context.record_pending_authorization(
            context=command.execution_context,
            authorization_transaction_ref=result.authorization_transaction_ref,
            provider_id=action.provider_id,
            required_user_action=result.required_user_action,
        )

        return ConversationExecutionArtifact(
            artifact_type=action.action_type,
            data={
                "provider_id": action.provider_id,
                "required_user_action": result.required_user_action,
                "authorization_url": result.authorization_url,
                "expires_at": result.expires_at.isoformat(),
                "manual_instructions": result.manual_instructions,
            },
        )

    async def _execute_tenant_connection_complete_manual_authorization(
        self,
        *,
        command: ConversationInteractionExecutionCommand,
    ) -> ConversationExecutionArtifact:
        auth_context = self._require_connection_auth_context_service()
        try:
            pending = auth_context.require_pending_manual_authorization(
                context=command.execution_context,
            )
        except ConversationConnectionAuthContextError as exc:
            raise RuntimeError(exc.error_code) from exc
        try:
            credential_payload = parse_manual_credential_payload(
                command.planning_request.message_text
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        service = auth_context.orchestration_for(command.execution_context.tenant_id)
        try:
            result = service.complete_connection_authorization(
                authorization_transaction_ref=pending.authorization_transaction_ref,
                credential_payload=credential_payload,
            )
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc
        auth_context.clear_pending_authorization(context=command.execution_context)
        return ConversationExecutionArtifact(
            artifact_type="tenant_connection.authorization.complete_manual",
            data={
                "connection": safe_connection_payload(result.connection),
                "disposition": result.disposition,
                "redact_user_message": True,
            },
        )

    async def _execute_tenant_connection_reconnect(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        command: ConversationInteractionExecutionCommand,
    ) -> ConversationExecutionArtifact:
        service = self._tenant_orchestration(tenant_id)
        redirect_uri = self._oauth_redirect_uri()
        try:
            result = service.reconnect_connection(
                connection_ref=connection_ref,
                redirect_uri=redirect_uri,
            )
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc
        connection = service.get_connection(connection_ref)
        auth_context = self._require_connection_auth_context_service()
        auth_context.record_pending_authorization(
            context=command.execution_context,
            authorization_transaction_ref=result.authorization_transaction_ref,
            provider_id=connection.provider_id,
            required_user_action=result.required_user_action,
        )
        return ConversationExecutionArtifact(
            artifact_type="tenant_connection.connection.reconnect",
            data={
                "connection_ref": connection_ref,
                "required_user_action": result.required_user_action,
                "authorization_url": result.authorization_url,
                "expires_at": result.expires_at.isoformat(),
                "manual_instructions": result.manual_instructions,
            },
        )

    async def _execute_tenant_connection_revoke(
        self,
        *,
        action: TenantConnectionRevokePlannedAction,
        tenant_id: str,
        command: ConversationInteractionExecutionCommand,
    ) -> ConversationExecutionArtifact:
        service = self._tenant_orchestration(tenant_id)
        try:
            connection = service.get_connection(action.connection_ref)
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc
        if action.confirmation_token is not None:
            return await self._execute_tenant_connection_revoke_confirmed(
                confirmation=self._verify_destructive_confirmation(
                    token=action.confirmation_token,
                    tenant_id=tenant_id,
                    workspace_id=command.execution_context.workspace_id,
                    action_kind=tenant_connection_revoke_action_kind(),
                    target_id=action.connection_ref,
                ),
                tenant_id=tenant_id,
            )
        codec = self._require_destructive_confirmation_codec()
        token = codec.issue(
            DestructiveActionConfirmationV1(
                token="",
                tenant_id=tenant_id,
                workspace_id=command.execution_context.workspace_id,
                action_kind=tenant_connection_revoke_action_kind(),
                target_id=connection.connection_ref,
                expected_state_version=connection.configuration_version,
                expires_at=self._utc_now() + codec.ttl,
            )
        )
        return ConversationExecutionArtifact(
            artifact_type=action.action_type,
            data={
                "status": "confirmation_required",
                "action_kind": tenant_connection_revoke_action_kind(),
                "connection_ref": connection.connection_ref,
                "display_name": connection.safe_display_name,
                "confirmation_token": token,
            },
        )

    async def _execute_tenant_connection_revoke_confirmed(
        self,
        *,
        confirmation: DestructiveActionConfirmationV1,
        tenant_id: str,
    ) -> ConversationExecutionArtifact:
        if confirmation.tenant_id != tenant_id:
            raise RuntimeError("destructive_confirmation_invalid")
        if confirmation.action_kind != tenant_connection_revoke_action_kind():
            raise RuntimeError("destructive_confirmation_invalid")
        service = self._tenant_orchestration(tenant_id)
        try:
            current = service.get_connection(confirmation.target_id)
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc
        if current.configuration_version != confirmation.expected_state_version:
            raise RuntimeError("destructive_confirmation_stale")
        try:
            revoked = service.revoke_connection(connection_ref=confirmation.target_id)
        except TenantConnectionProductError as exc:
            raise RuntimeError(exc.error_code) from exc
        return ConversationExecutionArtifact(
            artifact_type="tenant_connection.connection.revoke",
            data={
                "status": "completed",
                "connection": safe_connection_payload(revoked),
            },
        )

    def _thread_memory_user_text(
        self,
        *,
        plan: ConversationInteractionPlan,
    ) -> str | None:
        if interaction_plan_requires_credential_memory_redaction(plan):
            return THREAD_MEMORY_CREDENTIAL_REDACTION
        return None

    def _require_candidate_service(self) -> CandidateIntakeService:
        if self._source_candidate_service is None:
            raise RuntimeError("source_candidate_unavailable")
        return self._source_candidate_service

    def _require_knowledge_plugin_configuration(
        self,
    ) -> KnowledgePluginConfigurationService:
        if self._knowledge_plugin_configuration is None:
            raise RuntimeError("knowledge_plugin_configuration_unavailable")
        return self._knowledge_plugin_configuration

    def _require_knowledge_inspection(self) -> KnowledgeInspectionService:
        if self._knowledge_inspection is None:
            raise RuntimeError("knowledge_inventory_unavailable")
        return self._knowledge_inspection

    def _require_knowledge_operations(self) -> KnowledgeOperationsService:
        if self._knowledge_operations is None:
            raise RuntimeError("knowledge_inventory_unavailable")
        return self._knowledge_operations

    def _require_destructive_confirmation_codec(
        self,
    ) -> HmacDestructiveActionConfirmationCodec:
        if self._destructive_confirmation is None:
            raise RuntimeError("destructive_confirmation_required")
        return self._destructive_confirmation

    def _list_knowledge_inventory(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> KnowledgeInventoryV1:
        try:
            return self._require_knowledge_inspection().list_items(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except KnowledgeInventoryError as exc:
            raise RuntimeError(exc.error_code) from exc

    async def _execute_knowledge_operation(
        self,
        *,
        action: KnowledgeOperationExecutePlannedAction,
        tenant_id: str,
        workspace_id: str,
        execution_id: str,
    ) -> ConversationExecutionArtifact:
        inventory = self._list_knowledge_inventory(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        resolution = resolve_knowledge_target(
            inventory.items,
            reference_kind=action.target_reference_kind.value,
            target_reference=action.target_reference,
        )
        if resolution.ambiguous:
            raise RuntimeError("knowledge_target_ambiguous")
        if resolution.item is None:
            raise RuntimeError("knowledge_target_not_found")
        item = resolution.item
        operation = _map_operation_kind(action.operation)
        if operation not in item.available_actions:
            raise RuntimeError("knowledge_operation_not_available")

        supplied_token = action.confirmation_token
        confirmation: DestructiveActionConfirmationV1 | None = None
        if supplied_token is not None:
            confirmation = self._verify_destructive_confirmation(
                token=supplied_token,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                action_kind=knowledge_operation_action_kind(operation),
                target_id=item.knowledge_item_id,
            )
            item = self._require_knowledge_inspection().get_item(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                knowledge_item_id=item.knowledge_item_id,
            )
            if item.revision != confirmation.expected_state_version:
                raise RuntimeError("destructive_confirmation_stale")
            if operation not in item.available_actions:
                raise RuntimeError("knowledge_operation_not_available")
        elif operation is KnowledgeOperationV1.DETACH:
            codec = self._require_destructive_confirmation_codec()
            token = codec.issue(
                DestructiveActionConfirmationV1(
                    token="",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    action_kind=knowledge_detach_action_kind(),
                    target_id=item.knowledge_item_id,
                    expected_state_version=item.revision,
                    expires_at=self._utc_now() + codec.ttl,
                )
            )
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "status": "confirmation_required",
                    "action_kind": knowledge_detach_action_kind(),
                    "operation": operation.value,
                    "knowledge_item_id": item.knowledge_item_id,
                    "display_label": item.display_label,
                    "confirmation_token": token,
                },
            )

        command = KnowledgeOperationCommandV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            knowledge_item_id=item.knowledge_item_id,
            operation=operation,
            expected_revision=(
                confirmation.expected_state_version
                if confirmation is not None
                else item.revision
            ),
            idempotency_key_hash=self._knowledge_idempotency.create(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                knowledge_item_id=item.knowledge_item_id,
                operation=operation,
                request_id=execution_id,
            ),
        )
        try:
            result = await self._require_knowledge_operations().execute(command)
        except KnowledgeOperationError as exc:
            if exc.error_code == "knowledge_operation_conflict":
                raise RuntimeError("knowledge_operation_conflict") from exc
            if exc.error_code == "knowledge_item_not_found":
                raise RuntimeError("knowledge_target_not_found") from exc
            raise RuntimeError("knowledge_inventory_unavailable") from exc
        return ConversationExecutionArtifact(
            artifact_type=action.action_type,
            data={
                "status": "completed",
                "operation": operation.value,
                "item": _safe_inventory_item(result.item),
            },
        )

    async def _execute_destructive_confirm(
        self,
        *,
        action: DestructiveActionConfirmPlannedAction,
        tenant_id: str,
        resolver: ConversationInteractionReferenceResolver,
        execution_id: str,
    ) -> ConversationExecutionArtifact:
        codec = self._require_destructive_confirmation_codec()
        try:
            confirmation = codec.verify(action.confirmation_token)
        except DestructiveActionConfirmationError as exc:
            raise RuntimeError(exc.error_code) from exc

        if confirmation.action_kind == DestructiveActionKindV1.WORKSPACE_DELETE:
            if (
                confirmation.tenant_id != tenant_id
                or confirmation.workspace_id != confirmation.target_id
            ):
                raise RuntimeError("destructive_confirmation_invalid")
            outcome, workspace_name = self._workspace_service.delete_workspace_with_revision_claim(
                tenant_id=tenant_id,
                workspace_id=confirmation.target_id,
                expected_revision=confirmation.expected_state_version,
            )
            if outcome == "not_found":
                raise RuntimeError("workspace_not_found")
            if outcome == "stale":
                raise RuntimeError("destructive_confirmation_stale")
            if resolver.current_active_workspace_id == confirmation.target_id:
                resolver.clear_active_workspace()
            return ConversationExecutionArtifact(
                artifact_type=action.action_type,
                data={
                    "status": "completed",
                    "action_kind": confirmation.action_kind,
                    "workspace_id": confirmation.target_id,
                    "name": str(workspace_name),
                    "deleted": True,
                },
            )

        if confirmation.action_kind == knowledge_detach_action_kind():
            inventory = self._list_knowledge_inventory(
                tenant_id=tenant_id,
                workspace_id=confirmation.workspace_id,
            )
            item = next(
                (
                    candidate
                    for candidate in inventory.items
                    if candidate.knowledge_item_id == confirmation.target_id
                ),
                None,
            )
            if item is None:
                raise RuntimeError("knowledge_target_not_found")
            operation_action = KnowledgeOperationExecutePlannedAction(
                action_id=action.action_id,
                action_type="knowledge.operation.execute",
                workspace=WorkspaceReference(
                    kind=WorkspaceReferenceKind.active,
                    value=None,
                ),
                operation=KnowledgeOperationKind.detach,
                target_reference_kind=KnowledgeTargetReferenceKind.knowledge_item_id,
                target_reference=confirmation.target_id,
                confirmation_token=action.confirmation_token,
            )
            return await self._execute_knowledge_operation(
                action=operation_action,
                tenant_id=tenant_id,
                workspace_id=confirmation.workspace_id,
                execution_id=execution_id,
            )

        if confirmation.action_kind == tenant_connection_revoke_action_kind():
            return await self._execute_tenant_connection_revoke_confirmed(
                confirmation=confirmation,
                tenant_id=tenant_id,
            )

        raise RuntimeError("destructive_confirmation_invalid")

    def _verify_destructive_confirmation(
        self,
        *,
        token: str,
        tenant_id: str,
        workspace_id: str,
        action_kind: str,
        target_id: str,
    ) -> DestructiveActionConfirmationV1:
        codec = self._require_destructive_confirmation_codec()
        try:
            confirmation = codec.verify(token)
        except DestructiveActionConfirmationError as exc:
            raise RuntimeError(exc.error_code) from exc
        if (
            confirmation.tenant_id != tenant_id
            or confirmation.workspace_id != workspace_id
            or confirmation.action_kind != action_kind
            or confirmation.target_id != target_id
        ):
            raise RuntimeError("destructive_confirmation_invalid")
        return confirmation

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


def _filter_knowledge_inventory(
    inventory: KnowledgeInventoryV1,
    inventory_filter: KnowledgeInventoryFilter,
) -> KnowledgeInventoryV1:
    items = inventory.items
    if inventory_filter is KnowledgeInventoryFilter.all:
        filtered = items
    elif inventory_filter is KnowledgeInventoryFilter.indexed:
        filtered = tuple(item for item in items if item.mode is KnowledgeAccessModeV1.INDEXED)
    elif inventory_filter is KnowledgeInventoryFilter.live:
        filtered = tuple(item for item in items if item.mode is KnowledgeAccessModeV1.LIVE)
    elif inventory_filter is KnowledgeInventoryFilter.active:
        filtered = tuple(item for item in items if item.lifecycle_state == "active")
    elif inventory_filter is KnowledgeInventoryFilter.disabled:
        filtered = tuple(item for item in items if item.lifecycle_state == "disabled")
    else:
        filtered = tuple(
            item
            for item in items
            if item.lifecycle_state in {"error", "detach_blocked"}
            or (
                item.runtime_available is False
                and item.enabled
                and not item.detached
            )
        )
    summary = KnowledgeInventorySummaryV1(
        total=len(filtered),
        indexed=sum(item.mode is KnowledgeAccessModeV1.INDEXED for item in filtered),
        live=sum(item.mode is KnowledgeAccessModeV1.LIVE for item in filtered),
        active=sum(item.lifecycle_state == "active" for item in filtered),
        disabled=sum(item.lifecycle_state == "disabled" for item in filtered),
        attention_required=sum(
            item.lifecycle_state in {"error", "detach_blocked"}
            or (
                item.runtime_available is False
                and item.enabled
                and not item.detached
            )
            for item in filtered
        ),
    )
    return KnowledgeInventoryV1(
        tenant_id=inventory.tenant_id,
        workspace_id=inventory.workspace_id,
        items=filtered,
        summary=summary,
        updated_at=inventory.updated_at,
    )


def _safe_inventory_item(item: KnowledgeInventoryItemV1) -> dict[str, object]:
    needs_attention = (
        item.lifecycle_state in {"error", "detach_blocked"}
        or (
            item.runtime_available is False
            and item.enabled
            and not item.detached
        )
    )
    last_sync = item.last_successful_sync_at
    return {
        "display_label": item.display_label,
        "mode": item.mode.value,
        "lifecycle_state": item.lifecycle_state,
        "enabled": item.enabled,
        "detached": item.detached,
        "runtime_available": item.runtime_available,
        "sync_state": item.sync_state,
        "last_successful_sync_at": (
            last_sync.isoformat() if last_sync is not None else None
        ),
        "needs_attention": needs_attention,
        "available_actions": [action.value for action in item.available_actions],
    }


def _safe_workspace_name(workspace: Workspace | None) -> str:
    if workspace is None:
        return "Workspace"
    name = workspace.name
    if name.strip():
        return name.strip()
    return "Workspace"


def _map_operation_kind(kind: KnowledgeOperationKind) -> KnowledgeOperationV1:
    return KnowledgeOperationV1(kind.value)


def _safe_exception_code(exc: Exception) -> str:
    normalized_aliases = {
        "candidate_not_found": "source_candidate_not_found",
        "source_candidate_configuration_invalid": "source_candidate_unavailable",
        "source_candidate_configuration_changed": "source_candidate_unavailable",
    }
    code: str
    if isinstance(exc, _PreflightFailure):
        code = exc.code
    elif isinstance(exc, ConversationReferenceResolutionError):
        code = exc.code
    elif isinstance(
        exc,
        (
            ConversationConnectionAuthContextError,
            TenantConnectionProductError,
            ConversationWorkspaceSelectionError,
            KnowledgeInventoryError,
            KnowledgeOperationError,
            DocumentInspectError,
            DestructiveActionConfirmationError,
            ConversationCitationContextError,
        ),
    ):
        code = exc.error_code
    elif isinstance(exc, RuntimeError):
        code = str(exc).strip()
    else:
        code = str(exc).strip()
    if code in normalized_aliases:
        code = normalized_aliases[code]
    return code if code in _SAFE_ERROR_CODES else "action_execution_failed"


def _safe_candidate(candidate: SourceCandidateSummary) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "source_type": candidate.source_type,
        "available": candidate.available,
    }


def _safe_json(value: object) -> object:
    if isinstance(value, BaseModel):
        return _safe_json(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value
