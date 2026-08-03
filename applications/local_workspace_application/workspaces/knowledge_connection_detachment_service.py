# © Artur Czarnecki. All rights reserved.

"""Workspace Connection Detachment domain service."""

from __future__ import annotations

from dataclasses import dataclass

from local_workspace_application.workspaces.knowledge_access_service import (
    TenantKnowledgeSourceBindingPort,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    connection_attachment_id,
    connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    WorkspaceConnectionAttachmentError,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_handler import (
    DetachConnectionMutationIntent,
    detach_connection_request_hash,
    detach_connection_stage_manifest_hash,
)


@dataclass(frozen=True, slots=True)
class DetachWorkspaceConnectionCommand:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    expected_revision: int
    idempotency_key_hash: str


@dataclass(frozen=True, slots=True)
class DetachWorkspaceConnectionResult:
    attachment: WorkspaceConnectionAttachment
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1


def _projection_incomplete() -> WorkspaceConnectionAttachmentError:
    return WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")


class WorkspaceConnectionDetachmentService:
    def __init__(
        self,
        *,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
        tenant_binding_port: TenantKnowledgeSourceBindingPort,
    ) -> None:
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine
        self._tenant_binding_port = tenant_binding_port

    def detach_connection(
        self,
        command: DetachWorkspaceConnectionCommand,
    ) -> DetachWorkspaceConnectionResult:
        tenant_id = command.tenant_id.strip()
        workspace_id = command.workspace_id.strip()
        connection_ref = command.connection_ref.strip()
        if not connection_ref:
            raise WorkspaceConnectionAttachmentError("connection_attachment_not_found")

        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceConnectionAttachmentError("workspace_not_found")

        attachment_id = connection_attachment_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        attachment = _resolve_current_attachment(
            configuration=configuration,
            attachment_id=attachment_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        if attachment is None:
            raise WorkspaceConnectionAttachmentError("connection_attachment_not_found")

        indexed_ids = _resolve_indexed_dependencies(
            configuration=configuration,
            tenant_binding_port=self._tenant_binding_port,
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        live_ids = _resolve_live_dependencies(
            configuration=configuration,
            connection_ref=connection_ref,
        )
        intent = DetachConnectionMutationIntent(
            attachment_id=attachment_id,
            connection_ref=connection_ref,
            indexed_source_binding_ids=indexed_ids,
            live_access_binding_ids=live_ids,
        )
        manifest_hash = detach_connection_stage_manifest_hash(
            attachment_id=attachment_id,
            connection_ref=connection_ref,
            indexed_source_binding_ids=intent.indexed_source_binding_ids,
            live_access_binding_ids=intent.live_access_binding_ids,
        )
        mutation_result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=detach_connection_request_hash(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            ),
            semantic_identity_hash=connection_attachment_semantic_identity_hash(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            ),
            stage_manifest_hash=manifest_hash,
            intent=intent,
        )
        resolved_configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if resolved_configuration is None:
            raise WorkspaceConnectionAttachmentError("workspace_not_found")

        resolved_attachment = _resolve_committed_detached_attachment(
            configuration=resolved_configuration,
            result_entity_id=mutation_result.result_entity_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            configuration_revision=mutation_result.configuration_revision,
        )
        _verify_cascade(
            configuration=resolved_configuration,
            indexed_ids=intent.indexed_source_binding_ids,
            live_ids=intent.live_access_binding_ids,
        )
        return DetachWorkspaceConnectionResult(
            attachment=resolved_attachment,
            configuration_revision=mutation_result.configuration_revision,
            disposition=mutation_result.disposition,
        )


def _resolve_current_attachment(
    *,
    configuration: WorkspaceKnowledgeConfigurationV1,
    attachment_id: str,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
) -> WorkspaceConnectionAttachment | None:
    matches = [
        item
        for item in configuration.connection_attachments
        if item.attachment_id == attachment_id
    ]
    if len(matches) != 1:
        return None
    attachment = matches[0]
    if (
        attachment.tenant_id != tenant_id
        or attachment.workspace_id != workspace_id
        or attachment.connection_ref != connection_ref
    ):
        raise _projection_incomplete()
    return attachment


def _resolve_indexed_dependencies(
    *,
    configuration: WorkspaceKnowledgeConfigurationV1,
    tenant_binding_port: TenantKnowledgeSourceBindingPort,
    tenant_id: str,
    connection_ref: str,
) -> tuple[str, ...]:
    result: list[str] = []
    for binding in configuration.indexed_sources:
        if binding.status not in (
            WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
            WorkspaceIndexedSourceBindingStatusV1.ERROR,
        ):
            continue
        binding_ref = binding.knowledge_source_binding_ref.strip()
        try:
            tenant_binding = tenant_binding_port.get_binding(
                tenant_id=tenant_id,
                binding_id=binding_ref,
            )
        except Exception:
            raise WorkspaceConnectionAttachmentError(
                "connection_detach_dependency_resolution_failed"
            ) from None
        if (
            tenant_binding is None
            or tenant_binding.tenant_id != tenant_id
            or tenant_binding.binding_id != binding_ref
            or not tenant_binding.connection_ref.strip()
        ):
            raise WorkspaceConnectionAttachmentError(
                "connection_detach_dependency_resolution_failed"
            )
        if tenant_binding.connection_ref.strip() == connection_ref:
            result.append(binding.indexed_source_binding_id)
    result.sort()
    return tuple(result)


def _resolve_live_dependencies(
    *,
    configuration: WorkspaceKnowledgeConfigurationV1,
    connection_ref: str,
) -> tuple[str, ...]:
    result = [
        binding.live_access_binding_id
        for binding in configuration.live_access_bindings
        if binding.connection_ref.strip() == connection_ref
        and binding.status is LiveAccessBindingStatusV1.ACTIVE
    ]
    result.sort()
    return tuple(result)


def _resolve_committed_detached_attachment(
    *,
    configuration: WorkspaceKnowledgeConfigurationV1,
    result_entity_id: str,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    configuration_revision: int,
) -> WorkspaceConnectionAttachment:
    for item in configuration.connection_attachments:
        if item.attachment_id != result_entity_id:
            continue
        if (
            item.tenant_id != tenant_id
            or item.workspace_id != workspace_id
            or item.connection_ref != connection_ref
            or item.status is not WorkspaceConnectionAttachmentStatusV1.DETACHED
            or item.effective_revision > configuration_revision
        ):
            raise _projection_incomplete()
        return item
    raise _projection_incomplete()


def _verify_cascade(
    *,
    configuration: WorkspaceKnowledgeConfigurationV1,
    indexed_ids: tuple[str, ...],
    live_ids: tuple[str, ...],
) -> None:
    indexed_map = {b.indexed_source_binding_id: b for b in configuration.indexed_sources}
    live_map = {b.live_access_binding_id: b for b in configuration.live_access_bindings}
    for binding_id in indexed_ids:
        binding = indexed_map.get(binding_id)
        if binding is None or binding.status is not WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE:
            raise _projection_incomplete()
    for binding_id in live_ids:
        binding = live_map.get(binding_id)
        if binding is None or binding.status is not LiveAccessBindingStatusV1.UNAVAILABLE:
            raise _projection_incomplete()
