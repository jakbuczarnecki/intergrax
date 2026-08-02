# © Artur Czarnecki. All rights reserved.

"""Workspace Connection Attachment domain service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import TenantConnectionPort
from intergrax.runtime.vendor_knowledge.tenant_connections import TenantConnectionAdministrativeStatus
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationIntent,
    connection_attachment_id,
    connection_attachment_request_hash,
    connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
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

_AUTHORIZATION_RE = re.compile(r"authorization\s*[:=]", re.IGNORECASE)
_BEARER_RE = re.compile(r"bearer\s+\S", re.IGNORECASE)
_API_KEY_RE = re.compile(r"api[_-]?key\s*[=:]", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_SECRET_QUERY_KEYS = frozenset(
    {
        "api_key",
        "api-key",
        "apikey",
        "secret",
        "token",
        "password",
        "access_token",
        "refresh_token",
        "client_secret",
    }
)


class WorkspaceConnectionAttachmentError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class AttachWorkspaceConnectionCommand:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    expected_revision: int
    idempotency_key_hash: str
    requested_safe_display_label: str | None = None


@dataclass(frozen=True, slots=True)
class AttachWorkspaceConnectionResult:
    attachment: WorkspaceConnectionAttachment
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1


def _trim_url_suffix(url: str) -> str:
    return url.rstrip(".,;:!?)]}")


def _url_has_credential_material(url: str) -> bool:
    trimmed = _trim_url_suffix(url)
    parsed = urlparse(trimmed)
    if parsed.username or parsed.password:
        return True
    query = parse_qs(parsed.query, keep_blank_values=True)
    return any(key.casefold() in _SECRET_QUERY_KEYS for key in query)


def _label_contains_credential_material(value: str) -> bool:
    if _AUTHORIZATION_RE.search(value) is not None:
        return True
    if _BEARER_RE.search(value) is not None:
        return True
    if _API_KEY_RE.search(value) is not None:
        return True
    for url in _URL_RE.findall(value):
        if _url_has_credential_material(url):
            return True
    return False


def _validate_safe_display_label(value: str) -> str:
    resolved = value.strip()
    if not resolved or len(resolved) > 256:
        raise WorkspaceConnectionAttachmentError("safe_display_label_invalid")
    if _label_contains_credential_material(resolved):
        raise WorkspaceConnectionAttachmentError("safe_display_label_invalid")
    return resolved


def _resolve_safe_display_label(
    *,
    requested: str | None,
    default_label: str,
) -> str:
    if requested is None:
        return _validate_safe_display_label(default_label)
    return _validate_safe_display_label(requested)


def _connection_unavailable(status: TenantConnectionAdministrativeStatus) -> bool:
    return status in {
        TenantConnectionAdministrativeStatus.DISABLED,
        TenantConnectionAdministrativeStatus.REVOKED,
    }


class WorkspaceConnectionAttachmentService:
    def __init__(
        self,
        *,
        connection_port: TenantConnectionPort,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
    ) -> None:
        self._connection_port = connection_port
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine

    def attach_connection(
        self,
        command: AttachWorkspaceConnectionCommand,
    ) -> AttachWorkspaceConnectionResult:
        tenant_id = command.tenant_id.strip()
        workspace_id = command.workspace_id.strip()
        normalized_ref = command.connection_ref.strip()
        if not normalized_ref:
            raise WorkspaceConnectionAttachmentError("connection_not_found")

        connection = self._connection_port.get_connection(
            tenant_id=tenant_id,
            connection_ref=normalized_ref,
        )
        if connection is None or connection.tenant_id != tenant_id:
            raise WorkspaceConnectionAttachmentError("connection_not_found")
        if connection.connection_ref.strip() != normalized_ref:
            raise WorkspaceConnectionAttachmentError("connection_not_found")
        if _connection_unavailable(connection.administrative_status):
            raise WorkspaceConnectionAttachmentError("connection_unavailable")
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            raise WorkspaceConnectionAttachmentError("connection_unavailable")

        safe_label = _resolve_safe_display_label(
            requested=command.requested_safe_display_label,
            default_label=connection.safe_display_name,
        )

        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceConnectionAttachmentError("workspace_not_found")

        attachment_id = connection_attachment_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=normalized_ref,
        )
        intent = AttachConnectionMutationIntent(
            attachment_id=attachment_id,
            connection_ref=normalized_ref,
            safe_display_label=safe_label,
        )
        mutation_result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=connection_attachment_request_hash(
                connection_ref=normalized_ref,
                safe_display_label=safe_label,
            ),
            semantic_identity_hash=connection_attachment_semantic_identity_hash(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=normalized_ref,
            ),
            intent=intent,
        )

        resolved_configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if resolved_configuration is None:
            raise WorkspaceConnectionAttachmentError("workspace_not_found")

        attachment = _resolve_committed_attachment(
            configuration=resolved_configuration,
            result_entity_id=mutation_result.result_entity_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=normalized_ref,
            configuration_revision=mutation_result.configuration_revision,
        )
        return AttachWorkspaceConnectionResult(
            attachment=attachment,
            configuration_revision=mutation_result.configuration_revision,
            disposition=mutation_result.disposition,
        )


def _resolve_committed_attachment(
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
        if item.tenant_id != tenant_id:
            raise WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")
        if item.workspace_id != workspace_id:
            raise WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")
        if item.connection_ref != connection_ref:
            raise WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")
        if item.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            raise WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")
        if item.effective_revision > configuration_revision:
            raise WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")
        return item

    raise WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")
