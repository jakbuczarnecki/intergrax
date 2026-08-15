# © Artur Czarnecki. All rights reserved.

"""Durable pending-authorization context for conversational tenant connections."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connections import SafeTenantConnectionV1
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationConnectionAuthContextV1,
    ConversationExecutionContextV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
)
from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    TenantConnectionPendingManualAuthorizationV1,
    TenantConnectionPlanningConnectionV1,
    TenantConnectionPlanningProviderV1,
    TenantConnectionPlanningSnapshotV1,
)
from local_workspace_application.workspaces.tenant_connection_product_orchestration import (
    TenantConnectionProductOrchestrationFactory,
    TenantConnectionProductOrchestrationService,
)


class ConversationConnectionAuthContextError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class TenantConnectionConversationConfig:
    oauth_redirect_uri: str | None = None


def build_tenant_connection_planning_snapshot(
    service: TenantConnectionProductOrchestrationService,
    *,
    pending_manual_authorization: TenantConnectionPendingManualAuthorizationV1 | None,
) -> TenantConnectionPlanningSnapshotV1:
    providers = tuple(
        TenantConnectionPlanningProviderV1(
            provider_id=str(item["provider_id"]),
            safe_display_name=str(item["safe_display_name"]),
            auth_mode=str(item["auth_mode"]),
            qualification=str(item["qualification"]),
        )
        for item in service.list_supported_connection_providers()
    )
    connections = tuple(
        TenantConnectionPlanningConnectionV1(
            connection_ref=connection.connection_ref,
            provider_id=connection.provider_id,
            safe_display_name=connection.safe_display_name,
            administrative_status=connection.administrative_status.value,
            connected_principal_ref=connection.connected_principal_ref,
            configuration_version=connection.configuration_version,
        )
        for connection in service.list_connections()
    )
    return TenantConnectionPlanningSnapshotV1(
        providers=providers,
        connections=connections,
        pending_manual_authorization=pending_manual_authorization,
    )


def parse_manual_credential_payload(message_text: str) -> Mapping[str, JsonValue]:
    cleaned = message_text.strip()
    if not cleaned:
        raise ValueError("credential_binding_invalid")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("credential_binding_invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError("credential_binding_invalid")
    app_token = parsed.get("app_token")
    bot_token = parsed.get("bot_token")
    if not isinstance(app_token, str) or not app_token.strip():
        raise ValueError("credential_binding_invalid")
    if not isinstance(bot_token, str) or not bot_token.strip():
        raise ValueError("credential_binding_invalid")
    return {
        "app_token": app_token.strip(),
        "bot_token": bot_token.strip(),
    }


class ConversationConnectionAuthContextService:
    def __init__(
        self,
        *,
        context_repository: ConversationContextRepository,
        orchestration_factory: TenantConnectionProductOrchestrationFactory,
        clock: Callable[[], datetime] | None = None,
        max_conflict_retries: int = 3,
    ) -> None:
        if isinstance(max_conflict_retries, bool) or not 1 <= max_conflict_retries <= 3:
            raise ValueError("max_conflict_retries must be between 1 and 3")
        self._context_repository = context_repository
        self._orchestration_factory = orchestration_factory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._max_conflict_retries = max_conflict_retries

    def orchestration_for(self, tenant_id: str) -> TenantConnectionProductOrchestrationService:
        return self._orchestration_factory.for_tenant(tenant_id)

    def build_planning_snapshot(
        self,
        *,
        tenant_id: str,
        context: ConversationExecutionContextV1,
    ) -> TenantConnectionPlanningSnapshotV1 | None:
        service = self.orchestration_for(tenant_id)
        pending = self._pending_manual_authorization(context)
        return build_tenant_connection_planning_snapshot(
            service,
            pending_manual_authorization=pending,
        )

    def record_pending_authorization(
        self,
        *,
        context: ConversationExecutionContextV1,
        authorization_transaction_ref: str,
        provider_id: str,
        required_user_action: str,
    ) -> None:
        now = self._clock()
        replacement = ConversationConnectionAuthContextV1(
            tenant_id=context.tenant_id,
            conversation_context_binding_id=context.conversation_context_binding_id,
            authorization_transaction_ref=authorization_transaction_ref.strip(),
            provider_id=provider_id.strip(),
            required_user_action=required_user_action.strip(),
            configuration_version=1,
            updated_at=now,
        )
        try:
            current = self._context_repository.get_connection_auth_context(
                tenant_id=context.tenant_id,
                conversation_context_binding_id=context.conversation_context_binding_id,
            )
        except ConversationContextRepositoryError as exc:
            raise ConversationConnectionAuthContextError(
                "conversation_context_storage_unavailable"
            ) from exc

        if current is None:
            try:
                if not self._context_repository.put_connection_auth_context_if_absent(
                    replacement
                ):
                    current = self._context_repository.get_connection_auth_context(
                        tenant_id=context.tenant_id,
                        conversation_context_binding_id=context.conversation_context_binding_id,
                    )
            except ConversationContextRepositoryError as exc:
                raise ConversationConnectionAuthContextError(
                    "conversation_context_storage_unavailable"
                ) from exc

        if current is None:
            return

        for _ in range(self._max_conflict_retries):
            next_replacement = replacement.model_copy(
                update={
                    "configuration_version": current.configuration_version + 1,
                    "updated_at": self._clock(),
                }
            )
            try:
                if self._context_repository.replace_connection_auth_context_if_match(
                    expected=current,
                    replacement=next_replacement,
                ):
                    return
            except ConversationContextRepositoryError as exc:
                raise ConversationConnectionAuthContextError(
                    "conversation_context_storage_unavailable"
                ) from exc
            try:
                current = self._context_repository.get_connection_auth_context(
                    tenant_id=context.tenant_id,
                    conversation_context_binding_id=context.conversation_context_binding_id,
                )
            except ConversationContextRepositoryError as exc:
                raise ConversationConnectionAuthContextError(
                    "conversation_context_storage_unavailable"
                ) from exc
            if current is None:
                return

    def clear_pending_authorization(
        self,
        *,
        context: ConversationExecutionContextV1,
    ) -> None:
        try:
            self._context_repository.delete_connection_auth_context(
                tenant_id=context.tenant_id,
                conversation_context_binding_id=context.conversation_context_binding_id,
            )
        except ConversationContextRepositoryError as exc:
            raise ConversationConnectionAuthContextError(
                "conversation_context_storage_unavailable"
            ) from exc

    def require_pending_manual_authorization(
        self,
        *,
        context: ConversationExecutionContextV1,
    ) -> ConversationConnectionAuthContextV1:
        pending = self._pending_auth_context(context)
        if pending is None:
            raise ConversationConnectionAuthContextError(
                "tenant_connection_authorization_pending_not_found"
            )
        if pending.required_user_action != "present_manual_instructions":
            raise ConversationConnectionAuthContextError(
                "tenant_connection_authorization_pending_not_found"
            )
        return pending

    def _pending_manual_authorization(
        self,
        context: ConversationExecutionContextV1,
    ) -> TenantConnectionPendingManualAuthorizationV1 | None:
        pending = self._pending_auth_context(context)
        if pending is None:
            return None
        if pending.required_user_action != "present_manual_instructions":
            return None
        return TenantConnectionPendingManualAuthorizationV1(
            authorization_transaction_ref=pending.authorization_transaction_ref,
            provider_id=pending.provider_id,
        )

    def _pending_auth_context(
        self,
        context: ConversationExecutionContextV1,
    ) -> ConversationConnectionAuthContextV1 | None:
        try:
            return self._context_repository.get_connection_auth_context(
                tenant_id=context.tenant_id,
                conversation_context_binding_id=context.conversation_context_binding_id,
            )
        except ConversationContextRepositoryError as exc:
            raise ConversationConnectionAuthContextError(
                "conversation_context_storage_unavailable"
            ) from exc


def safe_connection_payload(connection: SafeTenantConnectionV1) -> dict[str, Any]:
    return {
        "connection_ref": connection.connection_ref,
        "provider_id": connection.provider_id,
        "safe_display_name": connection.safe_display_name,
        "administrative_status": connection.administrative_status.value,
        "connected_principal_ref": connection.connected_principal_ref,
        "configuration_version": connection.configuration_version,
    }


__all__ = [
    "ConversationConnectionAuthContextError",
    "ConversationConnectionAuthContextService",
    "TenantConnectionConversationConfig",
    "build_tenant_connection_planning_snapshot",
    "parse_manual_credential_payload",
    "safe_connection_payload",
]
