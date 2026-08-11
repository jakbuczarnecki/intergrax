# © Artur Czarnecki. All rights reserved.

"""Durable conversation ingress bootstrap for Slack personal DM (no onboarding persistence)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationContextBindingStatus,
    ConversationContextBindingV1,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationThreadContextPolicy,
    ConversationWorkspaceResolutionPolicy,
    WorkspaceConversationAudience,
    WorkspaceConversationAudiencePolicyV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
)


_PRE_WORKSPACE_PLACEHOLDER_ID = "workspace.unselected"


def pre_workspace_placeholder_id() -> str:
    """Stable placeholder workspace ref for pre-selection execution context."""
    return _PRE_WORKSPACE_PLACEHOLDER_ID


def conversation_context_binding_id_for_ingress(
    *,
    tenant_id: str,
    conversation_connection_ref: str,
    opaque_conversation_ref: str,
) -> str:
    canonical = "\x1f".join(
        (
            tenant_id.strip(),
            conversation_connection_ref.strip(),
            opaque_conversation_ref.strip(),
        )
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:40]
    return f"ctx.{digest}"


class ConversationIngressBootstrapService:
    """Ensures durable conversation binding exists for one personal ingress."""

    def __init__(
        self,
        repository: ConversationContextRepository,
        *,
        clock: Callable[[], datetime],
        frontend_provider_id: str = "slack",
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._frontend_provider_id = frontend_provider_id.strip() or "slack"

    def ensure_personal_binding(
        self,
        *,
        tenant_id: str,
        ingress: ConversationIngressContextV1,
    ) -> ConversationContextBindingV1:
        if ingress.observed_audience is not ConversationObservedAudience.PERSONAL:
            raise ValueError("personal_ingress_required")

        binding_id = conversation_context_binding_id_for_ingress(
            tenant_id=tenant_id,
            conversation_connection_ref=ingress.conversation_connection_ref,
            opaque_conversation_ref=ingress.opaque_conversation_ref,
        )
        existing = self._repository.get_binding(
            tenant_id=tenant_id,
            conversation_context_binding_id=binding_id,
        )
        if existing is not None:
            return existing

        now = self._clock()
        binding = ConversationContextBindingV1(
            conversation_context_binding_id=binding_id,
            tenant_id=tenant_id.strip(),
            conversation_connection_ref=ingress.conversation_connection_ref,
            frontend_provider_id=self._frontend_provider_id,
            opaque_conversation_ref=ingress.opaque_conversation_ref,
            audience_mode=ConversationAudienceMode.PERSONAL,
            workspace_resolution_policy=ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION,
            workspace_id=None,
            owner_principal_ref=ingress.actor_principal_ref,
            activation_policy=ConversationActivationPolicy.ALWAYS,
            thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
            administrative_status=ConversationContextBindingStatus.ACTIVE,
            configuration_version=1,
            created_at=now,
            updated_at=now,
        )
        try:
            inserted = self._repository.put_binding_if_absent(binding)
        except ConversationContextRepositoryError:
            inserted = False
        if inserted:
            return binding
        loaded = self._repository.get_binding(
            tenant_id=tenant_id,
            conversation_context_binding_id=binding_id,
        )
        if loaded is None:
            raise RuntimeError("conversation_binding_bootstrap_failed")
        return loaded

    def ensure_workspace_audience_policy(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> None:
        workspace = workspace_id.strip()
        tenant = tenant_id.strip()
        if not tenant or not workspace:
            return
        existing = self._repository.get_workspace_audience_policy(
            tenant_id=tenant,
            workspace_id=workspace,
        )
        if existing is not None:
            return
        now = self._clock()
        policy = WorkspaceConversationAudiencePolicyV1(
            tenant_id=tenant,
            workspace_id=workspace,
            audience=WorkspaceConversationAudience.PERSONAL,
            configuration_version=1,
            updated_at=now,
        )
        try:
            self._repository.put_workspace_audience_policy_if_absent(policy)
        except ConversationContextRepositoryError:
            return


__all__ = [
    "ConversationIngressBootstrapService",
    "conversation_context_binding_id_for_ingress",
    "pre_workspace_placeholder_id",
]
