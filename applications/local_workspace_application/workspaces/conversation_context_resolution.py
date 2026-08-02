# © Artur Czarnecki. All rights reserved.

"""Deterministic Conversation Context workspace resolution (LKW-CONVERSATION-CONTEXT-1A)."""

from __future__ import annotations

from typing import Protocol

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationActivationSignal,
    ConversationAudienceMode,
    ConversationContextBindingStatus,
    ConversationContextBindingV1,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationWorkspaceResolutionPolicy,
    ResolvedConversationWorkspaceContextV1,
    WorkspaceConversationAudience,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
)


class ConversationContextResolutionError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class ConversationConnectionAuthorizationPort(Protocol):
    def is_conversation_connection_active_and_tenant_owned(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
    ) -> bool:
        """Return whether the conversation connection is active and tenant-owned."""


class WorkspaceAuthorizationPort(Protocol):
    def is_workspace_active(self, *, tenant_id: str, workspace_id: str) -> bool:
        """Return whether the workspace exists and remains active."""

    def may_principal_use_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_ref: str,
    ) -> bool:
        """Return whether the principal may use the workspace."""


def _activation_allowed(
    policy: ConversationActivationPolicy,
    signal: ConversationActivationSignal,
) -> bool:
    if policy is ConversationActivationPolicy.ALWAYS:
        return True
    if policy is ConversationActivationPolicy.MENTION_ONLY:
        return signal in (
            ConversationActivationSignal.MENTION,
            ConversationActivationSignal.THREAD_CONTINUATION,
        )
    if policy is ConversationActivationPolicy.EXPLICIT_COMMAND:
        return signal is ConversationActivationSignal.EXPLICIT_COMMAND
    return False


def _observed_to_audience_mode(
    observed: ConversationObservedAudience,
) -> ConversationAudienceMode:
    if observed is ConversationObservedAudience.PERSONAL:
        return ConversationAudienceMode.PERSONAL
    return ConversationAudienceMode.SHARED


class ConversationContextResolver:
    def __init__(
        self,
        repository: ConversationContextRepository,
        *,
        connection_port: ConversationConnectionAuthorizationPort,
        workspace_port: WorkspaceAuthorizationPort,
    ) -> None:
        self._repository = repository
        self._connection_port = connection_port
        self._workspace_port = workspace_port

    def resolve(
        self,
        *,
        tenant_id: str,
        ingress: ConversationIngressContextV1,
    ) -> ResolvedConversationWorkspaceContextV1:
        if ingress.observed_audience is ConversationObservedAudience.UNKNOWN:
            raise ConversationContextResolutionError("OBSERVED_AUDIENCE_UNKNOWN")

        bindings = self._repository.list_bindings_for_semantic_identity(
            tenant_id=tenant_id,
            conversation_connection_ref=ingress.conversation_connection_ref,
            opaque_conversation_ref=ingress.opaque_conversation_ref,
        )
        active_bindings = [
            binding
            for binding in bindings
            if binding.administrative_status is ConversationContextBindingStatus.ACTIVE
        ]
        if not active_bindings:
            raise ConversationContextResolutionError("NO_ACTIVE_BINDING")
        if len(active_bindings) > 1:
            raise ConversationContextResolutionError("AMBIGUOUS_ACTIVE_BINDING")
        binding = active_bindings[0]

        if not self._connection_port.is_conversation_connection_active_and_tenant_owned(
            tenant_id=tenant_id,
            conversation_connection_ref=ingress.conversation_connection_ref,
        ):
            raise ConversationContextResolutionError("CONVERSATION_CONNECTION_UNAVAILABLE")

        observed_mode = _observed_to_audience_mode(ingress.observed_audience)
        if binding.audience_mode != observed_mode:
            raise ConversationContextResolutionError("AUDIENCE_MISMATCH")

        if binding.audience_mode is ConversationAudienceMode.PERSONAL:
            if binding.owner_principal_ref != ingress.actor_principal_ref:
                raise ConversationContextResolutionError("PERSONAL_PRINCIPAL_MISMATCH")
        elif binding.owner_principal_ref is not None:
            raise ConversationContextResolutionError("PERSONAL_PRINCIPAL_MISMATCH")

        if not _activation_allowed(binding.activation_policy, ingress.activation_signal):
            raise ConversationContextResolutionError("ACTIVATION_NOT_ALLOWED")

        workspace_id = self._resolve_workspace_id(
            tenant_id=tenant_id,
            binding=binding,
            actor_principal_ref=ingress.actor_principal_ref,
        )

        if not self._workspace_port.is_workspace_active(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            raise ConversationContextResolutionError("WORKSPACE_UNAVAILABLE")

        audience_policy = self._repository.get_workspace_audience_policy(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if audience_policy is None:
            raise ConversationContextResolutionError("WORKSPACE_AUDIENCE_INCOMPATIBLE")

        if binding.audience_mode is ConversationAudienceMode.SHARED:
            if audience_policy.audience is not WorkspaceConversationAudience.SHARED:
                raise ConversationContextResolutionError("WORKSPACE_AUDIENCE_INCOMPATIBLE")
        elif audience_policy.audience not in (
            WorkspaceConversationAudience.PERSONAL,
            WorkspaceConversationAudience.SHARED,
        ):
            raise ConversationContextResolutionError("WORKSPACE_AUDIENCE_INCOMPATIBLE")

        if not self._workspace_port.may_principal_use_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_ref=ingress.actor_principal_ref,
        ):
            raise ConversationContextResolutionError("WORKSPACE_NOT_AUTHORIZED")

        return ResolvedConversationWorkspaceContextV1(
            tenant_id=tenant_id,
            conversation_context_binding_id=binding.conversation_context_binding_id,
            audience_mode=binding.audience_mode,
            workspace_id=workspace_id,
            principal_ref=ingress.actor_principal_ref,
            canonical_thread_ref=ingress.opaque_thread_ref,
            activation_policy=binding.activation_policy,
            thread_context_policy=binding.thread_context_policy,
        )

    def _resolve_workspace_id(
        self,
        *,
        tenant_id: str,
        binding: ConversationContextBindingV1,
        actor_principal_ref: str,
    ) -> str:
        if (
            binding.workspace_resolution_policy
            is ConversationWorkspaceResolutionPolicy.FIXED_WORKSPACE
        ):
            if binding.workspace_id is None:
                raise ConversationContextResolutionError("FIXED_WORKSPACE_MISSING")
            return binding.workspace_id

        if binding.owner_principal_ref is None:
            raise ConversationContextResolutionError("PERSONAL_WORKSPACE_SELECTION_MISSING")

        state = self._repository.get_personal_state(
            tenant_id=tenant_id,
            conversation_context_binding_id=binding.conversation_context_binding_id,
            owner_principal_ref=binding.owner_principal_ref,
        )
        if state is None:
            raise ConversationContextResolutionError("PERSONAL_WORKSPACE_SELECTION_MISSING")
        if state.owner_principal_ref != actor_principal_ref:
            raise ConversationContextResolutionError("PERSONAL_PRINCIPAL_MISMATCH")
        return state.selected_workspace_id
