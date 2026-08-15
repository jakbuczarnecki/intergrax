from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_reference_resolver import (
    ConversationInteractionReferenceResolver,
    ConversationReferenceResolutionError,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)


class WorkspaceAuthorityFake:
    def __init__(self) -> None:
        self.items = {
            "w1": SimpleNamespace(workspace_id="w1", name="Alpha"),
            "w2": SimpleNamespace(workspace_id="w2", name="Beta"),
        }
        self.get_calls: list[tuple[str, str]] = []

    def list_workspaces(self, *, tenant_id: str) -> list[object]:
        return list(self.items.values())

    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> object | None:
        self.get_calls.append((tenant_id, workspace_id))
        return self.items.get(workspace_id) if tenant_id == "tenant-1" else None


def _context() -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id="tenant-1",
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="w1",
        principal_ref="principal-1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(
            {ConversationProductCapability.WORKSPACE_DISCOVERY}
        ),
    )


def _resolver(authority: WorkspaceAuthorityFake) -> ConversationInteractionReferenceResolver:
    return ConversationInteractionReferenceResolver(
        planning_request=ConversationPlanningRequest(
            message_text="Alpha Beta",
            available_workspaces=(
                ConversationPlanningWorkspace(workspace_id="w2", name="Beta", is_active=False),
                ConversationPlanningWorkspace(workspace_id="w1", name="Alpha", is_active=True),
            ),
        ),
        execution_context=_context(),
        workspace_service=authority,  # type: ignore[arg-type]
    )


def test_active_name_ordinal_and_created_references_revalidate_authority() -> None:
    authority = WorkspaceAuthorityFake()
    resolver = _resolver(authority)

    assert resolver.resolve_workspace(_ref("active")).workspace_id == "w1"
    assert resolver.resolve_workspace(_ref("name", "alpha")).workspace_id == "w1"
    assert resolver.resolve_workspace(_ref("ordinal", "1")).workspace_id == "w2"
    assert resolver.resolve_workspace(
        _ref("created_by_action", "create-1"),
        created_workspace_ids={"create-1": "w2"},
    ).workspace_id == "w2"
    assert authority.get_calls == [
        ("tenant-1", "w1"),
        ("tenant-1", "w1"),
        ("tenant-1", "w2"),
        ("tenant-1", "w2"),
    ]


def test_name_ambiguity_and_cross_tenant_state_fail_closed() -> None:
    authority = WorkspaceAuthorityFake()
    authority.items["w3"] = SimpleNamespace(workspace_id="w3", name="Alpha")
    resolver = _resolver(authority)

    with pytest.raises(ConversationReferenceResolutionError) as ambiguous:
        resolver.resolve_workspace(_ref("name", "alpha"))
    assert ambiguous.value.code == "workspace_reference_ambiguous"

    authority.items.pop("w1")
    with pytest.raises(ConversationReferenceResolutionError) as missing:
        resolver.resolve_workspace(_ref("active"))
    assert missing.value.code == "workspace_not_found"


def test_deleted_active_workspace_becomes_unavailable() -> None:
    authority = WorkspaceAuthorityFake()
    resolver = _resolver(authority)
    resolver.clear_active_workspace()

    with pytest.raises(ConversationReferenceResolutionError) as missing:
        resolver.resolve_workspace(_ref("active"))
    assert missing.value.code == "active_workspace_required"


def _ref(kind: str, value: str | None = None) -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind(kind), value=value)
