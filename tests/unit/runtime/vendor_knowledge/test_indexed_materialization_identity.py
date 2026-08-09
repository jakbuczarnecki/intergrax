from __future__ import annotations

import pytest

from applications.local_workspace_application.workspaces.connected_source_ids import (
    connected_document_id,
    connected_logical_path,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    build_materialized_connected_source_document,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

pytestmark = pytest.mark.unit


def _source() -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-a",
        provider_id="provider-a",
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind="slack_conversation",
        connection_ref="connection-a",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope-a",
            remote_scope_type="conversation",
            safe_display_name="Conversation",
        ),
    )


@pytest.mark.parametrize(
    (
        "tenant_id",
        "workspace_id",
        "binding_id",
        "source_id",
        "remote_id",
    ),
    (
        (
            "tenant-a",
            "workspace-a",
            "binding-a",
            "source-a",
            "message-42",
        ),
        (
            "tenant-łódź",
            "zespół-kraków",
            "wiązanie-źródła",
            "źródło-łódź",
            "wiadomość-42",
        ),
    ),
)
def test_public_materialization_identity_matches_legacy_canonical_contract(
    tenant_id: str,
    workspace_id: str,
    binding_id: str,
    source_id: str,
    remote_id: str,
) -> None:
    identity = VendorKnowledgeSourceIdentity(
        provider_id="provider-a",
        integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind="slack_conversation",
    )

    materialized = build_materialized_connected_source_document(
        identity=identity,
        source=_source(),
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        binding_id=binding_id,
        source_id=source_id,
        remote_id=remote_id,
        markdown="message body",
        safe_file_name="message.md",
        revision=None,
        permissions=None,
    )

    assert materialized.document_id == connected_document_id(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        provider_id=identity.provider_id,
        integration_kind=identity.integration_category.value,
        source_kind=identity.source_kind,
        binding_id=binding_id,
        remote_id=remote_id,
    )
    assert materialized.logical_source_path == connected_logical_path(
        source_id=source_id,
        remote_id=remote_id,
        source_kind=identity.source_kind,
    )
