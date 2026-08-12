# © Artur Czarnecki. All rights reserved.

"""VK-EXT-3-REVIEW-FIX-1 qualification for generic scoped-source host extension seam."""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.runtime.vendor_knowledge.adapter_composition import (
    build_default_vendor_knowledge_adapter_registry,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBindingService
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_adapter_registry,
    build_vendor_knowledge_source_plugin_registry,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceRef, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from local_workspace_application.workspaces.connected_source_discovery_atlassian import (
    ConfluenceKnownSpaceCatalog,
    JiraKnownProjectCatalog,
)
from local_workspace_application.workspaces.connected_source_discovery_google_workspace import (
    GoogleWorkspaceKnownResourceCatalog,
)
from local_workspace_application.workspaces.connected_source_materializer import (
    ConnectedSourceContentMaterializerRegistry,
    ConnectedSourceSyncSinkError,
    default_connected_source_materializer_registry,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceBindingError,
    ConnectedSourceDiscoveryError,
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
    RemoteResourceOpaqueRefCodecError,
    VendorKnowledgeScopedSourceCandidatePayload,
    _VENDOR_KNOWLEDGE_SCOPED_SOURCE_CANDIDATE_PAYLOAD_SCHEMA,
)
from local_workspace_application.workspaces.connected_source_tenant_binding import (
    ProviderNeutralConnectedSourceCandidateAdapter,
    SlackConnectedSourceCandidateAdapter,
    WorkspaceConnectedSourceTenantBindingService,
)
from local_workspace_application.workspaces.connected_source_wiring import (
    VendorKnowledgeApplicationExtensionContext,
    build_default_remote_resource_discovery_registry,
)
from local_workspace_application.workspaces.vendor_knowledge_extension_composition import (
    build_default_vendor_knowledge_application_contribution_catalog as build_app_catalog,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ACME_REFERENCE_VK_PLUGIN_PKG = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "intergrax_reference_vendor_knowledge_plugin"
)


def _install_acme_reference_vk_plugin_package() -> None:
    module_name = "acme_reference_vk_plugin"
    if importlib.util.find_spec(module_name) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(_ACME_REFERENCE_VK_PLUGIN_PKG), "-q"],
            cwd=str(_REPO_ROOT),
        )
    importlib.import_module(module_name)

_TENANT = "tenant-scoped-src"
_WORKSPACE = "workspace-scoped-src"
_CONNECTION = "conn.second-ref"
_SIGNING_KEY = "scoped-source-seam-signing-key"
_ROOT_OLDEST = "2026-01-01T00:00:00+00:00"
_ROOT_LATEST = "2026-02-01T00:00:00+00:00"

_SECOND_PROVIDER = "second_reference_provider"
_SECOND_SOURCE = "second_reference_records"
_SCOPE_TYPE = "dataset"
_SCOPE_ID = "dataset-qualification-001"
_SECOND_LABEL = "Second Reference Dataset"


@pytest.fixture
def codec() -> RemoteResourceOpaqueRefCodec:
    return RemoteResourceOpaqueRefCodec.from_signing_key_material(_SIGNING_KEY)


@pytest.fixture
def second_reference_candidate_ref(codec: RemoteResourceOpaqueRefCodec) -> str:
    return codec.encode_vendor_knowledge_scoped_source_candidate(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        provider_id=_SECOND_PROVIDER,
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
        source_kind=_SECOND_SOURCE,
        scope_id=_SCOPE_ID,
        scope_type=_SCOPE_TYPE,
        safe_display_label=_SECOND_LABEL,
    )


def _candidate_adapter(
    codec: RemoteResourceOpaqueRefCodec,
    discovery: Any | None = None,
) -> ProviderNeutralConnectedSourceCandidateAdapter:
    discovery_service = discovery or _RecordingDiscovery()
    slack = SlackConnectedSourceCandidateAdapter(
        codec=codec,
        discovery_service=discovery_service,
    )
    return ProviderNeutralConnectedSourceCandidateAdapter(
        slack=slack,
        codec=codec,
        discovery_service=discovery_service,
    )


class _RecordingDiscovery:
    def __init__(self, *, label: str = "fresh-label", error: Exception | None = None) -> None:
        self.label = label
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def revalidate_candidate_label(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.label


class _ConnectionAwareResolver:
    def __init__(self, registry: KnowledgeConnectionRegistry) -> None:
        self._registry = registry

    def resolve(self, *, source):
        return self._registry.resolve(
            tenant_id=source.tenant_id,
            connection_ref=source.connection_ref,
            provider_id=source.provider_id,
            integration_kind=source.integration_kind,
        )


def _binding_service(
    store: InMemoryDocumentStore,
    *,
    tenant_id: str,
    registry: KnowledgeConnectionRegistry,
    adapter_registry: KnowledgeAdapterRegistry | None = None,
) -> KnowledgeSourceBindingService:
    resolved_adapters = adapter_registry or build_default_vendor_knowledge_adapter_registry()
    return KnowledgeSourceBindingService(
        tenant_id=tenant_id,
        repository=DocumentStoreKnowledgeSourceBindingRepository(store),
        integration_resolver=_ConnectionAwareResolver(registry),
        adapter_registry=resolved_adapters,
    )


def _register_slack_connection(registry: KnowledgeConnectionRegistry) -> None:
    integration = SlackConversationChannelIntegration.from_backend(
        object(),  # type: ignore[arg-type]
        enabled=True,
    )
    registry.register(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=integration,
    )


def _count_application_hooks(*, discover_entry_points: bool) -> tuple[int, int]:
    context = VendorKnowledgeApplicationExtensionContext(
        connection_registry=KnowledgeConnectionRegistry(),
        opaque_ref_codec=RemoteResourceOpaqueRefCodec.from_signing_key_material(_SIGNING_KEY),
    )
    catalog = build_app_catalog(context, discover_entry_points=discover_entry_points)
    discovery = sum(len(item.discovery_contributions) for item in catalog.list_contributions())
    materializers = sum(len(item.indexed_materializers) for item in catalog.list_contributions())
    return discovery, materializers


# --- G1: scoped-source model / opaque-ref ---


def test_scoped_source_payload_schema_contract() -> None:
    payload = VendorKnowledgeScopedSourceCandidatePayload(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        resource_type=RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE,
        provider_id=_SECOND_PROVIDER,
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
        source_kind=_SECOND_SOURCE,
        scope_id=_SCOPE_ID,
        scope_type=_SCOPE_TYPE,
        safe_display_label=_SECOND_LABEL,
    )
    assert payload.schema_version == _VENDOR_KNOWLEDGE_SCOPED_SOURCE_CANDIDATE_PAYLOAD_SCHEMA
    with pytest.raises(ValueError):
        VendorKnowledgeScopedSourceCandidatePayload(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            resource_type=RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE,
            provider_id=_SECOND_PROVIDER,
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
            source_kind=_SECOND_SOURCE,
            scope_id=_SCOPE_ID,
            scope_type=_SCOPE_TYPE,
            safe_display_label=_SECOND_LABEL,
            extra_field="rejected",
        )


def test_scoped_source_opaque_ref_roundtrip(codec: RemoteResourceOpaqueRefCodec) -> None:
    token = codec.encode_vendor_knowledge_scoped_source_candidate(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        provider_id=_SECOND_PROVIDER,
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
        source_kind=_SECOND_SOURCE,
        scope_id=_SCOPE_ID,
        scope_type=_SCOPE_TYPE,
        safe_display_label=_SECOND_LABEL,
    )
    payload = codec.decode_vendor_knowledge_scoped_source_candidate(token)
    assert payload.provider_id == _SECOND_PROVIDER
    assert payload.source_kind == _SECOND_SOURCE
    assert payload.scope_type == _SCOPE_TYPE
    assert payload.scope_id == _SCOPE_ID
    assert payload.safe_display_label == _SECOND_LABEL
    assert payload.resource_type is RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE
    assert "api_key" not in token
    assert "secret" not in token.lower()


def test_scoped_source_opaque_ref_tampering_rejected(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    tampered = second_reference_candidate_ref[:-1] + (
        "A" if second_reference_candidate_ref[-1] != "A" else "B"
    )
    with pytest.raises(ConnectedSourceDiscoveryError):
        codec.decode_vendor_knowledge_scoped_source_candidate(tampered)


def test_scoped_source_opaque_ref_wrong_signing_key_rejected(
    second_reference_candidate_ref: str,
) -> None:
    other_codec = RemoteResourceOpaqueRefCodec.from_signing_key_material("other-signing-key")
    with pytest.raises(ConnectedSourceDiscoveryError):
        other_codec.decode_vendor_knowledge_scoped_source_candidate(second_reference_candidate_ref)


def test_scoped_source_opaque_ref_wrong_schema_and_resource_type_rejected(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    padding = "=" * (-len(second_reference_candidate_ref) % 4)
    raw = base64.urlsafe_b64decode(second_reference_candidate_ref + padding)
    envelope = json.loads(raw.decode("utf-8"))
    envelope["payload"]["schema_version"] = "lkw.vendor_knowledge_scoped_source_candidate.v0"
    mutated = base64.urlsafe_b64encode(json.dumps(envelope).encode("utf-8")).decode("ascii").rstrip("=")
    with pytest.raises(ConnectedSourceDiscoveryError):
        codec.decode_vendor_knowledge_scoped_source_candidate(mutated)

    envelope = json.loads(raw.decode("utf-8"))
    envelope["payload"]["resource_type"] = RemoteResourceTypeV1.JIRA_PROJECT.value
    mutated_type = base64.urlsafe_b64encode(json.dumps(envelope).encode("utf-8")).decode("ascii").rstrip("=")
    with pytest.raises(ConnectedSourceDiscoveryError):
        codec.decode_vendor_knowledge_scoped_source_candidate(mutated_type)


def test_scoped_source_opaque_ref_malformed_and_oversized_rejected(
    codec: RemoteResourceOpaqueRefCodec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ConnectedSourceDiscoveryError):
        codec.decode_vendor_knowledge_scoped_source_candidate("not-valid-base64!!!")

    monkeypatch.setattr(
        "local_workspace_application.workspaces.connected_source_opaque_ref_codec._MAX_TOKEN_LEN",
        16,
    )
    with pytest.raises(RemoteResourceOpaqueRefCodecError, match="opaque_ref_token_too_large"):
        codec.encode_vendor_knowledge_scoped_source_candidate(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            provider_id=_SECOND_PROVIDER,
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
            source_kind=_SECOND_SOURCE,
            scope_id=_SCOPE_ID,
            scope_type=_SCOPE_TYPE,
            safe_display_label=_SECOND_LABEL,
        )


def test_second_synthetic_identity_encode_decode_and_binding(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    adapter = _candidate_adapter(codec)
    binding = adapter.build_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        opaque_candidate_ref=second_reference_candidate_ref,
        root_oldest=_ROOT_OLDEST,
        root_latest=_ROOT_LATEST,
    )
    assert binding.provider_id == _SECOND_PROVIDER
    assert binding.integration_kind is IntegrationCategory.WIKI_KNOWLEDGE
    assert binding.source_kind == _SECOND_SOURCE
    assert binding.scope.remote_scope_type == _SCOPE_TYPE
    assert binding.scope.remote_scope_id == _SCOPE_ID
    assert binding.safe_display_name == _SECOND_LABEL
    assert binding.tenant_id == _TENANT
    assert binding.connection_ref == _CONNECTION


# --- G2: binding / security fences ---


@pytest.mark.parametrize(
    ("tenant_id", "workspace_id", "connection_ref", "error_code"),
    [
        ("other-tenant", _WORKSPACE, _CONNECTION, "workspace_not_found"),
        (_TENANT, "other-workspace", _CONNECTION, "workspace_not_found"),
        (_TENANT, _WORKSPACE, "other-connection", "connection_not_attached"),
    ],
)
def test_scoped_source_binding_ownership_fence(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    error_code: str,
) -> None:
    adapter = _candidate_adapter(codec)
    with pytest.raises(ConnectedSourceBindingError) as exc:
        adapter.build_binding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            opaque_candidate_ref=second_reference_candidate_ref,
            root_oldest=_ROOT_OLDEST,
            root_latest=_ROOT_LATEST,
        )
    assert exc.value.error_code == error_code


def test_scoped_source_unknown_integration_category_fails_closed(
    codec: RemoteResourceOpaqueRefCodec,
) -> None:
    token = codec.encode_vendor_knowledge_scoped_source_candidate(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        provider_id=_SECOND_PROVIDER,
        integration_kind="not_a_real_category",
        source_kind=_SECOND_SOURCE,
        scope_id=_SCOPE_ID,
        scope_type=_SCOPE_TYPE,
        safe_display_label=_SECOND_LABEL,
    )
    adapter = _candidate_adapter(codec)
    with pytest.raises(ConnectedSourceBindingError, match="candidate_inaccessible"):
        adapter.build_binding(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            opaque_candidate_ref=token,
            root_oldest=_ROOT_OLDEST,
            root_latest=_ROOT_LATEST,
        )


def test_scoped_source_provider_and_category_mismatch_fail_on_persist(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    store = InMemoryDocumentStore()
    registry = KnowledgeConnectionRegistry()
    _register_slack_connection(registry)
    adapter = _candidate_adapter(codec)
    binding = adapter.build_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        opaque_candidate_ref=second_reference_candidate_ref,
        root_oldest=_ROOT_OLDEST,
        root_latest=_ROOT_LATEST,
    )
    service = WorkspaceConnectedSourceTenantBindingService(
        lambda tenant_id: _binding_service(store, tenant_id=tenant_id, registry=registry)
    )
    with pytest.raises(ConnectedSourceBindingError):
        service.create_or_get_equivalent(binding)


def test_scoped_source_unknown_source_kind_fails_on_persist(
    codec: RemoteResourceOpaqueRefCodec,
) -> None:
    token = codec.encode_vendor_knowledge_scoped_source_candidate(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL.value,
        source_kind="unknown_source_kind",
        scope_id=_SCOPE_ID,
        scope_type=_SCOPE_TYPE,
        safe_display_label=_SECOND_LABEL,
    )
    store = InMemoryDocumentStore()
    registry = KnowledgeConnectionRegistry()
    _register_slack_connection(registry)
    adapter = _candidate_adapter(codec)
    binding = adapter.build_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        opaque_candidate_ref=token,
        root_oldest=_ROOT_OLDEST,
        root_latest=_ROOT_LATEST,
    )
    binding_service = _binding_service(store, tenant_id=_TENANT, registry=registry)
    with pytest.raises(VendorKnowledgeError) as exc:
        binding_service.create_or_get_equivalent(binding)
    assert exc.value.code is VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND


@pytest.mark.asyncio
async def test_scoped_source_label_revalidation_uses_provider_owned_discovery(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    discovery = _RecordingDiscovery(label="updated-safe-label")
    adapter = _candidate_adapter(codec, discovery=discovery)
    label = await adapter.revalidate_candidate_label(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        opaque_candidate_ref=second_reference_candidate_ref,
    )
    assert label == "updated-safe-label"
    assert len(discovery.calls) == 1
    assert discovery.calls[0]["resource_type"] is RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE


@pytest.mark.asyncio
async def test_scoped_source_label_revalidation_ownership_fence(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    adapter = _candidate_adapter(codec)
    with pytest.raises(ConnectedSourceBindingError, match="workspace_not_found"):
        await adapter.revalidate_candidate_label(
            tenant_id="other-tenant",
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            opaque_candidate_ref=second_reference_candidate_ref,
        )


@pytest.mark.asyncio
async def test_scoped_source_label_revalidation_inaccessible_resource(
    codec: RemoteResourceOpaqueRefCodec,
    second_reference_candidate_ref: str,
) -> None:
    discovery = _RecordingDiscovery(
        error=ConnectedSourceDiscoveryError("candidate_inaccessible"),
    )
    adapter = _candidate_adapter(codec, discovery=discovery)
    with pytest.raises(ConnectedSourceDiscoveryError, match="candidate_inaccessible"):
        await adapter.revalidate_candidate_label(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            opaque_candidate_ref=second_reference_candidate_ref,
        )


# --- G3: materializer discovery passthrough ---


def test_builtin_parity_with_external_discovery_disabled() -> None:
    disabled_catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=False,
    )
    assert len(build_vendor_knowledge_adapter_registry(disabled_catalog).registered_keys()) == 12
    assert len(build_vendor_knowledge_source_plugin_registry(disabled_catalog).list_plugins()) == 12
    assert sum(len(item.connection_factories) for item in disabled_catalog.list_contributions()) == 6
    discovery_count, materializer_count = _count_application_hooks(discover_entry_points=False)
    assert discovery_count == 10
    assert materializer_count == 10
    assert materializer_count == len(
        default_connected_source_materializer_registry(discover_entry_points=False)._by_runtime_ref
    )
    assert len(build_vendor_knowledge_live_registration_registry(discover_entry_points=False).list_registrations()) == 8


def test_external_discovery_enabled_adds_reference_hooks() -> None:
    _install_acme_reference_vk_plugin_package()
    enabled_catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=True,
    )
    assert len(build_vendor_knowledge_adapter_registry(enabled_catalog).registered_keys()) == 13
    assert len(build_vendor_knowledge_source_plugin_registry(enabled_catalog).list_plugins()) == 13
    assert sum(len(item.connection_factories) for item in enabled_catalog.list_contributions()) == 7
    discovery_count, materializer_count = _count_application_hooks(discover_entry_points=True)
    assert discovery_count == 11
    assert materializer_count == 11
    assert materializer_count == len(
        default_connected_source_materializer_registry(discover_entry_points=True)._by_runtime_ref
    )
    live_disabled = build_vendor_knowledge_live_registration_registry(discover_entry_points=False)
    live_enabled = build_vendor_knowledge_live_registration_registry(discover_entry_points=True)
    assert len(live_disabled.list_registrations()) == len(live_enabled.list_registrations())


def test_discovery_registry_respects_entry_point_flag() -> None:
    codec = RemoteResourceOpaqueRefCodec.from_signing_key_material(_SIGNING_KEY)
    registry = KnowledgeConnectionRegistry()
    disabled = build_default_remote_resource_discovery_registry(
        connection_registry=registry,
        opaque_ref_codec=codec,
        google_known_resource_catalog=GoogleWorkspaceKnownResourceCatalog(),
        jira_known_project_catalog=JiraKnownProjectCatalog(),
        confluence_known_space_catalog=ConfluenceKnownSpaceCatalog(),
        msgraph_mailbox_user_id=None,
        discover_entry_points=False,
    )
    enabled = build_default_remote_resource_discovery_registry(
        connection_registry=registry,
        opaque_ref_codec=codec,
        google_known_resource_catalog=GoogleWorkspaceKnownResourceCatalog(),
        jira_known_project_catalog=JiraKnownProjectCatalog(),
        confluence_known_space_catalog=ConfluenceKnownSpaceCatalog(),
        msgraph_mailbox_user_id=None,
        discover_entry_points=True,
    )
    assert len(disabled._strategies) == 10
    assert len(enabled._strategies) == 11


def test_materializer_registry_runtime_ref_resolution_enforced() -> None:
    registry = default_connected_source_materializer_registry(discover_entry_points=False)
    source = KnowledgeSourceRef(
        tenant_id=_TENANT,
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        connection_ref=_CONNECTION,
        scope=KnowledgeSourceScope(
            remote_scope_id="mailbox/folder",
            remote_scope_type="mail_folder",
            safe_display_name="Inbox",
            parameters={},
        ),
    )
    resolved = registry.resolve(source)
    assert resolved.runtime_ref == "indexed-source:ms365_graph:mail"
    wrong_source = source.model_copy(update={"source_kind": "unknown_mail"})
    with pytest.raises(ConnectedSourceSyncSinkError):
        registry.resolve(wrong_source)


def test_materializer_registry_duplicate_runtime_ref_conflicts_fail_closed() -> None:
    registry = default_connected_source_materializer_registry(discover_entry_points=False)
    first = next(iter(registry._by_runtime_ref.values()))
    second = next(
        item
        for item in registry._by_runtime_ref.values()
        if item.runtime_ref != first.runtime_ref
    )

    class _AliasMaterializer:
        identity = second.identity
        runtime_ref = first.runtime_ref
        schema_name = f"{second.schema_name}.alias"

        def materialize(self, **kwargs):
            raise NotImplementedError

    with pytest.raises(
        ConnectedSourceSyncSinkError,
        match="connected_source_materializer_runtime_duplicate",
    ):
        ConnectedSourceContentMaterializerRegistry(
            materializers=(
                first,
                _AliasMaterializer(),
            ),
        )
