# © Artur Czarnecki. All rights reserved.

"""P1.7 — CredentialRef / late-bound credential resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from intergrax.applications._shared.profile_resolution import (
    EffectiveProfileActivationDependencies,
    EffectiveProfileActivationService,
    InMemoryActiveEffectiveProfileRevisionStore,
    InMemoryEffectiveProfileRevisionStore,
    materialize_effective_profile_revision,
    resolve_profile,
)
from intergrax.applications._shared.runtime_inspection.redaction import (
    profile_contains_no_raw_secrets,
    redacted_profile_snapshot,
    safe_effective_profile_revision_view,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution import (
    ActivateEffectiveProfileRevisionRequest,
    EffectiveProfileRevisionScope,
)
from intergrax.runtime.task.task_state import TaskState
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.credential import (
    CredentialNotFoundError,
    CredentialRef,
    CredentialResolutionContext,
    CredentialScopeMismatchError,
    ResolvedCredential,
)
from intergrax.integrations.credentials.errors import sanitize_credential_error_message
from intergrax.integrations.credentials.google_workspace import (
    GoogleWorkspaceSecretsStoreCredentialResolver,
)
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceClientFamily,
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.contracts.credential import CredentialResolutionMode
from intergrax.integrations.providers.collaboration_suite.google_workspace.tenant_connection_factory import (
    GoogleWorkspaceTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_factory import (
    SlackTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_registry import (
    TenantConnectionIntegrationFactoryRegistry,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
)
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SENTINEL = "SUPER_SECRET_P1_7_SENTINEL"
_CREDENTIAL_REF_PATH = "secrets/tenant-a/google-workspace"
_SCOPE = EffectiveProfileRevisionScope(application_id="p1.7.test", tenant_id="tenant-a")


def _assert_sentinel_absent(payload: object) -> None:
    text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    assert _SENTINEL not in text


class _CountingSecretsStore:
    def __init__(
        self,
        values: Mapping[str, str] | None = None,
        *,
        versions: Mapping[tuple[str, str | None], str] | None = None,
    ) -> None:
        self.values = dict(values or {})
        self.versions = dict(versions or {})
        self.get_secret_calls: list[tuple[str, str | None]] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.get_secret_calls.append((path.strip(), version))
        key = (path.strip(), version)
        if key in self.versions:
            return self.versions[key]
        if version is not None:
            raise KeyError(path)
        return self.values[path.strip()]

    def put_secret(self, path: str, value: str) -> None:
        self.values[path.strip()] = value

    def delete_secret(self, path: str) -> None:
        self.values.pop(path.strip(), None)


class _FakeTransport:
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        return {}


class _FakeClientFamily:
    @property
    def transport(self) -> _FakeTransport:
        return _FakeTransport()


class _SpyClientFactory:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, str]] = []

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        self.calls.append(dict(credential_material))
        return _FakeClientFamily()


def _credential_json(secret_value: str = _SENTINEL) -> str:
    return json.dumps(
        {
            "type": "service_account",
            "client_id": "client-id",
            "private_key": secret_value,
        }
    )


def _google_connection(
    *,
    tenant_id: str = "tenant-a",
    credential_ref: str = _CREDENTIAL_REF_PATH,
) -> TenantConnection:
    now = datetime(2026, 9, 6, tzinfo=UTC)
    return TenantConnection(
        connection_ref="gw-conn",
        tenant_id=tenant_id,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        safe_display_name="Google Workspace",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref=credential_ref,
        validated_secret_free_config={},
        configuration_version=1,
        created_at=now,
        updated_at=now,
    )


def _late_google_factory(
    secrets: _CountingSecretsStore,
) -> GoogleWorkspaceTenantConnectionIntegrationFactory:
    return GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
        secrets_store=secrets,
    )


def test_credential_ref_is_safe_to_serialize() -> None:
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    payload = ref.model_dump()
    _assert_sentinel_absent(payload)
    _assert_sentinel_absent(ref.model_dump_json())
    assert payload["secret_path"] == _CREDENTIAL_REF_PATH
    assert "value" not in payload


def test_resolved_credential_repr_hides_secret() -> None:
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    resolved = ResolvedCredential(ref=ref, value=_SENTINEL)
    _assert_sentinel_absent(repr(resolved))
    _assert_sentinel_absent(str(resolved))


def test_rotation_does_not_change_credential_ref_fingerprint() -> None:
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    assert ref.identity_fingerprint() == ref.identity_fingerprint()


def test_wrong_tenant_resolution_fails_closed() -> None:
    store = _CountingSecretsStore({_CREDENTIAL_REF_PATH: _SENTINEL})
    resolver = SecretsStoreCredentialResolver(store)
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    with pytest.raises(CredentialScopeMismatchError):
        resolver.resolve(
            ref,
            context=CredentialResolutionContext(tenant_id="tenant-b"),
        )


def test_missing_secret_fails_closed_before_provider_operation() -> None:
    store = _CountingSecretsStore()
    factory = _late_google_factory(store)
    integration = factory.create_late_bound_integration(
        tenant_id="tenant-a",
        connection_ref="gw-conn",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        credential_ref=_CREDENTIAL_REF_PATH,
        secret_free_config={},
    )
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    assert store.get_secret_calls == []
    with pytest.raises(Exception):
        integration.require_client_family()
    assert len(factory._client_factory.calls) == 0  # type: ignore[attr-defined]


def test_malformed_credential_fails_before_external_operation() -> None:
    store = _CountingSecretsStore({_CREDENTIAL_REF_PATH: "not-json"})
    factory = _late_google_factory(store)
    integration = factory.create_late_bound_integration(
        tenant_id="tenant-a",
        connection_ref="gw-conn",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        credential_ref=_CREDENTIAL_REF_PATH,
        secret_free_config={},
    )
    with pytest.raises(Exception) as exc_info:
        integration.require_client_family()
    _assert_sentinel_absent(str(exc_info.value))
    assert len(factory._client_factory.calls) == 0  # type: ignore[attr-defined]


def test_late_resolution_spy_rehydrate_then_operate() -> None:
    store = _CountingSecretsStore({_CREDENTIAL_REF_PATH: _credential_json()})
    document_store = ConditionalInMemoryDocumentStore()
    repository = DocumentStoreTenantConnectionRepository(document_store)
    connection = _google_connection()
    repository.create(connection)
    registry = KnowledgeConnectionRegistry()
    factory = _late_google_factory(store)
    rehydrator = TenantConnectionRehydrator(
        repository=repository,
        secrets_store=store,
        integration_factory=factory,
        connection_registry=registry,
    )
    result = rehydrator.rehydrate_connection(
        tenant_id="tenant-a",
        connection_ref="gw-conn",
    )
    assert result.status is TenantConnectionRehydrationStatus.REGISTERED
    assert store.get_secret_calls == []
    integration = registry.resolve(
        tenant_id="tenant-a",
        connection_ref="gw-conn",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    integration.require_client_family()
    assert store.get_secret_calls == [(_CREDENTIAL_REF_PATH, None)]


def test_rotation_without_profile_rebuild() -> None:
    store = _CountingSecretsStore(
        {
            _CREDENTIAL_REF_PATH: _credential_json("v1-material"),
        }
    )
    factory = _late_google_factory(store)
    integration = factory.create_late_bound_integration(
        tenant_id="tenant-a",
        connection_ref="gw-conn",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        credential_ref=_CREDENTIAL_REF_PATH,
        secret_free_config={},
    )
    integration.require_client_family()
    store.values[_CREDENTIAL_REF_PATH] = _credential_json("v2-material")
    integration_b = factory.create_late_bound_integration(
        tenant_id="tenant-a",
        connection_ref="gw-conn",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        credential_ref=_CREDENTIAL_REF_PATH,
        secret_free_config={},
    )
    integration_b.require_client_family()
    assert store.get_secret_calls[-1] == (_CREDENTIAL_REF_PATH, None)
    assert store.values[_CREDENTIAL_REF_PATH] == _credential_json("v2-material")


def test_version_pinned_resolution() -> None:
    store = _CountingSecretsStore(
        versions={
            (_CREDENTIAL_REF_PATH, "v1"): _credential_json("pinned-v1"),
            (_CREDENTIAL_REF_PATH, None): _credential_json("current"),
        }
    )
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
        version="v1",
    )
    resolver = SecretsStoreCredentialResolver(store)
    resolved = resolver.resolve(ref, context=CredentialResolutionContext(tenant_id="tenant-a"))
    assert "pinned-v1" in resolved.value
    store.values[(_CREDENTIAL_REF_PATH, None)] = _credential_json("rotated")
    resolved_again = resolver.resolve(ref, context=CredentialResolutionContext(tenant_id="tenant-a"))
    assert "pinned-v1" in resolved_again.value


def test_two_operations_cause_two_reads() -> None:
    store = _CountingSecretsStore({_CREDENTIAL_REF_PATH: _credential_json()})
    factory = _late_google_factory(store)
    for _ in range(2):
        integration = factory.create_late_bound_integration(
            tenant_id="tenant-a",
            connection_ref="gw-conn",
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            credential_ref=_CREDENTIAL_REF_PATH,
            secret_free_config={},
        )
        integration.require_client_family()
    assert len(store.get_secret_calls) == 2


def test_safe_exception_sanitization() -> None:
    message = sanitize_credential_error_message(f"failed token={_SENTINEL}")
    _assert_sentinel_absent(message)


def test_profile_resolution_materialization_activation_no_secret_fetch() -> None:
    store = _CountingSecretsStore({_CREDENTIAL_REF_PATH: _SENTINEL})
    application = ApplicationEnvironmentProfile.lab_defaults(profile_id="p1.7.test")
    resolution = resolve_profile(application, layers=())
    revision_store = InMemoryEffectiveProfileRevisionStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    revision = materialize_effective_profile_revision(
        resolution,
        scope=_SCOPE,
        store=revision_store,
    )
    assert store.get_secret_calls == []
    service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=revision_store,
            active_store=active_store,
        ),
    )
    service.activate(
        ActivateEffectiveProfileRevisionRequest(
            scope=_SCOPE,
            candidate_revision_id=revision.revision_id,
        ),
    )
    assert store.get_secret_calls == []
    _assert_sentinel_absent(revision.model_dump_json())
    _assert_sentinel_absent(resolution.model_dump_json())


def test_effective_profile_revision_and_inspection_exclude_secret_material() -> None:
    application = ApplicationEnvironmentProfile.lab_defaults(profile_id="p1.7.test")
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    resolution = resolve_profile(application, layers=())
    revision = materialize_effective_profile_revision(resolution, scope=_SCOPE)
    revision_payload = {
        **revision.model_dump(mode="json"),
        "credential_ref": ref.model_dump(mode="json"),
    }
    _assert_sentinel_absent(json.dumps(revision_payload))
    inspection_view = safe_effective_profile_revision_view(revision)
    _assert_sentinel_absent(inspection_view.model_dump_json())
    snapshot = redacted_profile_snapshot(application)
    assert profile_contains_no_raw_secrets(snapshot, raw_secret=_SENTINEL)


def test_checkpoint_serialization_excludes_secret_material() -> None:
    checkpoint = TaskCheckpoint(
        task_id="task-1",
        tenant_id="tenant-a",
        resume_token="resume-1",
        task_state=TaskState.RUNNING,
        task_snapshot={"credential_ref": _CREDENTIAL_REF_PATH, "status": "running"},
    )
    _assert_sentinel_absent(checkpoint.model_dump_json())


def test_google_workspace_adapter_resolves_at_operation_boundary() -> None:
    store = _CountingSecretsStore({_CREDENTIAL_REF_PATH: _credential_json()})
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    adapter = GoogleWorkspaceSecretsStoreCredentialResolver(
        resolver=SecretsStoreCredentialResolver(store),
        credential_ref=ref,
        context=CredentialResolutionContext(tenant_id="tenant-a"),
    )
    assert store.get_secret_calls == []
    material = adapter.resolve_credential(_CREDENTIAL_REF_PATH)
    assert material["private_key"] == _SENTINEL
    assert store.get_secret_calls == [(_CREDENTIAL_REF_PATH, None)]


def test_credential_not_found_error_is_safe() -> None:
    store = _CountingSecretsStore()
    resolver = SecretsStoreCredentialResolver(store)
    ref = CredentialRef.from_secret_path(
        provider_id="google_workspace",
        secret_path=_CREDENTIAL_REF_PATH,
        tenant_id="tenant-a",
    )
    with pytest.raises(CredentialNotFoundError) as exc_info:
        resolver.resolve(ref, context=CredentialResolutionContext(tenant_id="tenant-a"))
    _assert_sentinel_absent(str(exc_info.value))


_P1_7A_DISPATCH_SOURCE_FILES = (
    "intergrax/runtime/vendor_knowledge/tenant_connection_rehydration.py",
    "intergrax/runtime/vendor_knowledge/tenant_connection_factory_registry.py",
    "intergrax/integrations/contracts/credential.py",
)


def test_p1_7a_no_reflection_dispatch_in_canonical_sources() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    for relative_path in _P1_7A_DISPATCH_SOURCE_FILES:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "hasattr(" not in source
        assert "getattr(" not in source
    repo_text = "\n".join(
        (repo_root / relative_path).read_text(encoding="utf-8")
        for relative_path in _P1_7A_DISPATCH_SOURCE_FILES
    )
    assert 'credential=""' not in repo_text


def test_p1_7a_google_workspace_declares_late_bound_mode() -> None:
    store = _CountingSecretsStore()
    factory = _late_google_factory(store)
    assert factory.credential_resolution_mode is CredentialResolutionMode.LATE_BOUND


def test_p1_7a_slack_declares_resolved_material_mode() -> None:
    factory = SlackTenantConnectionIntegrationFactory()
    assert factory.credential_resolution_mode is CredentialResolutionMode.RESOLVED_MATERIAL


def test_p1_7a_registry_forwards_factory_mode() -> None:
    store = _CountingSecretsStore()
    gw_factory = _late_google_factory(store)
    slack_factory = SlackTenantConnectionIntegrationFactory()
    registry = TenantConnectionIntegrationFactoryRegistry(
        [
            (
                GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                IntegrationCategory.COLLABORATION_SUITE,
                gw_factory,
            ),
            (
                "slack",
                IntegrationCategory.CONVERSATION_CHANNEL,
                slack_factory,
            ),
        ]
    )
    assert (
        registry.credential_resolution_mode_for(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
        is CredentialResolutionMode.LATE_BOUND
    )
    assert (
        registry.credential_resolution_mode_for(
            provider_id="slack",
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        )
        is CredentialResolutionMode.RESOLVED_MATERIAL
    )
    with pytest.raises(ValueError, match="factory is unavailable"):
        registry.credential_resolution_mode_for(
            provider_id="missing",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        )


def test_p1_7a_malformed_factory_registration_fails() -> None:
    class _BrokenFactory:
        credential_resolution_mode = "late_bound"

        def credential_resolution_mode_for(self, **_: object) -> CredentialResolutionMode:
            return CredentialResolutionMode.LATE_BOUND

        def create_integration_with_resolved_credential(self, **_: object) -> object:
            return object()

        def create_late_bound_integration(self, **_: object) -> object:
            return object()

    registry = TenantConnectionIntegrationFactoryRegistry()
    with pytest.raises(TypeError, match="valid CredentialResolutionMode"):
        registry.register(
            provider_id="broken",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            factory=_BrokenFactory(),  # type: ignore[arg-type]
        )

