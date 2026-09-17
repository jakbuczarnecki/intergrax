# © Artur Czarnecki. All rights reserved.

"""LKW canonical host runtime composition authority proofs."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from pathlib import Path

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.integrations.contracts.document_store import DocumentStore
from local_workspace_application.host.host_runtime_composition import (
    LocalWorkspaceHostRuntimeAuthorityError,
    LocalWorkspaceHostTenantBinding,
    build_local_workspace_harness_host_runtime,
    build_local_workspace_host_environment,
    resolve_local_workspace_host_tenant_binding,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _settings_with_tenant(tenant_id: str) -> LocalWorkspaceBackendSettings:
    base = LocalWorkspaceBackendSettings.from_env()
    return replace(
        base,
        api_keys_map={
            "test-key": ApiKeyIdentity(
                tenant_id=tenant_id,
                user_id="user-1",
                scopes=("*",),
            )
        },
    )


def test_tenant_binding_propagates_to_harness_runtime() -> None:
    settings = _settings_with_tenant("tenant-alpha")
    env = build_local_workspace_host_environment(settings)
    projection = build_lkw_test_registry_projection(settings)
    captured: list[str] = []

    def _capture_tenant(*_args: object, **kwargs: object) -> MagicMock:
        captured.append(str(kwargs.get("tenant_id")))
        runtime = MagicMock()
        runtime.registry = MagicMock()
        runtime.reliability = MagicMock(idempotency_store=None)
        runtime.agent_checkpoint_store = None
        runtime.compensation_queue_store = None
        runtime.env_wiring = MagicMock()
        runtime.execution = MagicMock()
        runtime.registry_projection_evidence = projection.evidence
        return runtime

    with patch(
        "local_workspace_application.host.host_runtime_composition.build_harness_host_runtime",
        side_effect=_capture_tenant,
    ):
        composition = build_local_workspace_harness_host_runtime(
            settings=settings,
            registry_projection=projection,
            environment=env,
        )
    assert composition.tenant_binding.tenant_id == "tenant-alpha"
    assert captured == ["tenant-alpha"]


def test_cross_tenant_bindings_differ() -> None:
    binding_a = resolve_local_workspace_host_tenant_binding(
        _settings_with_tenant("tenant-a"),
    )
    binding_b = resolve_local_workspace_host_tenant_binding(
        _settings_with_tenant("tenant-b"),
    )
    assert binding_a.tenant_id != binding_b.tenant_id


def test_custom_document_store_injection_is_used() -> None:
    settings = _settings_with_tenant("tenant-custom-store")
    env = build_local_workspace_host_environment(settings)
    projection = build_lkw_test_registry_projection(settings)
    custom_store = MagicMock(spec=DocumentStore)
    captured_stores: list[object] = []

    def _capture_store(*_args: object, **kwargs: object) -> MagicMock:
        captured_stores.append(kwargs.get("document_store"))
        runtime = MagicMock()
        runtime.registry = MagicMock()
        runtime.reliability = MagicMock(idempotency_store=None)
        runtime.agent_checkpoint_store = None
        runtime.compensation_queue_store = None
        runtime.env_wiring = MagicMock()
        runtime.execution = MagicMock()
        return runtime

    with patch(
        "local_workspace_application.host.host_runtime_composition.build_harness_host_runtime",
        side_effect=_capture_store,
    ):
        composition = build_local_workspace_harness_host_runtime(
            settings=settings,
            registry_projection=projection,
            environment=env,
            document_store=custom_store,
        )
    assert composition.document_store is custom_store
    assert captured_stores == [custom_store]


def test_missing_registry_projection_fails_closed() -> None:
    settings = _settings_with_tenant("tenant-fail")
    env = build_local_workspace_host_environment(settings)
    with pytest.raises(HarnessHostRegistryAuthorityError, match="MaterializedRegistryProjection"):
        build_local_workspace_harness_host_runtime(
            settings=settings,
            registry_projection=None,  # type: ignore[arg-type]
            environment=env,
        )


def test_ambiguous_api_key_tenants_fail_closed() -> None:
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        api_keys_map={
            "key-a": ApiKeyIdentity(tenant_id="tenant-a", user_id="u1", scopes=("*",)),
            "key-b": ApiKeyIdentity(tenant_id="tenant-b", user_id="u2", scopes=("*",)),
        },
    )
    env = build_local_workspace_host_environment(settings)
    with pytest.raises(
        LocalWorkspaceHostRuntimeAuthorityError,
        match="local_workspace_host_tenant_authority_ambiguous",
    ):
        resolve_local_workspace_host_tenant_binding(settings)


def test_http_and_worker_share_tenant_resolver_semantics() -> None:
    host_dir = Path(__file__).resolve().parents[4] / "applications" / "local_workspace_application" / "host"
    factory_source = (host_dir / "factory.py").read_text(encoding="utf-8")
    worker_factory_source = (host_dir / "background_worker_factory.py").read_text(encoding="utf-8")
    worker_main_source = (host_dir / "background_worker_main.py").read_text(encoding="utf-8")
    assert "build_local_workspace_harness_host_runtime" in factory_source
    assert "build_local_workspace_harness_host_runtime" in worker_factory_source
    assert "resolve_local_workspace_host_tenant_binding" in worker_main_source
    settings = _settings_with_tenant("tenant-shared")
    env = build_local_workspace_host_environment(settings)
    assert resolve_local_workspace_host_tenant_binding(settings).tenant_id == "tenant-shared"


def test_explicit_tenant_binding_override() -> None:
    settings = _settings_with_tenant("tenant-from-keys")
    env = build_local_workspace_host_environment(settings)
    projection = build_lkw_test_registry_projection(settings)
    override = LocalWorkspaceHostTenantBinding(tenant_id="tenant-injected")
    with patch(
        "local_workspace_application.host.host_runtime_composition.build_harness_host_runtime",
    ) as build_runtime:
        build_runtime.return_value = MagicMock(
            registry=MagicMock(),
            reliability=MagicMock(idempotency_store=None),
        )
        composition = build_local_workspace_harness_host_runtime(
            settings=settings,
            registry_projection=projection,
            environment=env,
            tenant_binding=override,
        )
    build_runtime.assert_called_once()
    assert build_runtime.call_args.kwargs["tenant_id"] == "tenant-injected"
    assert composition.tenant_binding.tenant_id == "tenant-injected"


def test_configured_host_tenant_without_api_keys() -> None:
    base = LocalWorkspaceBackendSettings.from_env()
    settings = replace(
        base,
        api_keys_map={},
        host_tenant_id="tenant-configured",
    )
    binding = resolve_local_workspace_host_tenant_binding(settings)
    assert binding.tenant_id == "tenant-configured"


def test_missing_tenant_authority_fails_closed() -> None:
    base = LocalWorkspaceBackendSettings.from_env()
    settings = replace(
        base,
        api_keys_map={},
        host_tenant_id="",
    )
    with pytest.raises(
        LocalWorkspaceHostRuntimeAuthorityError,
        match="local_workspace_host_tenant_authority_missing",
    ):
        resolve_local_workspace_host_tenant_binding(settings)


def test_blank_host_tenant_config_treated_as_missing() -> None:
    base = LocalWorkspaceBackendSettings.from_env()
    settings = replace(
        base,
        api_keys_map={},
        host_tenant_id="   ",
    )
    with pytest.raises(
        LocalWorkspaceHostRuntimeAuthorityError,
        match="local_workspace_host_tenant_authority_missing",
    ):
        resolve_local_workspace_host_tenant_binding(settings)


def test_profile_id_is_not_tenant_authority_fallback() -> None:
    base = LocalWorkspaceBackendSettings.from_env()
    settings = replace(
        base,
        api_keys_map={},
        host_tenant_id="",
    )
    env = build_local_workspace_host_environment(settings)
    assert env.profile_id.strip()
    with pytest.raises(
        LocalWorkspaceHostRuntimeAuthorityError,
        match="local_workspace_host_tenant_authority_missing",
    ):
        resolve_local_workspace_host_tenant_binding(settings)


def test_configured_host_tenant_overrides_api_key_tenant() -> None:
    settings = replace(
        _settings_with_tenant("tenant-from-keys"),
        host_tenant_id="tenant-configured",
    )
    assert resolve_local_workspace_host_tenant_binding(settings).tenant_id == "tenant-configured"
