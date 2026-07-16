# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8A — LKW hosted profile builder tests."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.hosting import resolve_hosted_application_definition
from intergrax.hosting.contracts.context import (
    HostedApplicationContext,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.events import HostedApplicationEvent
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
    HostedApplicationShutdownCoordinator,
)
from intergrax.hosting.contracts.policies import InstanceExclusivityMode, RestartMode
from intergrax.hosting.contracts.profile import HostedApplicationProfile
from intergrax.hosting.engine.health import (
    HostedApplicationHealthSnapshot,
    HostedApplicationReadinessService,
)
from intergrax.hosting.engine.ports import HostedApplicationRuntime
from intergrax.hosting.engine.runtime import invoke_application_factory
from intergrax.hosting.services import HostedApplicationServiceRegistry
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting import build_local_workspace_hosted_profile
from local_workspace_application.hosting.profile import LOCAL_WORKSPACE_HOSTED_FACTORY_ID
from local_workspace_application.hosting.runtime import _LocalWorkspaceHostedRuntime
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST


class _Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class _Logger:
    def debug(self, message: str, **fields: object) -> None:
        del message, fields

    def info(self, message: str, **fields: object) -> None:
        del message, fields

    def warning(self, message: str, **fields: object) -> None:
        del message, fields

    def error(self, message: str, **fields: object) -> None:
        del message, fields


class _EventPublisher:
    async def publish(self, event: HostedApplicationEvent) -> None:
        del event


class _Shutdown:
    def request_shutdown(self, *, reason_code: str = "test") -> None:
        del reason_code

    def is_shutdown_requested(self) -> bool:
        return False

    async def wait_until_requested(self) -> None:
        return None


class _LifecycleProvider:
    def snapshot(self) -> HostedApplicationLifecycleSnapshot:
        return HostedApplicationLifecycleSnapshot(
            state=HostedApplicationLifecycleState.READY,
            accepting_new_work=True,
            shutdown_requested=False,
            last_transition_at=datetime.now(timezone.utc),
            reason_code="ready",
        )


class _ReadinessService:
    def snapshot(self) -> HostedApplicationHealthSnapshot:
        return HostedApplicationHealthSnapshot(
            live=True,
            ready=True,
            degraded=False,
            accepting_new_work=True,
            runtime_ready=True,
            instance_ownership_valid=True,
            shutdown_requested=False,
            last_evaluated_at=datetime.now(timezone.utc),
        )

    def accepts_new_work(self) -> bool:
        return True


def _minimal_context(profile: HostedApplicationProfile) -> HostedApplicationContext:
    public = profile.public_view()
    digest = profile.profile_digest()
    services = HostedApplicationServiceRegistry()
    services.register(HostedApplicationReadinessService, _ReadinessService())
    return HostedApplicationContext(
        application_id=profile.application_id,
        instance_id="01TESTHOSTEDPROFILEINSTANCE0001",
        profile=public,
        profile_digest=digest,
        paths=HostedApplicationPaths(
            data_home=Path("build/test-lkw-hosting-8a"),
            run_directory=Path("build/test-lkw-hosting-8a/run"),
        ),
        process_identity=HostedApplicationProcessIdentity(
            process_id=1,
            started_at=datetime.now(timezone.utc),
        ),
        services=services,
        clock=_Clock(),
        logger=_Logger(),
        event_publisher=_EventPublisher(),
        shutdown=cast(HostedApplicationShutdownCoordinator, _Shutdown()),
        lifecycle=_LifecycleProvider(),
    )


def test_builder_returns_hosted_application_profile() -> None:
    profile = build_local_workspace_hosted_profile(
        settings=LocalWorkspaceBackendSettings(),
    )
    assert isinstance(profile, HostedApplicationProfile)
    assert profile.application_id == "local_workspace"
    assert profile.application_id == LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id
    assert profile.application_factory_id == LOCAL_WORKSPACE_HOSTED_FACTORY_ID
    assert profile.instance.exclusivity_mode is InstanceExclusivityMode.SINGLE_INSTANCE
    assert profile.restart.mode is RestartMode.ON_FAILURE
    assert profile.restart.max_attempts == 3
    assert profile.hooks.flattened_public_descriptors() == ()
    assert profile.components == ()
    assert profile.event_subscriptions == ()
    assert profile.metadata == {
        "product_id": "local_workspace",
        "product_tier": "tier3",
        "runtime_kind": "fastapi_uvicorn",
    }


def test_profile_resolves_and_digests_are_deterministic() -> None:
    settings = LocalWorkspaceBackendSettings()
    profile_a = build_local_workspace_hosted_profile(settings=settings)
    profile_b = build_local_workspace_hosted_profile(settings=settings)
    definition_a = resolve_hosted_application_definition(profile_a)
    definition_b = resolve_hosted_application_definition(profile_b)
    assert definition_a.application_id == "local_workspace"
    assert profile_a.profile_digest() == profile_b.profile_digest()
    assert definition_a.definition_digest == definition_b.definition_digest


def test_settings_snapshot_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    snapshot = LocalWorkspaceBackendSettings(backend_port=18020)

    def _from_env() -> LocalWorkspaceBackendSettings:
        calls["count"] += 1
        return snapshot

    monkeypatch.setattr(LocalWorkspaceBackendSettings, "from_env", staticmethod(_from_env))
    profile = build_local_workspace_hosted_profile()
    assert calls["count"] == 1

    context = _minimal_context(profile)
    runtime_one = cast(
        _LocalWorkspaceHostedRuntime,
        profile.application_factory(context),
    )
    runtime_two = cast(
        _LocalWorkspaceHostedRuntime,
        profile.application_factory(context),
    )
    assert calls["count"] == 1
    assert runtime_one is not runtime_two
    assert runtime_one._settings is snapshot  # noqa: SLF001
    assert runtime_two._settings is snapshot  # noqa: SLF001
    assert runtime_one._bind_host == runtime_two._bind_host  # noqa: SLF001
    assert runtime_one._bind_port == runtime_two._bind_port == 18020  # noqa: SLF001
    assert runtime_one._hosted_context is context  # noqa: SLF001
    assert runtime_two._hosted_context is context  # noqa: SLF001


def test_public_view_excludes_settings_and_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    marker_api = "SECRET_API_KEY_MARKER_8A"
    marker_sentry = "SECRET_SENTRY_DSN_MARKER_8A"
    marker_otlp = "SECRET_OTLP_MARKER_8A"
    marker_es = "SECRET_ES_MARKER_8A"
    marker_data = "SECRET_DATA_HOME_MARKER_8A"
    marker_root = "SECRET_ROOT_MARKER_8A"
    marker_host = "SECRET_BIND_HOST_MARKER_8A"
    marker_port = 59876

    monkeypatch.setenv("LOCAL_WORKSPACE_BACKEND_HOST", marker_host)
    settings = LocalWorkspaceBackendSettings(
        backend_port=marker_port,
        api_keys_map={
            marker_api: ApiKeyIdentity(
                tenant_id="tenant-a",
                user_id="user-a",
                scopes=("*",),
            ),
        },
        observability_sentry_dsn=f"https://{marker_sentry}@example.com/1",
        observability_otlp_endpoint=f"https://{marker_otlp}.example.com",
        observability_elasticsearch_url=f"https://{marker_es}.example.com",
        data_home=f"/tmp/{marker_data}",
        allowed_read_roots=frozenset({f"/tmp/{marker_root}"}),
    )
    profile = build_local_workspace_hosted_profile(settings=settings)
    public_json = profile.public_view().model_dump_json()
    digest = profile.profile_digest()
    definition = resolve_hosted_application_definition(profile)

    forbidden = (
        marker_api,
        marker_sentry,
        marker_otlp,
        marker_es,
        marker_data,
        marker_root,
        marker_host,
        str(marker_port),
        "api_keys",
        "DATA_HOME",
        "LKW_DATA_HOME",
        repr(settings),
    )
    for marker in forbidden:
        assert marker not in public_json
        assert marker not in digest
        assert marker not in definition.definition_digest
        assert marker not in definition.profile_digest


@pytest.mark.asyncio
async def test_factory_protocol_via_invoke_application_factory() -> None:
    profile = build_local_workspace_hosted_profile(
        settings=LocalWorkspaceBackendSettings(),
    )
    context = _minimal_context(profile)
    runtime_one = await invoke_application_factory(profile.application_factory, context)
    runtime_two = await invoke_application_factory(profile.application_factory, context)
    assert isinstance(runtime_one, HostedApplicationRuntime)
    assert isinstance(runtime_two, HostedApplicationRuntime)
    assert runtime_one is not runtime_two


def test_hosting_package_import_boundary() -> None:
    package_root = Path(__file__).resolve().parents[2] / "hosting"
    forbidden_imports = {
        "HostedApplicationEngine",
        "HostedApplicationSupervisor",
        "FileHostedApplicationInstanceGuard",
        "PortableForegroundSignalAdapter",
        "HostedApplicationControlCoordinator",
        "SystemMonotonicClock",
        "run_hosted_application",
        "NexusLoop",
    }
    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[-1] not in forbidden_imports
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in forbidden_imports
                if node.module:
                    for name in forbidden_imports:
                        assert name not in node.module
