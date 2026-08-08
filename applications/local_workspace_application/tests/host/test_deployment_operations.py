from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import (
    ManagedWorkspaceRepository,
    mount_managed_workspace_routes,
    resolve_managed_workspace_document_store,
)
from local_workspace_application.workspaces.sync_runtime import (
    ManagedWorkspaceSyncRuntime,
)

pytestmark = pytest.mark.unit


def _production_settings(**updates: object) -> LocalWorkspaceBackendSettings:
    values: dict[str, object] = {
        "environment": ApiEnvironment.PROD,
        "data_home": "/var/lib/intergrax/lkw",
    }
    values.update(updates)
    return LocalWorkspaceBackendSettings(**values)


def test_production_rejects_in_memory_durable_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", "mongodb://store")
    monkeypatch.setenv("INTERGRAX_QDRANT_URL", "http://qdrant:6333")

    with pytest.raises(ValueError, match="development-only"):
        _production_settings(document_store_backend="inmemory").validate_for_runtime()


def test_production_requires_durable_indexed_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.delenv("INTERGRAX_QDRANT_URL", raising=False)

    with pytest.raises(ValueError, match="INTERGRAX_MONGODB_URI"):
        _production_settings().validate_for_runtime()


def test_production_rejects_partial_live_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", "mongodb://store")
    monkeypatch.setenv("INTERGRAX_QDRANT_URL", "http://qdrant:6333")

    with pytest.raises(
        ValueError,
        match="connected_source_live_configuration_incomplete",
    ):
        _production_settings(
            connected_source_opaque_ref_signing_key="configured"
        ).validate_for_runtime()


def test_readiness_reports_capabilities_and_liveness_without_secret(
    tmp_path: Path,
) -> None:
    secret = "deployment-secret-that-is-not-in-health"
    app = create_local_workspace_backend_app(
        settings=LocalWorkspaceBackendSettings(
            data_home=str(tmp_path / "data"),
            include_mcp=False,
            include_scheduler=False,
            knowledge_admin_confirmation_secret=secret,
        )
    )

    with TestClient(app) as client:
        assert client.get("/v1/local_workspace/liveness").json() == {"alive": True}
        body = client.get("/v1/local_workspace/readiness").json()

    assert secret not in repr(body)
    components = {item["name"]: item for item in body["components"]}
    assert components["durable_store"]["healthy"] is True
    assert components["indexed"]["enabled"] is True
    assert components["live"]["enabled"] is False
    assert components["nl_administration"]["enabled"] is True
    assert components["sync_runtime"]["detail"] == "running"


def test_mandatory_store_failure_blocks_app_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _UnavailableStore:
        def query(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("provider detail must not become the contract")

    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda **_kwargs: _UnavailableStore(),
    )

    with pytest.raises(RuntimeError, match="lkw_durable_store_unavailable"):
        create_local_workspace_backend_app(
            settings=LocalWorkspaceBackendSettings(
                data_home=str(tmp_path / "data"),
                include_mcp=False,
                include_scheduler=False,
            )
        )


def test_sync_runtime_start_is_guarded_and_failed_start_is_stopped() -> None:
    class _Worker:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.start_calls = 0
            self.stop_calls = 0

        def start(self) -> None:
            self.start_calls += 1
            if self.fail:
                raise RuntimeError("worker-start-failure")

        def stop(self) -> None:
            self.stop_calls += 1

    def build(worker: _Worker) -> ManagedWorkspaceSyncRuntime:
        return ManagedWorkspaceSyncRuntime(
            message_bus=cast(MessageBus, object()),
            wiring_context=cast(Any, object()),
            worker=cast(Any, worker),
            registry=TaskExecutionRegistry(),
        )

    failed_worker = _Worker(fail=True)
    with pytest.raises(RuntimeError, match="worker-start-failure"):
        build(failed_worker).start()
    assert failed_worker.stop_calls == 1

    worker = _Worker()
    runtime = build(worker)
    runtime.start()
    with pytest.raises(RuntimeError, match="sync_runtime_already_started"):
        runtime.start()
    assert worker.start_calls == 1


class _TrackingStore:
    def __init__(self, delegate: Any, events: list[str]) -> None:
        self._delegate = delegate
        self._events = events
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self._events.append("close")
        self._delegate.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _PartiallyStartedRuntime:
    def __init__(self, events: list[str], *, fail: bool) -> None:
        self.wiring_context = cast(Any, object())
        self._events = events
        self._fail = fail
        self.started = False
        self.start_calls = 0
        self.stop_calls = 0

    def bind_main_loop(self, _loop: Any) -> None:
        pass

    def register_knowledge_ingestion_service(self, _service: Any) -> None:
        pass

    def attach_recovery_service(self, _service: Any) -> None:
        pass

    def start(self) -> None:
        self.start_calls += 1
        self.started = True
        self._events.append("start")
        if self._fail:
            raise RuntimeError("partial-start-failure")

    def stop(self) -> None:
        self.stop_calls += 1
        self.started = False
        self._events.append("stop")


def _mount_test_host(
    *,
    settings: LocalWorkspaceBackendSettings,
    repository: ManagedWorkspaceRepository | None = None,
) -> FastAPI:
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=cast(Any, object()),
        settings=settings,
        repository=repository,
    )
    return app


def test_sync_runtime_start_failure_cleans_owned_runtime_and_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = _TrackingStore(
        resolve_managed_workspace_document_store(backend="inmemory"),
        events,
    )
    runtime = _PartiallyStartedRuntime(events, fail=True)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda **_kwargs: store,
    )
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.build_managed_workspace_sync_runtime",
        lambda **_kwargs: runtime,
    )

    settings = LocalWorkspaceBackendSettings(
        data_home=str(tmp_path / "data"),
        document_store_backend="inmemory",
        include_mcp=False,
        include_scheduler=False,
    )
    with pytest.raises(RuntimeError, match="lkw_sync_runtime_start_failed"):
        with TestClient(_mount_test_host(settings=settings)):
            pass

    assert events == ["start", "stop", "close"]
    assert runtime.start_calls == 1
    assert runtime.stop_calls == 1
    assert runtime.started is False
    assert store.close_calls == 1


def test_sync_runtime_start_failure_preserves_injected_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = _TrackingStore(
        resolve_managed_workspace_document_store(backend="inmemory"),
        events,
    )
    runtime = _PartiallyStartedRuntime(events, fail=True)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.build_managed_workspace_sync_runtime",
        lambda **_kwargs: runtime,
    )
    repository = ManagedWorkspaceRepository(store)
    settings = LocalWorkspaceBackendSettings(
        data_home=str(tmp_path / "data"),
        document_store_backend="inmemory",
        include_mcp=False,
        include_scheduler=False,
    )

    with pytest.raises(RuntimeError, match="lkw_sync_runtime_start_failed"):
        with TestClient(_mount_test_host(settings=settings, repository=repository)):
            pass

    assert events == ["start", "stop"]
    assert runtime.stop_calls == 1
    assert store.close_calls == 0


def test_successful_sync_runtime_shutdown_closes_owned_store_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    store = _TrackingStore(
        resolve_managed_workspace_document_store(backend="inmemory"),
        events,
    )
    runtime = _PartiallyStartedRuntime(events, fail=False)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda **_kwargs: store,
    )
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.build_managed_workspace_sync_runtime",
        lambda **_kwargs: runtime,
    )
    settings = LocalWorkspaceBackendSettings(
        data_home=str(tmp_path / "data"),
        document_store_backend="inmemory",
        include_mcp=False,
        include_scheduler=False,
    )

    with TestClient(_mount_test_host(settings=settings)):
        assert runtime.started is True

    assert events == ["start", "stop", "close"]
    assert runtime.stop_calls == 1
    assert store.close_calls == 1


def test_sync_runtime_cleanup_failures_preserve_startup_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _TrackingStore(
        resolve_managed_workspace_document_store(backend="inmemory"),
        [],
    )
    runtime = _PartiallyStartedRuntime([], fail=True)

    def fail_stop() -> None:
        raise OSError("stop-failure")

    def fail_close() -> None:
        raise OSError("close-failure")

    monkeypatch.setattr(runtime, "stop", fail_stop)
    monkeypatch.setattr(store, "close", fail_close)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda **_kwargs: store,
    )
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.build_managed_workspace_sync_runtime",
        lambda **_kwargs: runtime,
    )
    settings = LocalWorkspaceBackendSettings(
        data_home=str(tmp_path / "data"),
        document_store_backend="inmemory",
        include_mcp=False,
        include_scheduler=False,
    )

    with pytest.raises(RuntimeError, match="lkw_sync_runtime_start_failed"):
        with TestClient(_mount_test_host(settings=settings)):
            pass
