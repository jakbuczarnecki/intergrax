# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.execution.adapters.default_adapter import DefaultExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore

pytestmark = pytest.mark.unit


class _CountingAdapter(DefaultExecutionAdapter):
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls += 1


def test_http_only_app_lifecycle_shuts_down_implicit_adapter_once() -> None:
    adapter = _CountingAdapter()
    app = create_app(ApiConfig(execution_adapter=adapter))
    with TestClient(app):
        pass
    assert adapter.shutdown_calls == 1


def test_explicit_run_service_does_not_shutdown_unowned_adapter() -> None:
    adapter = _CountingAdapter()
    store = InMemoryRunStore()
    service = DefaultRunService(store, adapter)
    app = create_app(ApiConfig(run_service=service, run_store=store))
    with TestClient(app):
        pass
    assert adapter.shutdown_calls == 0


def test_implicit_default_adapter_serves_health() -> None:
    app = create_app(ApiConfig(environment=ApiEnvironment.DEV))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_mcp_mounted_app_runs_inner_core_lifecycle() -> None:
    adapter = _CountingAdapter()
    core = create_app(ApiConfig(execution_adapter=adapter))
    mcp = FastMCP("FactoryLifecycleProbe")

    @mcp.tool
    def ping() -> str:
        return "pong"

    wrapper = couple_fastapi_with_mcp(core, mcp, mount_path="/mcp")
    assert isinstance(wrapper, FastAPI)
    with TestClient(wrapper) as client:
        assert client.get("/health").status_code == 200
    assert adapter.shutdown_calls == 1
