# © Artur Czarnecki. All rights reserved.

"""CONFIG-X-R1 remediation gates (blocker exit + tenant isolation audit)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.applications._shared.harness_task_routes import (
    HarnessAsyncRunRequest,
    task_from_harness_async_run_request,
)
from intergrax.applications._shared.trace_explorer_routes import create_trace_explorer_router
from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _TracesOnlyA:
    def query_traces(self, *, limit: int = 20, name=None):
        return None


class _TracesOnlyB:
    def query_traces(self, *, limit: int = 20, name=None):
        return None


def test_o1_observability_no_backend_configured() -> None:
    ctx = ToolWiringContext()
    with pytest.raises(RuntimeError, match="observability_backend_not_configured"):
        resolve_observability_backend(ctx)


def test_o2_observability_role_without_sanctioned_backend_fails_closed() -> None:
    ctx = ToolWiringContext(
        observability_backends={"custom_a": _TracesOnlyA(), "custom_b": _TracesOnlyB()},
    )
    with pytest.raises(RuntimeError, match="observability_role_backend_not_configured:traces"):
        resolve_observability_backend(ctx, role="traces")


def test_o3_observability_does_not_use_registration_order() -> None:
    first = _TracesOnlyA()
    second = _TracesOnlyB()
    ctx_forward = ToolWiringContext(
        observability_backends={"custom_a": first, "custom_b": second},
    )
    ctx_reverse = ToolWiringContext(
        observability_backends={"custom_b": second, "custom_a": first},
    )
    with pytest.raises(RuntimeError):
        resolve_observability_backend(ctx_forward, role="traces")
    with pytest.raises(RuntimeError):
        resolve_observability_backend(ctx_reverse, role="traces")


def test_o4_observability_explicit_default_backend() -> None:
    explicit = object()
    ctx = ToolWiringContext(observability_backend=explicit)
    assert resolve_observability_backend(ctx, role="default") is explicit


def test_t1_harness_async_run_requires_resolved_tenant_parameter() -> None:
    body = HarnessAsyncRunRequest(message="hi", capability="cap.demo")
    task = task_from_harness_async_run_request(body, tenant_id="tenant-explicit")
    assert task.tenant_id == "tenant-explicit"


def test_t2_trace_explorer_missing_tenant_rejected() -> None:
    app = FastAPI()
    app.include_router(create_trace_explorer_router(enabled=True))
    client = TestClient(app)
    response = client.get("/ops/trace/runs/run-1")
    assert response.status_code == 422


def test_t4_image_smart_loader_requires_explicit_tenant() -> None:
    text = (_REPO_ROOT / "intergrax/multimedia/image_smart_loader.py").read_text(
        encoding="utf-8",
    )
    assert 'tenant_id: str = "default"' not in text


def test_t5_vector_integration_config_missing_tenant_fails_closed() -> None:
    with pytest.raises(IntegrationConfigurationError, match="TENANT_ID"):
        VectorIntegrationConfig.from_env("INTERGRAX_CONFIG_X_PROBE")
    cfg = VectorIntegrationConfig(url="http://127.0.0.1", tenant_id="")
    with pytest.raises(IntegrationConfigurationError):
        cfg.require_tenant_id()


def test_blocker_exit_no_arbitrary_observability_backend_selection() -> None:
    text = (
        _REPO_ROOT / "intergrax/tools/providers/observability/resolve.py"
    ).read_text(encoding="utf-8")
    assert "next(iter(backends.values()))" not in text


def test_blocker_exit_tokenizer_registry_no_first_registered_default() -> None:
    registry = TokenizerRegistry()
    registry.register(SimpleTokenizer())
    registry.register(TiktokenTokenizer())
    with pytest.raises(ValueError, match="No default tokenizer configured"):
        registry.get(None)


def test_blocker_exit_production_surfaces_no_ambient_default_tenant_literal() -> None:
    paths = (
        "intergrax/applications/_shared/harness_task_routes.py",
        "intergrax/applications/_shared/trace_explorer_routes.py",
        "intergrax/multimedia/image_smart_loader.py",
        "intergrax/integrations/_shared/p3/configs.py",
    )
    for rel in paths:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'tenant_id: str = "default"' not in text
        assert 'PREFIX_TENANT_ID", "default")' not in text


def test_t1_harness_http_route_rejects_missing_tenant_when_unauthenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    from tests.unit.applications.harness_canonical_task_routes_test_support import (
        mount_canonical_harness_task_routes_for_tests,
    )

    app = FastAPI()
    mount_canonical_harness_task_routes_for_tests(app)
    client = TestClient(app)
    response = client.post(
        "/v1/tasks/run-async",
        json={"message": "hello", "capability": "demo.cap"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "tenant_id_required"
