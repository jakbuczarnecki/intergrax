"""Red-first NT-TEN probes using MCP identity APIs present at FRZ-TEN-07 prefix."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastmcp.exceptions import ToolError
from fastmcp.server.context import RequestContext, request_ctx
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from starlette.requests import Request

pytestmark = pytest.mark.unit


def test_nt_ten_01_canonical_tenant_identity_where_tenant_scope_applies(tmp_path: Path) -> None:
    """NT-TEN-01 / FRZ-TEN-01: canonical tenant identity where tenant scope applies."""
    from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    sig = inspect.signature(wire_application_evaluation)
    assert "tenant_id" in sig.parameters, (
        "NT-TEN-01 canonical tenant identity where tenant scope applies: "
        "wire_application_evaluation must accept tenant_id when tenant scope applies"
    )

    from intergrax.runtime.architecture.online_evaluation_registry import TenantScopedOnlineEvaluationRegistry

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="nt.ten.01")
    env = env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(
                update={"online_registry_enabled": True, "registry_path": tmp_path / "ten01.json"}
            )
        }
    )
    wiring = wire_application_evaluation(env, tenant_id="tenant-canonical")
    assert isinstance(wiring.registry, TenantScopedOnlineEvaluationRegistry), (
        "NT-TEN-01 canonical tenant identity where tenant scope applies: "
        "tenant-bound evaluation wiring must expose TenantScopedOnlineEvaluationRegistry"
    )
    assert wiring.tenant_id == "tenant-canonical", (
        "NT-TEN-01 canonical tenant identity where tenant scope applies: "
        "ApplicationEvaluationWiring must carry the canonical tenant_id"
    )


def test_nt_ten_10_missing_or_invalid_tenant_must_not_silently_become_global_scope() -> None:
    """NT-TEN-10 / FRZ-TEN-10: missing/invalid tenant must not silently become global scope."""
    from intergrax.runtime.architecture.online_evaluation_models import (
        OnlineEvaluationMode,
        OnlineEvaluationObservation,
    )

    assert "tenant_id" in OnlineEvaluationObservation.model_fields, (
        "NT-TEN-10 missing/invalid tenant must not silently become global: "
        "OnlineEvaluationObservation must declare tenant_id for canonical stamping"
    )
    with pytest.raises(ValueError, match="tenant_id must be a non-blank string"):
        OnlineEvaluationObservation(
            observation_id="obs-1",
            run_id="run-1",
            agent_id="agent-1",
            mode=OnlineEvaluationMode.ONLINE,
            scenario_id="scenario-1",
            passed=True,
            score=1.0,
            tenant_id="   ",
        )


@contextmanager
def _request(*, client_host: str = "127.0.0.1") -> Iterator[Request]:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "query_string": b"",
        "client": (client_host, 12345),
        "app": FastAPI(),
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ctx = RequestContext(request_id="nt-ten-07", meta=None, session=None, lifespan_context=None, request=request)
    token = request_ctx.set(ctx)
    try:
        yield request
    finally:
        request_ctx.reset(token)


def test_nt_ten_07_mcp_client_asserted_tenant_identity_must_fail_closed() -> None:
    """NT-TEN-07 / FRZ-TEN-07: MCP client-asserted tenant identity must fail closed."""
    with _request():
        try:
            resolved = resolve_mcp_run_agent_identity(
                harness_auth=HarnessAuthState(),
                asserted_tenant_id="victim",
                asserted_user_id=None,
                default_tenant_id="mcp",
                default_user_id="mcp-user",
            )
        except ToolError:
            return
        assert resolved.tenant_id != "victim", (
            "NT-TEN-07 MCP client-asserted tenant identity must fail closed: "
            "client-asserted tenant must not establish truth without verified harness principal"
        )
