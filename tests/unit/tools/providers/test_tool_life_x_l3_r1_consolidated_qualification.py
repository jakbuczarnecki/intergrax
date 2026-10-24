# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R1 — MCP subprocess + HTTP lost-ack RuntimeToolInvoker qualification."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import pytest
from testing_support.builder import (
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.idempotency_store import (
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.contracts.tool_execution_outcome import ToolEffectCertainty
from intergrax.integrations.providers.http_client.allowlist.client import AllowlistHttpClient
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
    SideEffectAuthorizationFailureReason,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.mcp_result_bounds import bound_mcp_structured_result
from intergrax.tools.providers.http.bundle import register_http_tools
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.service import HTTP_REQUEST_TOOL_ID
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.fixtures.tool_life_x_mcp_subprocess_qualification import (
    mcp_call_tool as _mcp_call_tool,
    mcp_qualification_subprocess as _mcp_subprocess,
)
from tests.unit.tools.providers.test_tool_life_x_batch_e_real_infrastructure_adversarial import (
    _allowlist_client,
    _local_fault_server,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_L3_RUN_ID = canonical_run_id_for_tests("tool-life-x-l3-r1")
pytest.importorskip("fastmcp")


class AllowAllScopePolicy:
    def is_allowed(self, *args: object, **kwargs: object) -> bool:
        return True


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="l3-r1.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _audit_only_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="l3-r1.audit")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="audit_only",
    )
    return wire_policy_bundle(env)


_TENANT_A = "tenant_l3_a"
_TENANT_B = "tenant_l3_b"


class _InvokeState:
    def __init__(
        self,
        *,
        tenant_id: str = "tenant_test",
        policy_bundle: object | None = _enforce_allow_bundle(),
    ) -> None:
        self.run_id = _L3_RUN_ID
        self._tenant_id = tenant_id
        self.request = type("Req", (), {"metadata": {}})()
        self.declarative_hitl_grant = None
        self._context = type(
            "Ctx",
            (),
            {"config": type("Cfg", (), {"policy_bundle": policy_bundle, "production_mode": False})()},
        )()

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def task_id(self) -> str | None:
        return None

    @property
    def context(self):
        return self._context

    def trace_event(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


@contextmanager
def _governed_invoke_scope(*, governance_tenant_id: str = "tenant_test"):
    with canonical_governed_execution_scope(
        _L3_RUN_ID,
        governance_tenant_id=governance_tenant_id,
        governance_principal_id="principal-tool-life-x-l3-r1",
    ):
        yield


@pytest.fixture(autouse=True)
def _l3_active_governance() -> None:
    with _governed_invoke_scope():
        yield


def _http_invoker(
    client: AllowlistHttpClient,
) -> tuple[RuntimeToolInvoker, InMemoryIdempotencyStore]:
    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    register_http_tools(registry, ToolWiringContext(http_client=client))
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        pre_effect_coordinator=IdempotencyPreEffectCoordinator(idempotency_store=store),
        scope_policy=AllowAllScopePolicy(),
    )
    return invoker, store


def test_l3_governance_unauthorized_http_zero_physical_effects() -> None:
    effects: list[str] = []

    def record_post(handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        effects.append(body.decode("utf-8"))
        handler.send_response(200)
        handler.end_headers()

    with _local_fault_server(record_post) as (base, hosts):
        client = _allowlist_client(hosts)
        invoker, _store = _http_invoker(client)
        state = _InvokeState(policy_bundle=None)
        request = ToolExecutionRequest(
            run_id=_L3_RUN_ID,
            step_id="1",
            tool_id=HTTP_REQUEST_TOOL_ID,
            input=HttpRequestInput(
                url=f"{base}/charge",
                method="POST",
                body=json.dumps({"amount": 1}),
                timeout_s=5.0,
            ),
            idempotency_key="l3-gov-deny-no-bundle",
        )
        with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError) as exc:
            invoker.invoke(state=state, agent_id="agent", request=request)

    assert exc.value.reason is SideEffectAuthorizationFailureReason.NOT_CONFIGURED
    assert len(effects) == 0


def test_l3_governance_audit_only_http_zero_physical_effects() -> None:
    effects: list[str] = []

    def record_post(handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        effects.append(body.decode("utf-8"))
        handler.send_response(200)
        handler.end_headers()

    with _local_fault_server(record_post) as (base, hosts):
        client = _allowlist_client(hosts)
        invoker, _store = _http_invoker(client)
        state = _InvokeState(policy_bundle=_audit_only_bundle())
        request = ToolExecutionRequest(
            run_id=_L3_RUN_ID,
            step_id="1",
            tool_id=HTTP_REQUEST_TOOL_ID,
            input=HttpRequestInput(
                url=f"{base}/charge",
                method="POST",
                body="{}",
                timeout_s=5.0,
            ),
            idempotency_key="l3-gov-deny-audit-only",
        )
        with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError) as exc:
            invoker.invoke(state=state, agent_id="agent", request=request)

    assert exc.value.reason is SideEffectAuthorizationFailureReason.NON_ENFORCING_MODE
    assert len(effects) == 0


def test_l3_http_idempotency_key_isolated_per_tenant() -> None:
    effects: dict[str, int] = {"a": 0, "b": 0}

    def record_tenant(handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        payload = json.loads(body.decode("utf-8"))
        effects[payload["tenant"]] += 1
        handler.send_response(200)
        handler.end_headers()
        handler.wfile.write(b"ok")

    with _local_fault_server(record_tenant) as (base, hosts):
        client = _allowlist_client(hosts)
        invoker, store = _http_invoker(client)
        shared_key = "l3-shared-tenant-key"
        for tenant, tag in ((_TENANT_A, "a"), (_TENANT_B, "b")):
            with _governed_invoke_scope(governance_tenant_id=tenant):
                state = _InvokeState(tenant_id=tenant)
                request = ToolExecutionRequest(
                    run_id=_L3_RUN_ID,
                    step_id="1",
                    tool_id=HTTP_REQUEST_TOOL_ID,
                    input=HttpRequestInput(
                        url=f"{base}/charge",
                        method="POST",
                        body=json.dumps({"tenant": tag}),
                        timeout_s=5.0,
                    ),
                    idempotency_key=shared_key,
                )
                result = invoker.invoke(state=state, agent_id="agent", request=request)
                assert result.success is True

    assert effects["a"] == 1
    assert effects["b"] == 1
    assert store.get_status(_TENANT_A, shared_key) == InvocationStatus.COMPLETED
    assert store.get_status(_TENANT_B, shared_key) == InvocationStatus.COMPLETED
    assert store.get_status("tenant_test", shared_key) is None


def test_l3_http_lost_ack_e2e_invoker_uncertain_single_effect() -> None:
    effects: list[str] = []

    def lost_ack_after_effect(handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        effects.append(body.decode("utf-8"))
        handler.request.shutdown(__import__("socket").SHUT_RDWR)

    with _local_fault_server(lost_ack_after_effect) as (base, hosts):
        client = _allowlist_client(hosts)
        invoker, store = _http_invoker(client)
        state = _InvokeState()
        request = ToolExecutionRequest(
            run_id=_L3_RUN_ID,
            step_id="1",
            tool_id=HTTP_REQUEST_TOOL_ID,
            input=HttpRequestInput(
                url=f"{base}/charge",
                method="POST",
                body=json.dumps({"amount": 1}),
                timeout_s=5.0,
            ),
            idempotency_key="l3-lost-ack-key",
        )
        result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert len(effects) == 1
    assert result.success is False
    assert result.effect_certainty == ToolEffectCertainty.UNCERTAIN
    assert store.get_status("tenant_test", "l3-lost-ack-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert len(effects) == 1


def test_l3_http_4xx_response_differs_from_lost_ack_uncertainty() -> None:
    """HTTP 4xx with full response is a definite tool outcome, not transport UNKNOWN."""

    def reject(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(b"rejected")

    with _local_fault_server(reject) as (base, hosts):
        client = _allowlist_client(hosts)
        invoker, store = _http_invoker(client)
        state = _InvokeState()
        request = ToolExecutionRequest(
            run_id=_L3_RUN_ID,
            step_id="1",
            tool_id=HTTP_REQUEST_TOOL_ID,
            input=HttpRequestInput(
                url=f"{base}/charge",
                method="POST",
                body="{}",
                timeout_s=5.0,
            ),
            idempotency_key="l3-reject-key",
        )
        result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.success is False
    assert result.output.status_code == 400
    assert result.effect_certainty == ToolEffectCertainty.COMPLETED
    assert store.get_status("tenant_test", "l3-reject-key") != InvocationStatus.UNCERTAIN


def test_mcp_subprocess_normal_echo() -> None:
    with _mcp_subprocess("ok") as (base, proc):
        assert base is not None
        result = asyncio.run(_mcp_call_tool(base, "echo", {"message": "hi"}, timeout_s=5.0))
        assert proc.poll() is None
        assert "hi" in str(result)


def test_mcp_subprocess_unavailable_exit_code() -> None:
    with _mcp_subprocess("unavailable") as (base, code):
        assert base is None
        assert code == 2


def test_mcp_subprocess_hang_client_times_out() -> None:
    with _mcp_subprocess("hang") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(_mcp_call_tool(base, "echo", {}, timeout_s=0.8))
        proc.terminate()


def test_mcp_subprocess_slow_within_deadline() -> None:
    with _mcp_subprocess("slow") as (base, _proc):
        assert base is not None
        result = asyncio.run(_mcp_call_tool(base, "echo", {}, timeout_s=5.0))
        assert "slow-ok" in str(result)


def test_mcp_subprocess_oversized_result_bounded() -> None:
    with _mcp_subprocess("huge") as (base, _proc):
        assert base is not None
        raw = asyncio.run(_mcp_call_tool(base, "echo", {}, timeout_s=5.0))
        payload = {"tool_result": str(raw)}
        bounded, truncated = bound_mcp_structured_result(payload, max_bytes=256)
        assert truncated is True
        assert bounded.get("error") == "mcp_payload_byte_limit_exceeded"


def test_mcp_subprocess_server_exit_mid_call() -> None:
    with _mcp_subprocess("crash_mid") as (base, proc):
        assert base is not None
        with pytest.raises(Exception):
            asyncio.run(_mcp_call_tool(base, "echo", {}, timeout_s=5.0))
        time.sleep(0.5)
        assert proc.poll() is not None
