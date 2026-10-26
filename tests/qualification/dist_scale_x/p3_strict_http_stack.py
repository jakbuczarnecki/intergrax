# © Artur Czarnecki. All rights reserved.

"""Canonical strict-host effective HTTP invoker stack for DIST-SCALE-X P3 WL-4."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import httpx

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.tool_life_execution_recovery_wiring import (
    wire_tool_life_durable_single_host_execution_recovery,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.http.bundle import register_http_tools
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.idempotent_post_qual import HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID
from intergrax.tools.providers.http.qualification_composition import (
    register_tool_life_http_class_b_qualification_tools,
)
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import build_runtime_state_for_tests, canonical_governed_execution_scope
from tests.unit.runtime.external_operations.tool_life_x_r4_class_b_http_provider_oracle import (
    ClassBHttpProviderOracle,
    class_b_http_handler_for_oracle,
)
from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import _RecordingMseBoundary
from tests.unit.tools.providers.test_tool_life_x_batch_e_real_infrastructure_adversarial import (
    _allowlist_client,
)

P3_HTTP_TOOL_ID = HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID
P3_RUN_SEED = "dist-scale-x-p3-wl4"


@dataclass
class P3HttpLoadStats:
    http_requests: int = 0
    logical_success: int = 0
    logical_failure: int = 0
    physical_attempts: int = 0


@dataclass
class P3StrictHttpStack:
    server: ThreadingHTTPServer
    thread: threading.Thread
    base_url: str
    invoker: RuntimeToolInvoker
    oracle: ClassBHttpProviderOracle
    stats: P3HttpLoadStats = field(default_factory=P3HttpLoadStats)
    _client: httpx.Client | None = None

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5.0)
        self.invoker.close()
        if self._client is not None:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                closer()


def _allowlist_client_for_host(host: str) -> httpx.Client:
    return _allowlist_client({host})


def _fault_mix_handler(
    oracle: ClassBHttpProviderOracle,
    stats: P3HttpLoadStats,
    mode: Literal["mix", "ok", "503", "slow", "lost_ack"],
    slow_seconds: float,
) -> type[BaseHTTPRequestHandler]:
    if mode == "lost_ack":
        return class_b_http_handler_for_oracle(oracle, mode="lost_ack_after_commit")
    if mode == "ok":
        return class_b_http_handler_for_oracle(oracle, mode="ok")

    ok_handler = class_b_http_handler_for_oracle(oracle, mode="ok")

    class _Handler(BaseHTTPRequestHandler):
        _seq = 0
        _lock = threading.Lock()

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            with _Handler._lock:
                _Handler._seq += 1
                seq = _Handler._seq
            stats.http_requests += 1
            bucket = seq % 100
            if mode == "503" or (mode == "mix" and bucket < 10):
                self.send_response(503)
                self.end_headers()
                return
            if mode == "slow" or (mode == "mix" and 10 <= bucket < 15):
                time.sleep(slow_seconds)
            ok_handler.do_POST(self)  # type: ignore[misc]

    return _Handler


def start_p3_strict_http_stack(
    tmp_path: Path,
    *,
    fault_mode: Literal["mix", "ok", "503", "slow", "lost_ack"] = "ok",
    slow_seconds: float = 30.0,
) -> P3StrictHttpStack:
    import os

    os.environ.setdefault("INTERGRAX_TOOL_LIFE_DURABLE_WIRING_ENABLED", "1")
    os.environ.setdefault(
        "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY",
        "dist-scale-x-p3-qual-revocation-mac",
    )
    stats = P3HttpLoadStats()
    oracle = ClassBHttpProviderOracle(db_path=tmp_path / "class_b_oracle.db")
    handler_cls = _fault_mix_handler(oracle, stats, fault_mode, slow_seconds)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    client = _allowlist_client_for_host(host)
    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=tmp_path / "tool_life_p3.db",
    )
    assert wiring is not None
    registry = ToolRegistry()
    ctx = ToolWiringContext(http_client=client)
    register_http_tools(registry, ctx)
    register_tool_life_http_class_b_qualification_tools(registry, ctx)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="dist-scale-x.p3.wl4")
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=[P3_HTTP_TOOL_ID]),
        wiring_context=ctx,
        registry=registry,
    )
    from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
        resolve_dependency_concurrency_admission_configuration,
    )
    from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
        materialize_tool_dependency_attempt_boundary,
    )

    admission = resolve_dependency_concurrency_admission_configuration(
        env.reliability_profile.dependency_concurrency_admission,
        production_mode=False,
    )
    boundary = materialize_tool_dependency_attempt_boundary(
        admission,
        production_mode=False,
    )
    bundle = build_declarative_invoker_from_tool_wiring(
        tool_wiring,
        idempotency_store=InMemoryIdempotencyStore(),
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        tool_life_execution_recovery=wiring,
        dependency_attempt_boundary=boundary,
        production_mode=False,
    )
    if bundle is None:
        raise RuntimeError("declarative invoker required for P3 WL-4")
    invoker = bundle.tool_invoker
    return P3StrictHttpStack(
        server=server,
        thread=thread,
        base_url=base_url,
        invoker=invoker,
        oracle=oracle,
        stats=stats,
        _client=client,
    )


def invoke_idempotent_post(
    stack: P3StrictHttpStack,
    *,
    tenant_id: str,
    idempotency_key: str,
    body: str,
    timeout_s: float = 5.0,
) -> bool:
    from testing_support.builder import canonical_run_id_for_tests

    run_id = canonical_run_id_for_tests(f"{P3_RUN_SEED}-{idempotency_key}")
    state = build_runtime_state_for_tests(run_id=P3_RUN_SEED)
    state.context.config.tenant_id = tenant_id
    state.request.tenant_id = tenant_id
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="dist-scale-x.p3.wl4")
    state.context.config.policy_bundle = wire_policy_bundle(
        env.model_copy(
            update={
                "policy_rules": PolicyRulesProfile(
                    inline_rules=[],
                    policy_enforcement_mode="enforce",
                ),
            },
        ),
    )
    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id=P3_HTTP_TOOL_ID,
        input=HttpRequestInput(
            url=f"{stack.base_url}/effects",
            method="POST",
            body=body,
            timeout_s=timeout_s,
            headers={"Idempotency-Key": idempotency_key},
        ),
        step_id="p3-wl4-step",
        idempotency_key=idempotency_key,
    )
    stack.stats.physical_attempts += 1
    with canonical_governed_execution_scope(
        P3_RUN_SEED,
        governance_tenant_id=tenant_id,
        governance_principal_id="dist-scale-x-p3",
    ):
        result = stack.invoker.invoke(state=state, agent_id="dist-scale-x", request=request)
    if result.success:
        stack.stats.logical_success += 1
        return True
    stack.stats.logical_failure += 1
    return False


def run_concurrent_invocations(
    stack: P3StrictHttpStack,
    *,
    duration_s: float,
    tenants: tuple[str, ...],
    concurrency: int,
) -> P3HttpLoadStats:
    import concurrent.futures
    import uuid

    stop = time.monotonic() + duration_s
    counter = 0

    def _one() -> None:
        nonlocal counter
        while time.monotonic() < stop:
            counter += 1
            tenant = tenants[counter % len(tenants)]
            key = f"idem-{uuid.uuid4().hex[:16]}"
            body = json.dumps({"n": counter})
            invoke_idempotent_post(stack, tenant_id=tenant, idempotency_key=key, body=body)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one) for _ in range(concurrency)]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
    return stack.stats
