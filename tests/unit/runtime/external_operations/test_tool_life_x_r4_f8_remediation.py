# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X R4 F8 — bundle isolation, Class B identity, UNKNOWN without coordinator."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)

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
from intergrax.contracts.external_operation_cancellation import ExternalOperationPhysicalState
from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.contracts.tool_execution_outcome import ToolEffectCertainty
from intergrax.integrations.providers.http_client.allowlist.client import AllowlistHttpClient
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    bind_authoritative_redrive_effect_to_operation,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.tool_external_operation_attempt import (
    tool_external_operation_identity,
)
from intergrax.runtime.external_operations.tool_external_operation_redrive_binding import (
    bind_sanctioned_redrive_effect_before_tool_physical_execution,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
)
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.http.bundle import HTTP_TOOL_IDS, register_http_tools
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.idempotent_post_qual import HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID
from intergrax.tools.providers.http.contracts import HttpRequestOutput
from intergrax.tools.providers.http.qualification_composition import (
    register_tool_life_http_class_b_qualification_tools,
)
from intergrax.tools.providers.http.service import HTTP_REQUEST_TOOL_ID
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import (
    _RecordingMseBoundary,
)
from tests.unit.tools.providers.test_tool_life_x_batch_e_real_infrastructure_adversarial import (
    _allowlist_client,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MAC_ENV = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


@pytest.fixture(autouse=True)
def _tool_life_revocation_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_MAC_ENV, "test-mac-key-tool-life-r4-f8")


_RUN_SEED = "tool-life-r4-f8-e2e"
_RUN_ID = canonical_run_id_for_tests(_RUN_SEED)
_TENANT = "test-tenant"
_IDEM_KEY = "f8-prod-entry-idem-1"
_EFFECTS: dict[str, int] = {}
_EFFECT_LOCK = threading.Lock()


class _LostAckHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        key = self.headers.get("Idempotency-Key", "").strip()
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        with _EFFECT_LOCK:
            _EFFECTS[key] = _EFFECTS.get(key, 0) + 1
            _EFFECTS[f"{key}:body"] = body
        self.request.shutdown(socket.SHUT_RDWR)


class _OkIdempotentHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        key = self.headers.get("Idempotency-Key", "").strip()
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        with _EFFECT_LOCK:
            if key not in _EFFECTS:
                _EFFECTS[key] = 1
            _EFFECTS[f"{key}:body"] = body
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


def test_production_http_bundle_excludes_qualification_class_b_tool() -> None:
    registry = ToolRegistry()
    register_http_tools(registry, ToolWiringContext())
    assert HTTP_REQUEST_TOOL_ID in HTTP_TOOL_IDS
    assert HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID not in HTTP_TOOL_IDS
    with pytest.raises(KeyError):
        registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID)


def test_isolated_qualification_composition_registers_sanctioned_class_b_tool() -> None:
    registry = ToolRegistry()
    register_tool_life_http_class_b_qualification_tools(registry, ToolWiringContext())
    registered = registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID)
    assert "tool-life-qual" in registered.contract.tags


def test_catalog_id_impersonation_without_qual_provenance_rejects_class_b_bind() -> None:
    store = InMemoryExternalOperationStateStore()
    record = store.create_if_absent(_TENANT, "op-impersonate")
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        bind_sanctioned_redrive_effect_before_tool_physical_execution(
            store,
            tenant_id=_TENANT,
            operation_id="op-impersonate",
            expected_revision=record.revision,
            catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            catalog_tool_tags=("http",),
            physical_idempotency_key=_IDEM_KEY,
            physical_effect_fingerprint="fp-1",
        )


def test_class_b_http_bind_rejects_missing_physical_identity() -> None:
    store = InMemoryExternalOperationStateStore()
    record = store.create_if_absent(_TENANT, "op-missing-id")
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        bind_authoritative_redrive_effect_to_operation(
            store,
            tenant_id=_TENANT,
            operation_id="op-missing-id",
            expected_revision=record.revision,
            trusted_effect_contract_id=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
            provider_id="http_rest",
            tool_id="http_idempotent_post",
            physical_idempotency_key=_IDEM_KEY,
            physical_effect_fingerprint=None,
        )


class _HttpTransportLossHandler:
    def execute(self, request: ToolExecutionRequest[HttpRequestInput]) -> HttpRequestOutput:
        raise RuntimeError("http_transport_error:RemoteProtocolError")


def test_invoker_http_transport_loss_without_pre_effect_coordinator_records_unknown() -> None:
    """Physical UNKNOWN when submit admitted but no idempotency pre-effect boundary."""
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="http.transport.loss.probe",
        name="probe",
        description="",
        input_schema=HttpRequestInput,
        output_schema=HttpRequestOutput,
        error_mapping={},
        side_effects=True,
    )
    registry.register(contract, _HttpTransportLossHandler())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="tool-life.r4.f8.transport")
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        external_operation_store=store,
        external_operation_owner=owner,
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_SEED)
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
        run_id=_RUN_ID,
        tool_id="http.transport.loss.probe",
        input=HttpRequestInput(url="http://127.0.0.1/effects", method="POST", body="{}"),
        step_id="step-f8-transport",
    )
    with canonical_governed_execution_scope(
        _RUN_SEED,
        governance_tenant_id=_TENANT,
        governance_principal_id="tool-life-r4-f8",
    ):
        identity = tool_external_operation_identity(
            tenant_id=_TENANT,
            tool_id="http.transport.loss.probe",
            step_id="step-f8-transport",
            physical_attempt_sequence=1,
        )
        result = invoker.invoke(state=state, agent_id="echo", request=request)
    assert result.success is False
    assert identity.operation_id is not None
    loaded = store.load(_TENANT, identity.operation_id)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN
    invoker.close()


def test_canonical_composition_production_entry_class_b_lost_ack_e2e(
    tmp_path: Path,
) -> None:
    """Canonical composition: qual registration, catalog invoke, UNKNOWN durable state, one effect."""
    _EFFECTS.clear()
    lost_server = ThreadingHTTPServer(("127.0.0.1", 0), _LostAckHandler)
    lost_thread = threading.Thread(target=lost_server.serve_forever, daemon=True)
    lost_thread.start()
    lost_host, _lost_port = lost_server.server_address
    lost_base = f"http://{lost_host}:{lost_server.server_address[1]}"
    client = _allowlist_client({lost_host})
    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=tmp_path / "f8-prod-e2e.db",
    )
    assert wiring is not None
    bundle = wiring.composition.bundle
    store = bundle.external_operation_state
    registry = ToolRegistry()
    register_http_tools(registry, ToolWiringContext(http_client=client))
    register_tool_life_http_class_b_qualification_tools(registry, ToolWiringContext(http_client=client))
    assert HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID not in HTTP_TOOL_IDS
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="tool-life.r4.f8.e2e")
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=[HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID]),
        wiring_context=ToolWiringContext(http_client=client),
        registry=registry,
    )
    invoker_bundle = build_declarative_invoker_from_tool_wiring(
        tool_wiring,
        idempotency_store=InMemoryIdempotencyStore(),
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        tool_life_execution_recovery=wiring,
        production_mode=False,
    )
    assert invoker_bundle is not None
    invoker = invoker_bundle.tool_invoker
    state = build_runtime_state_for_tests(run_id=_RUN_SEED)
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
    body = json.dumps({"action": "charge"})
    http_input_lost = HttpRequestInput(
        url=f"{lost_base}/effects",
        method="POST",
        body=body,
        timeout_s=5.0,
    )
    fingerprint = compute_invocation_operation_identity(
        HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        http_input_lost,
    ).operation_fingerprint
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        input=http_input_lost,
        step_id="step-f8-e2e",
        idempotency_key=_IDEM_KEY,
    )
    with canonical_governed_execution_scope(
        _RUN_SEED,
        governance_tenant_id=_TENANT,
        governance_principal_id="tool-life-r4-f8-e2e",
    ):
        identity = tool_external_operation_identity(
            tenant_id=_TENANT,
            tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            step_id="step-f8-e2e",
            physical_attempt_sequence=1,
        )
        first = invoker.invoke(state=state, agent_id="echo", request=request)
    assert first.success is False
    assert identity.operation_id is not None
    op_id = identity.operation_id
    after_first = store.load(_TENANT, op_id)
    assert after_first is not None
    assert after_first.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert after_first.bound_physical_idempotency_key == _IDEM_KEY
    assert after_first.bound_physical_effect_fingerprint == fingerprint
    with _EFFECT_LOCK:
        assert _EFFECTS[_IDEM_KEY] == 1
    lost_server.shutdown()
    invoker.close()

