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


_BOUND_EPOCH = 41
_RELEASE_LOST_ACK = threading.Event()
_EFFECT_COMMITTED = threading.Event()
_IDENTITY_READY = threading.Event()
_CAPTURED_OP_ID: dict[str, object] = {}


class _GatedLostAckHandler(BaseHTTPRequestHandler):
    _redrive_ok = False

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
        if type(self)._redrive_ok:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        _EFFECT_COMMITTED.set()
        _RELEASE_LOST_ACK.wait(timeout=15.0)
        self.request.shutdown(socket.SHUT_RDWR)


def _strict_durable_env() -> ApplicationEnvironmentProfile:
    from intergrax.applications.contracts.execution_mode import ExecutionMode
    from intergrax.contracts.persistence_topology import DeploymentTopology

    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="tool-life.r4.f8.strict")
    return base.model_copy(
        update={
            "meta": base.meta.model_copy(
                update={
                    "execution_mode": ExecutionMode.STRICT,
                    "deployment_topology": DeploymentTopology.SINGLE_HOST,
                    "required_persistence_topology": PersistenceTopology.DURABLE_SINGLE_HOST,
                },
            ),
        },
    )


def test_forged_qual_tags_with_substitute_handler_rejects_class_b_bind() -> None:
    registry = ToolRegistry()
    ctx = ToolWiringContext()
    registry.register(
        ToolContract(
            tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            name="forged",
            description="",
            input_schema=HttpRequestInput,
            output_schema=HttpRequestOutput,
            error_mapping={},
            side_effects=True,
            tags=("http", "tool-life-qual"),
        ),
        type(
            "ForgedHandler",
            (),
            {"execute": lambda self, request: HttpRequestOutput(success=True)},
        )(),
    )
    store = InMemoryExternalOperationStateStore()
    record = store.create_if_absent(_TENANT, "op-forged-handler")
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        bind_sanctioned_redrive_effect_before_tool_physical_execution(
            store,
            tenant_id=_TENANT,
            operation_id="op-forged-handler",
            expected_revision=record.revision,
            catalog_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            registered_tool=registry.get(HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID),
            physical_idempotency_key=_IDEM_KEY,
            physical_effect_fingerprint="fp-forged",
        )


def test_canonical_strict_host_class_b_lost_ack_recovery_single_effect_e2e(
    tmp_path: Path,
) -> None:
    """Strict profile + qual registration + invoker UNKNOWN + certified loss + redrive invoke."""
    from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
    from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationIntentState,
    )
    from intergrax.runtime.external_operations.recovery_external_operation_gate import (
        admit_unknown_external_operation_for_reconciliation,
    )
    from intergrax.runtime.external_operations.redrive_external_operation_authority import (
        transfer_redrive_authority_after_reconciliation,
    )
    from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
        bind_worker_execution_ownership,
        build_worker_execution_certified_owner_loss_certifier,
        configure_worker_execution_certified_owner_loss,
    )
    from tests.unit.autonomous_work.repository_contracts import worker_instance
    from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
        issue_sqlite_commit_authority_loss_decision_for_bound_operation,
    )
    from intergrax.runtime.external_operations.certified_owner_loss_composition import (
        require_certified_owner_loss_composition,
    )
    from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
        QualificationWorkerExecutionRevocationSupervision,
    )

    _EFFECTS.clear()
    _RELEASE_LOST_ACK.clear()
    _EFFECT_COMMITTED.clear()
    _IDENTITY_READY.clear()
    _CAPTURED_OP_ID.clear()
    lost_server = ThreadingHTTPServer(("127.0.0.1", 0), _GatedLostAckHandler)
    lost_thread = threading.Thread(target=lost_server.serve_forever, daemon=True)
    lost_thread.start()
    lost_host, lost_port = lost_server.server_address
    lost_base = f"http://{lost_host}:{lost_port}"
    allow_hosts = {lost_host}
    client = _allowlist_client(allow_hosts)

    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=tmp_path / "f8-strict-redrive-e2e.db",
    )
    assert wiring is not None
    bundle = wiring.composition.bundle
    store = bundle.external_operation_state
    repo = InMemoryWorkerInstanceRepository()
    configure_worker_execution_certified_owner_loss(
        worker_execution_ownership_revocation=bundle.worker_execution_ownership_revocation,
        worker_instance_repository=repo,
        authoritative_state_store=store,
        worker_execution_revocation_supervision=QualificationWorkerExecutionRevocationSupervision(),
    )

    registry = ToolRegistry()
    register_http_tools(registry, ToolWiringContext(http_client=client))
    register_tool_life_http_class_b_qualification_tools(registry, ToolWiringContext(http_client=client))

    env = _strict_durable_env()
    assert env.execution_mode.value == "strict"
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
        timeout_s=10.0,
    )
    fingerprint = compute_invocation_operation_identity(
        HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        http_input_lost,
    ).operation_fingerprint
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        input=http_input_lost,
        step_id="step-f8-strict-redrive",
        idempotency_key=_IDEM_KEY,
    )
    worker = worker_instance()
    repo.create(worker)

    invoke_outcome: list[object] = []

    def _invoke_first() -> None:
        try:
            with canonical_governed_execution_scope(
                _RUN_SEED,
                governance_tenant_id=_TENANT,
                governance_principal_id="tool-life-r4-f8-strict",
            ):
                from intergrax.contracts.execution_identity import (
                    require_active_execution_id,
                    require_active_execution_identity,
                )

                identity = tool_external_operation_identity(
                    tenant_id=_TENANT,
                    tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
                    step_id="step-f8-strict-redrive",
                    physical_attempt_sequence=1,
                )
                assert identity.operation_id is not None
                run_id, attempt_id = require_active_execution_identity()
                _CAPTURED_OP_ID["op_id"] = identity.operation_id
                _CAPTURED_OP_ID["run_id"] = run_id
                _CAPTURED_OP_ID["attempt_id"] = attempt_id
                _CAPTURED_OP_ID["execution_id"] = require_active_execution_id()
                _IDENTITY_READY.set()
                invoke_outcome.append(
                    invoker.invoke(state=state, agent_id="echo", request=request),
                )
        except BaseException as exc:  # noqa: BLE001
            invoke_outcome.append(exc)

    t = threading.Thread(target=_invoke_first, daemon=True)
    t.start()
    assert _IDENTITY_READY.wait(timeout=10.0), invoke_outcome
    op_id = str(_CAPTURED_OP_ID["op_id"])
    assert _EFFECT_COMMITTED.wait(timeout=10.0), invoke_outcome
    running = store.load(_TENANT, op_id)
    assert running is not None
    assert running.physical_state is ExternalOperationPhysicalState.RUNNING
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=running.revision,
        worker_instance_id=str(worker.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    issue_sqlite_commit_authority_loss_decision_for_bound_operation(
        bundle,
        tenant_id=_TENANT,
        operation_id=op_id,
    )
    recorder = require_certified_owner_loss_composition().worker_execution_revocation_recorder
    assert recorder is not None
    proof = recorder.record_revocation_for_bound_operation(
        tenant_id=_TENANT,
        operation_id=op_id,
    ).proof_ref
    certifier = build_worker_execution_certified_owner_loss_certifier(store=store)
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=op_id,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = store.apply_certified_owner_loss(
        _TENANT,
        op_id,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    _RELEASE_LOST_ACK.set()
    t.join(timeout=20.0)
    assert not t.is_alive()
    after_loss = store.load(_TENANT, op_id)
    assert after_loss is not None
    assert after_loss.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert unknown.consumed_certified_loss_evidence_ids
    admit_unknown_external_operation_for_reconciliation(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
    )
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    transferred = transfer_redrive_authority_after_reconciliation(
        store,
        tenant_id=_TENANT,
        operation_id=op_id,
        expected_revision=unknown.revision,
        trusted_effect_contract=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
        worker_instance_id=str(recovery_worker.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH + 1,
        provider_id="http_rest",
        tool_id="http_idempotent_post",
        physical_idempotency_key=_IDEM_KEY,
        physical_effect_fingerprint=fingerprint,
    )
    assert transferred.physical_state is ExternalOperationPhysicalState.NOT_STARTED
    _GatedLostAckHandler._redrive_ok = True
    redrive_request = ToolExecutionRequest(
        run_id=_RUN_ID,
        tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        input=http_input_lost,
        step_id="step-f8-strict-redrive",
        idempotency_key=_IDEM_KEY,
    )
    from tests.unit.runtime.governance.gr3_test_support import bound_gr3_active_execution

    with bound_gr3_active_execution(
        run_id=_CAPTURED_OP_ID["run_id"],
        attempt_id=_CAPTURED_OP_ID["attempt_id"],
        execution_id=_CAPTURED_OP_ID["execution_id"],
        governance_tenant_id=_TENANT,
        governance_principal_id="tool-life-r4-f8-strict-redrive",
    ):
        redrive = invoker.invoke_sanctioned_physical_redrive(
            state=state,
            agent_id="echo",
            request=redrive_request,
            physical_attempt_sequence=2,
            durable_operation_id=op_id,
        )
    assert redrive.success is True
    with _EFFECT_LOCK:
        assert _EFFECTS[_IDEM_KEY] == 1
    terminal = store.load(_TENANT, op_id)
    assert terminal is not None
    assert terminal.physical_state is ExternalOperationPhysicalState.SUCCEEDED
    assert terminal.intent_state in (
        ExternalOperationIntentState.ACTIVE,
        ExternalOperationIntentState.TERMINATED,
    )
    lost_server.shutdown()
    invoker.close()


def test_generic_http_class_c_still_executable_and_redrive_denied(tmp_path: Path) -> None:
    from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
        TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    )
    from intergrax.runtime.external_operations.redrive_external_operation_authority import (
        transfer_redrive_authority_after_reconciliation,
    )
    from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
        _OP,
        _TENANT as _L3_TENANT,
        _running_with_worker,
    )
    from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository

    registry = ToolRegistry()
    register_http_tools(registry, ToolWiringContext())
    contract = registry.get(HTTP_REQUEST_TOOL_ID).contract
    assert contract.side_effect_retry_safety is not None
    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=tmp_path / "f8-class-c.db",
    )
    assert wiring is not None
    bundle = wiring.composition.bundle
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(
        bundle,
        repo,
        bind_redrive_contract_id=TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id,
        bind_provider_id="http_rest",
        bind_tool_id="http_blind_post",
    )
    from tests.unit.autonomous_work.repository_contracts import worker_instance as _worker_instance

    recovery_worker = _worker_instance()
    repo.create(recovery_worker)
    loaded = bundle.external_operation_state.load(_L3_TENANT, _OP)
    assert loaded is not None
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        transfer_redrive_authority_after_reconciliation(
            bundle.external_operation_state,
            tenant_id=_L3_TENANT,
            operation_id=_OP,
            expected_revision=loaded.revision,
            trusted_effect_contract=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=12,
            provider_id="http_rest",
            tool_id="http_idempotent_post",
        )
    assert registry.get(HTTP_REQUEST_TOOL_ID).contract.side_effects is True

