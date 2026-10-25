# © Artur Czarnecki. All rights reserved.

"""EXT-L4 B2+F6 — integrated loopback HTTP redrive with durable store (single proof)."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationFencedCommitRejectedError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    mark_physical_terminal_if_owner_matches,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    admit_unknown_external_operation_for_reconciliation,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    transfer_redrive_authority_after_reconciliation,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    build_worker_execution_certified_owner_loss_certifier,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance
from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
    _BOUND_EPOCH,
    _OP,
    _TENANT,
    _revoke_bound,
    _running_with_worker,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_HTTP_PROVIDER = "http_rest"
_HTTP_TOOL_IDEMPOTENT = "http_request"
_HTTP_TOOL_BLIND = "http_blind_post"
_IDEMPOTENCY_KEY = "integrated-http-redrive-qual-1"

_EFFECTS: dict[str, dict[str, Any]] = {}
_EFFECT_LOCK = threading.Lock()


class _IdempotentPostHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        key = self.headers.get("Idempotency-Key", "").strip()
        if not key:
            self.send_response(400)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        with _EFFECT_LOCK:
            existing = _EFFECTS.get(key)
            if existing is None:
                _EFFECTS[key] = {
                    "status": "applied",
                    "body": body.decode("utf-8", errors="replace"),
                    "count": 1,
                }
                payload = _EFFECTS[key]
            else:
                existing["count"] = int(existing.get("count", 1))
                payload = existing
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def _post_effect(base_url: str, *, idempotency_key: str) -> httpx.Response:
    with httpx.Client(timeout=5.0) as client:
        return client.post(
            f"{base_url}/effects",
            headers={"Idempotency-Key": idempotency_key},
            content=b'{"action":"charge"}',
        )


def _post_effect_transport_loss_after_commit(base_url: str, *, idempotency_key: str) -> None:
    """Real transport loss: provider commits, then resets connection before HTTP response."""
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"{base_url}/effects",
                headers={"Idempotency-Key": idempotency_key},
                content=b'{"action":"charge"}',
            )
    except httpx.HTTPError:
        return
    raise AssertionError("expected transport failure after provider-side commit")


class _LostAckAfterCommitHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        key = self.headers.get("Idempotency-Key", "").strip()
        if not key:
            self.send_response(400)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        with _EFFECT_LOCK:
            existing = _EFFECTS.get(key)
            if existing is None:
                _EFFECTS[key] = {
                    "status": "applied",
                    "body": body.decode("utf-8", errors="replace"),
                    "count": 1,
                }
            else:
                existing["count"] = int(existing.get("count", 1))
        self.request.shutdown(socket.SHUT_RDWR)


@pytest.fixture
def idempotent_http_server() -> str:
    _EFFECTS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _IdempotentPostHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    yield base
    server.shutdown()


@pytest.fixture
def lost_ack_http_server() -> str:
    _EFFECTS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LostAckAfterCommitHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    yield base
    server.shutdown()


def test_integrated_http_class_b_lost_ack_redrive_single_physical_effect(
    tmp_path: Path,
    lost_ack_http_server: str,
    idempotent_http_server: str,
) -> None:
    """Steps 1–14: bind → RUNNING → lost-ack HTTP → no SUCCESS → UNKNOWN → redrive → one effect."""
    bundle = SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "integrated-http-redrive.db")
    repo = InMemoryWorkerInstanceRepository()
    initial_worker = worker_instance()
    repo.create(initial_worker)
    _worker, _bound, _worker_id, initial_owner = _running_with_worker(
        bundle,
        repo,
        bind_redrive_contract_id=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id,
        bind_provider_id=_HTTP_PROVIDER,
        bind_tool_id=_HTTP_TOOL_IDEMPOTENT,
        bind_physical_idempotency_key=_IDEMPOTENCY_KEY,
    )
    _post_effect_transport_loss_after_commit(
        lost_ack_http_server,
        idempotency_key=_IDEMPOTENCY_KEY,
    )
    with _EFFECT_LOCK:
        assert _EFFECTS[_IDEMPOTENCY_KEY]["count"] == 1
    after_loss = bundle.external_operation_state.load(_TENANT, _OP)
    assert after_loss is not None
    assert after_loss.physical_state is ExternalOperationPhysicalState.RUNNING
    assert after_loss.physical_state is not ExternalOperationPhysicalState.SUCCEEDED
    proof = _revoke_bound(bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = recover_running_operation_after_certified_owner_loss(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    admit_unknown_external_operation_for_reconciliation(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
    )
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    transferred = transfer_redrive_authority_after_reconciliation(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=unknown.revision,
        trusted_effect_contract=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
        worker_instance_id=str(recovery_worker.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH + 1,
        provider_id=_HTTP_PROVIDER,
        tool_id=_HTTP_TOOL_IDEMPOTENT,
        physical_idempotency_key=_IDEMPOTENCY_KEY,
    )
    assert transferred.physical_state is ExternalOperationPhysicalState.NOT_STARTED
    fresh_owner = ProcessLocalExternalOperationOwner.mint()
    running = mark_physical_running(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=transferred.revision,
        owner=fresh_owner,
    )
    assert running.physical_state is ExternalOperationPhysicalState.RUNNING
    redrive_ack = _post_effect(idempotent_http_server, idempotency_key=_IDEMPOTENCY_KEY)
    assert redrive_ack.status_code == 200
    with _EFFECT_LOCK:
        assert _EFFECTS[_IDEMPOTENCY_KEY]["count"] == 1
    terminal = mark_physical_terminal_if_owner_matches(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        physical_state=ExternalOperationPhysicalState.SUCCEEDED,
        owner=fresh_owner,
    )
    assert terminal.physical_state is ExternalOperationPhysicalState.SUCCEEDED
    assert terminal.intent_state is ExternalOperationIntentState.ACTIVE
    with pytest.raises(ExternalOperationFencedCommitRejectedError):
        mark_physical_terminal_if_owner_matches(
            bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=terminal.revision,
            physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            owner=initial_owner,
        )
    stale_owner = ProcessLocalExternalOperationOwner.mint()
    with pytest.raises(ExternalOperationFencedCommitRejectedError):
        mark_physical_terminal_if_owner_matches(
            bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=terminal.revision,
            physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            owner=stale_owner,
        )


def test_integrated_http_class_c_bound_operation_rejects_class_b_redrive(
    tmp_path: Path,
) -> None:
    bundle = SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "integrated-http-class-c.db")
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(
        bundle,
        repo,
        bind_redrive_contract_id=TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id,
        bind_provider_id=_HTTP_PROVIDER,
        bind_tool_id=_HTTP_TOOL_BLIND,
    )
    proof = _revoke_bound(bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = recover_running_operation_after_certified_owner_loss(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    admit_unknown_external_operation_for_reconciliation(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
    )
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        transfer_redrive_authority_after_reconciliation(
            bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            trusted_effect_contract=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
            provider_id=_HTTP_PROVIDER,
            tool_id=_HTTP_TOOL_IDEMPOTENT,
            physical_idempotency_key=_IDEMPOTENCY_KEY,
        )


def test_integrated_http_class_b_substitution_after_class_c_bind_rejected(
    tmp_path: Path,
) -> None:
    bundle = SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "integrated-http-substitution.db")
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(
        bundle,
        repo,
        bind_redrive_contract_id=TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT.contract_id,
        bind_provider_id=_HTTP_PROVIDER,
        bind_tool_id=_HTTP_TOOL_BLIND,
    )
    proof = _revoke_bound(bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    unknown = recover_running_operation_after_certified_owner_loss(
        bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=staged.staged_certified_owner_loss.evidence.evidence_id,
    )
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        transfer_redrive_authority_after_reconciliation(
            bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            trusted_effect_contract=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
            provider_id=_HTTP_PROVIDER,
            tool_id=_HTTP_TOOL_IDEMPOTENT,
            physical_idempotency_key=_IDEMPOTENCY_KEY,
        )
