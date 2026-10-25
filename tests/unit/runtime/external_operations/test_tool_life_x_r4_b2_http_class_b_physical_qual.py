# © Artur Czarnecki. All rights reserved.

"""R4 B2-A — isolated HTTP server idempotent POST physical qual (G-A02 Class B path)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveEffectClass,
    ExternalOperationRedriveEffectClassRejectedError,
    resolve_authoritative_redrive_effect_class,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    transfer_redrive_authority_after_reconciliation,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

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


def test_http_physical_idempotent_post_replay_does_not_duplicate_effect(
    idempotent_http_server: str,
) -> None:
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED
    )
    key = "qual-op-1"
    with httpx.Client(timeout=5.0) as client:
        first = client.post(
            f"{idempotent_http_server}/effects",
            headers={"Idempotency-Key": key},
            content=b'{"action":"charge"}',
        )
        second = client.post(
            f"{idempotent_http_server}/effects",
            headers={"Idempotency-Key": key},
            content=b'{"action":"charge"}',
        )
    assert first.status_code == 200
    assert second.status_code == 200
    with _EFFECT_LOCK:
        assert _EFFECTS[key]["count"] == 1


def test_http_blind_post_contract_is_class_c_fail_closed() -> None:
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED
    )


def test_redrive_transfer_rejects_class_c_http_contract() -> None:
    store = InMemoryExternalOperationStateStore()
    with pytest.raises(ExternalOperationRedriveEffectClassRejectedError):
        transfer_redrive_authority_after_reconciliation(
            store,
            tenant_id="tenant-a",
            operation_id="missing-op",
            expected_revision=0,
            trusted_effect_contract=TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
            worker_instance_id="worker-recovery",
            worker_ownership_epoch=2,
        )
