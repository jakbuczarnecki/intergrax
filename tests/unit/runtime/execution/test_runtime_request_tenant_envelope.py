# © Artur Czarnecki. All rights reserved.

"""RuntimeRequest tenant fail-closed envelope semantics (EBH-4-R1-R3 P7)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from testing_support.builder import canonical_task_id_for_tests
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest
from testing_support.builder import build_runtime_request_for_tests

pytestmark = pytest.mark.unit


def _base_request(**overrides: object) -> RuntimeRequest:
    req = build_runtime_request_for_tests(
        seed="tenant-envelope",
        agent_id="echo",
        user_id="user-a",
        session_id="sess-a",
        message="hello",
        tenant_id="tenant-a",
    )
    data = {
        "agent_id": req.agent_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "message": req.message,
        "task_id": req.task_id,
        "run_id": req.run_id,
        "tenant_id": req.tenant_id,
        "metadata": dict(req.metadata),
    }
    data.update(overrides)
    return RuntimeRequest(**data)


def test_runtime_request_envelope_preserves_explicit_tenant_a() -> None:
    envelope = _base_request(tenant_id="tenant-a").to_envelope()
    assert envelope.tenant_id == "tenant-a"


def test_runtime_request_envelope_preserves_explicit_tenant_b() -> None:
    envelope = _base_request(tenant_id="tenant-b").to_envelope()
    assert envelope.tenant_id == "tenant-b"


def test_runtime_request_envelope_rejects_metadata_tenant_override() -> None:
    with pytest.raises(ValueError, match="metadata tenant_id cannot override"):
        _base_request(
            tenant_id="tenant-a",
            metadata={"tenant_id": "tenant-b"},
        ).to_envelope()


def test_runtime_request_envelope_fail_closed_without_tenant() -> None:
    with pytest.raises(ValueError, match="tenant_id is required"):
        _base_request(tenant_id=None).to_envelope()


def test_runtime_request_envelope_no_implicit_default_tenant() -> None:
    req = RuntimeRequest(
        agent_id="echo",
        user_id="user",
        session_id="sess",
        message="m",
        task_id=canonical_task_id_for_tests("tenant-default-probe"),
        run_id=mint_run_id(),
        tenant_id=None,
        metadata={},
    )
    with pytest.raises(ValueError, match="tenant_id is required"):
        req.to_envelope()
