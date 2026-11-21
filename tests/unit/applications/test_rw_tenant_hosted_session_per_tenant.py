# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any

import pytest
from intergrax.applications._shared.sandbox_host_wiring import resolve_hosted_sandbox_session
from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.sandbox.hosted_session import (
    LAB_ONLY_HOSTED_SANDBOX_TASK_ID,
    LAB_ONLY_HOSTED_SANDBOX_TENANT_ID,
    HostedSandboxSession,
)

pytestmark = pytest.mark.unit


class _FakeSandboxHost:
    def __init__(self) -> None:
        self.sessions: list[dict[str, str]] = []

    def create_session(self) -> dict[str, str]:
        session = {"session_id": f"remote-{len(self.sessions) + 1}", "status": "running"}
        self.sessions.append(session)
        return session

    def exec(self, session_id: str, command: str) -> dict[str, Any]:
        return {"exit_code": 0, "stdout": command, "stderr": ""}

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> dict[str, str]:
        return {"artifact_id": remote_name, "uri": f"hosted://{session_id}/{remote_name}"}

    def health(self) -> bool:
        return True


def test_ten_01d_blank_labels_refused() -> None:
    backend = create_e2b_sandbox_host(client=_FakeSandboxHost())
    with pytest.raises(ValueError, match="tenant_id"):
        HostedSandboxSession.open(backend, tenant_id="  ", task_id="task")
    with pytest.raises(ValueError, match="task_id"):
        HostedSandboxSession.open(backend, tenant_id="tenant", task_id="")


def test_ten_01a_host_opens_session_under_host_tenant() -> None:
    profile = IntegrationProfile(sandbox_host=create_e2b_sandbox_host(client=_FakeSandboxHost()))
    session = resolve_hosted_sandbox_session(profile, tenant_id="tenant-a", task_id="my_lab")
    assert session is not None
    assert session.tenant_id == "tenant-a"
    assert session.task_id == "my_lab"
    assert session.tenant_id != LAB_ONLY_HOSTED_SANDBOX_TENANT_ID
    assert session.task_id != LAB_ONLY_HOSTED_SANDBOX_TASK_ID


def test_ten_01b_two_tenants_two_backend_sessions() -> None:
    fake = _FakeSandboxHost()
    profile = IntegrationProfile(sandbox_host=create_e2b_sandbox_host(client=fake))
    first = resolve_hosted_sandbox_session(profile, tenant_id="tenant-a", task_id="lab_a")
    second = resolve_hosted_sandbox_session(profile, tenant_id="tenant-b", task_id="lab_b")
    assert first is not None and second is not None
    assert first.tenant_id == "tenant-a"
    assert second.tenant_id == "tenant-b"
    assert len(fake.sessions) == 2
