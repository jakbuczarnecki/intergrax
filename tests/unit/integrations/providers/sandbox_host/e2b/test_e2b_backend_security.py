# © Artur Czarnecki. All rights reserved.

"""E2B sandbox host backend security behavior unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.integrations.contracts.sandbox_host import SandboxExecResult
from intergrax.integrations.providers.sandbox_host.e2b.backend import E2bSandboxHostBackend
from intergrax.integrations.providers.sandbox_host.e2b.client import E2bCreatedSandbox
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig
from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxSecurityError
from intergrax.integrations.providers.sandbox_host.e2b.network_policy import (
    E2bNetworkCreatePayload,
    E2bProviderNetworkState,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityRequirements
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

pytestmark = pytest.mark.unit

_ALLOWLIST = canonicalize_network_egress_allowlist(["https://approved.example.com"])
_REQUIREMENTS = SandboxSecurityRequirements(
    isolation_tier="cloud",
    network_egress="allowlist",
    network_egress_allowlist=_ALLOWLIST,
)


@dataclass
class _FakeE2bClient:
    created_network: list[E2bNetworkCreatePayload] = field(default_factory=list)
    destroyed: list[str] = field(default_factory=list)
    network_states: dict[str, E2bProviderNetworkState] = field(default_factory=dict)
    create_should_fail: bool = False
    info_should_fail: bool = False
    sessions: dict[str, str] = field(default_factory=dict)

    def create_sandbox(
        self,
        *,
        template_id: str,
        timeout_seconds: int,
        network: E2bNetworkCreatePayload | None = None,
    ) -> E2bCreatedSandbox:
        if self.create_should_fail:
            raise RuntimeError("provider rejected policy")
        if network is not None:
            self.created_network.append(network)
        sandbox_id = f"sbx-{len(self.sessions) + 1}"
        self.sessions[sandbox_id] = sandbox_id
        return E2bCreatedSandbox(sandbox_id=sandbox_id)

    def get_sandbox_network_state(self, sandbox_id: str) -> E2bProviderNetworkState:
        if self.info_should_fail:
            raise RuntimeError("info unavailable")
        state = self.network_states.get(sandbox_id)
        if state is None:
            raise RuntimeError("missing state")
        return state

    def exec_command(self, sandbox_id: str, command: str) -> SandboxExecResult:
        return SandboxExecResult(exit_code=0, stdout=command, stderr="")

    def upload_file(self, sandbox_id: str, *, local_path: str, remote_path: str) -> str:
        return remote_path

    def destroy_sandbox(self, sandbox_id: str) -> None:
        self.destroyed.append(sandbox_id)

    def health(self) -> bool:
        return True


def _backend(client: _FakeE2bClient) -> E2bSandboxHostBackend:
    config = E2bSandboxHostConfig(api_key="test-key", template_id="base")
    return E2bSandboxHostBackend(client=client, config=config)


def test_provider_create_rejects_policy_no_session() -> None:
    client = _FakeE2bClient(create_should_fail=True)
    backend = _backend(client)
    with pytest.raises(RuntimeError):
        backend.create_session_with_security(_REQUIREMENTS)
    assert client.destroyed == []


def test_provider_info_read_fails_cleanup_and_no_admission() -> None:
    client = _FakeE2bClient()
    client.network_states["sbx-1"] = E2bProviderNetworkState(
        allow_out=("approved.example.com",),
        deny_out=("0.0.0.0/0",),
    )
    client.info_should_fail = True
    backend = _backend(client)
    with pytest.raises(E2bSandboxSecurityError, match="attestation failed"):
        backend.create_session_with_security(_REQUIREMENTS)
    assert client.destroyed == ["sbx-1"]


def test_provider_broader_policy_cleanup_and_no_admission() -> None:
    client = _FakeE2bClient()
    client.network_states["sbx-1"] = E2bProviderNetworkState(
        allow_out=("approved.example.com", "extra.example.com"),
        deny_out=("0.0.0.0/0",),
    )
    backend = _backend(client)
    with pytest.raises(E2bSandboxSecurityError):
        backend.create_session_with_security(_REQUIREMENTS)
    assert client.destroyed == ["sbx-1"]


def test_successful_admission_exposes_session_bound_evidence() -> None:
    client = _FakeE2bClient()
    client.network_states["sbx-1"] = E2bProviderNetworkState(
        allow_out=("approved.example.com",),
        deny_out=("0.0.0.0/0",),
    )
    backend = _backend(client)
    session = backend.create_session_with_security(_REQUIREMENTS)
    evidence = backend.session_security_capabilities(session.session_id)
    assert evidence.provider_id == "e2b"
    assert evidence.network_egress_allowlist_enforced is True
    assert evidence.enforced_network_hosts == _ALLOWLIST


def test_concurrent_sessions_keep_isolated_evidence() -> None:
    client = _FakeE2bClient()
    allow_a = canonicalize_network_egress_allowlist(["https://a.example.com"])
    allow_b = canonicalize_network_egress_allowlist(["https://b.example.com"])
    backend = _backend(client)

    def _prime_state(sandbox_id: str, hostname: str) -> None:
        client.network_states[sandbox_id] = E2bProviderNetworkState(
            allow_out=(hostname,),
            deny_out=("0.0.0.0/0",),
        )

    _prime_state("sbx-1", "a.example.com")
    session_a = backend.create_session_with_security(
        SandboxSecurityRequirements(
            isolation_tier="cloud",
            network_egress="allowlist",
            network_egress_allowlist=allow_a,
        ),
    )
    _prime_state("sbx-2", "b.example.com")
    session_b = backend.create_session_with_security(
        SandboxSecurityRequirements(
            isolation_tier="cloud",
            network_egress="allowlist",
            network_egress_allowlist=allow_b,
        ),
    )

    evidence_a = backend.session_security_capabilities(session_a.session_id)
    evidence_b = backend.session_security_capabilities(session_b.session_id)
    assert evidence_a.enforced_network_hosts == allow_a
    assert evidence_b.enforced_network_hosts == allow_b
