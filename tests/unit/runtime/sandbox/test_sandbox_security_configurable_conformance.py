# © Artur Czarnecki. All rights reserved.

"""Sandbox security configuration seam conformance tests (AW-7C P0-3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.sandbox_host import SandboxArtifact, SandboxHostBackend, SandboxSession
from intergrax.runtime.sandbox.contracts import (
    SandboxSecurityCapabilities,
    SandboxSecurityCapable,
    SandboxSecurityConfigurable,
    SandboxSecurityRequirements,
)
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ALLOWLIST = canonicalize_network_egress_allowlist(["https://approved.example.com"])


class _LegacySandboxHostBackend:
    def __init__(self) -> None:
        self.create_session_calls = 0

    def create_session(self) -> SandboxSession:
        self.create_session_calls += 1
        return SandboxSession(session_id="legacy-1")

    def exec(self, session_id: str, command: str):
        return MagicMock(exit_code=0, stdout="", stderr="")

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        return SandboxArtifact(artifact_id="artifact-1")


class _QualifiedSandboxHostBackend:
    def __init__(self) -> None:
        self.create_session_calls = 0
        self.create_session_with_security_calls = 0

    def create_session(self) -> SandboxSession:
        self.create_session_calls += 1
        return SandboxSession(session_id="legacy-1")

    def create_session_with_security(
        self,
        requirements: SandboxSecurityRequirements,
    ) -> SandboxSession:
        self.create_session_with_security_calls += 1
        return SandboxSession(session_id="qualified-1")

    def exec(self, session_id: str, command: str):
        return MagicMock(exit_code=0, stdout="", stderr="")

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        return SandboxArtifact(artifact_id="artifact-1")

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier="cloud",
            provider_id="qualified-hosted",
            network_egress_allowlist_enforced=True,
            enforced_network_hosts=_ALLOWLIST,
        )


def test_legacy_backend_remains_valid_sandbox_host_backend() -> None:
    backend = _LegacySandboxHostBackend()
    assert isinstance(backend, SandboxHostBackend)
    assert not isinstance(backend, SandboxSecurityConfigurable)
    assert not isinstance(backend, SandboxSecurityCapable)


def test_allowlist_admission_requires_security_configurable_backend() -> None:
    backend = _LegacySandboxHostBackend()
    requirements = SandboxSecurityRequirements(
        isolation_tier="cloud",
        network_egress="allowlist",
        network_egress_allowlist=_ALLOWLIST,
    )
    session = HostedSandboxSession.open(
        backend,
        tenant_id="tenant-a",
        task_id="task-a",
        security_requirements=requirements,
    )
    assert session is None
    assert backend.create_session_calls == 0


def test_qualified_backend_uses_security_configurable_session_creation() -> None:
    backend = _QualifiedSandboxHostBackend()
    assert isinstance(backend, SandboxSecurityConfigurable)
    requirements = SandboxSecurityRequirements(
        isolation_tier="cloud",
        network_egress="allowlist",
        network_egress_allowlist=_ALLOWLIST,
    )
    session = HostedSandboxSession.open(
        backend,
        tenant_id="tenant-a",
        task_id="task-a",
        security_requirements=requirements,
    )
    assert isinstance(session, HostedSandboxSession)
    assert session.session_id == "qualified-1"
    assert backend.create_session_with_security_calls == 1
    assert backend.create_session_calls == 0
    assert isinstance(session, SandboxSecurityCapable)
    assert session.security_capabilities().network_egress_allowlist_enforced is True
