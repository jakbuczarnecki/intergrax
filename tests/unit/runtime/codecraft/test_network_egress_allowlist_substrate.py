# © Artur Czarnecki. All rights reserved.

"""CodeCraft substrate network egress allowlist fail-closed qualification tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.integrations.contracts.sandbox_host import SandboxArtifact, SandboxHostBackend
from intergrax.runtime.codecraft.substrate import resolve_craft_sandbox
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities, SandboxSecurityCapable
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist, parse_network_egress_host
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
TASK = "task-a"
_REQUESTED = canonicalize_network_egress_allowlist(["https://approved.example.com"])
_APPROVED = parse_network_egress_host("https://approved.example.com")
_UNAPPROVED = parse_network_egress_host("https://blocked.example.com")

_CODECRAFT_OPS = frozenset(
    {"echo", "write_file", "read_file", "list_files", "run_python", "run_script"},
)


class _PlainSandboxHostBackend:
    def create_session(self):
        return MagicMock(session_id="hosted-plain")

    def exec(self, session_id: str, command: str):
        return MagicMock(exit_code=0, stdout="", stderr="")

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        return SandboxArtifact(artifact_id="artifact-1")


class _HostedSecurityBackend:
    def __init__(
        self,
        *,
        provider_id: str = "fake-hosted",
        network_egress_deny_enforced: bool | None = None,
        network_egress_allowlist_enforced: bool | None = None,
        enforced_network_hosts= None,
    ) -> None:
        self._provider_id = provider_id
        self._network_egress_deny_enforced = network_egress_deny_enforced
        self._network_egress_allowlist_enforced = network_egress_allowlist_enforced
        self._enforced_network_hosts = enforced_network_hosts

    def create_session(self):
        return MagicMock(session_id="hosted-1")

    def exec(self, session_id: str, command: str):
        return MagicMock(exit_code=0, stdout="", stderr="")

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        return SandboxArtifact(artifact_id="artifact-1")

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier="cloud",
            provider_id=self._provider_id,
            network_egress_deny_enforced=self._network_egress_deny_enforced,
            network_egress_allowlist_enforced=self._network_egress_allowlist_enforced,
            enforced_network_hosts=self._enforced_network_hosts,
        )


def _allowlist_profile() -> CodeCraftProfile:
    return CodeCraftProfile(
        mode="autonomous",
        isolation_tier="cloud",
        network_egress="allowlist",
        network_egress_allowlist=_REQUESTED.hosts,
    )


def test_provider_deny_proof_accepted_for_deny() -> None:
    backend = _HostedSecurityBackend(network_egress_deny_enforced=True)
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession)
    assert resolution.capabilities is not None
    assert resolution.capabilities.network_egress_deny_enforced is True


def test_missing_deny_proof_rejected() -> None:
    backend = _PlainSandboxHostBackend()
    assert isinstance(backend, SandboxHostBackend)
    assert not isinstance(backend, SandboxSecurityCapable)
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_requirement_unsatisfied"


def test_provider_exact_allowlist_proof_accepted() -> None:
    backend = _HostedSecurityBackend(
        network_egress_allowlist_enforced=True,
        enforced_network_hosts=_REQUESTED,
    )
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": _allowlist_profile()}),
        _allowlist_profile(),
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession)
    assert resolution.capabilities is not None
    assert resolution.capabilities.network_egress_allowlist_enforced is True


def test_missing_allowlist_proof_rejected() -> None:
    backend = _HostedSecurityBackend(network_egress_deny_enforced=True)
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": _allowlist_profile()}),
        _allowlist_profile(),
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_allowlist_requirement_unsatisfied"


def test_wrong_host_set_rejected() -> None:
    backend = _HostedSecurityBackend(
        network_egress_allowlist_enforced=True,
        enforced_network_hosts=canonicalize_network_egress_allowlist([_UNAPPROVED]),
    )
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": _allowlist_profile()}),
        _allowlist_profile(),
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_allowlist_requirement_unsatisfied"


def test_superset_host_set_rejected() -> None:
    backend = _HostedSecurityBackend(
        network_egress_allowlist_enforced=True,
        enforced_network_hosts=canonicalize_network_egress_allowlist([_APPROVED, _UNAPPROVED]),
    )
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": _allowlist_profile()}),
        _allowlist_profile(),
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_allowlist_requirement_unsatisfied"


def test_allowlist_requested_unrestricted_provider_denied() -> None:
    backend = _HostedSecurityBackend(
        network_egress_deny_enforced=False,
        network_egress_allowlist_enforced=False,
        enforced_network_hosts=None,
    )
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": _allowlist_profile()}),
        _allowlist_profile(),
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_allowlist_requirement_unsatisfied"


def test_allowlist_requested_deny_only_provider_denied() -> None:
    backend = _HostedSecurityBackend(network_egress_deny_enforced=True)
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": _allowlist_profile()}),
        _allowlist_profile(),
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_allowlist_requirement_unsatisfied"


def test_local_sandbox_does_not_prove_allowlist(tmp_path: Path) -> None:
    sandbox = SandboxSession.create(
        tmp_path,
        tenant_id=TENANT,
        task_id=TASK,
        allowed_operations=_CODECRAFT_OPS,
    )
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="local",
        network_egress="allowlist",
        network_egress_allowlist=_REQUESTED.hosts,
    )
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_session=sandbox, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_allowlist_requirement_unsatisfied"


def test_physical_allowlist_qualification_blocked() -> None:
    """No current hosted provider attests physical exact host allowlist enforcement."""
    pytest.skip("REAL PROVIDER QUALIFICATION BLOCKED: no enforceable hosted substrate in repo")
