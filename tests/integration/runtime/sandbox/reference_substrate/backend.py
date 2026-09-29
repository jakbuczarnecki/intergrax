# © Artur Czarnecki. All rights reserved.

"""Qualification-only ``SandboxHostBackend`` over Linux network namespaces."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from intergrax.integrations.contracts.sandbox_host import (
    SandboxArtifact,
    SandboxExecResult,
    SandboxSession,
)
from intergrax.runtime.sandbox.contracts import (
    SandboxSecurityCapabilities,
    SandboxSecurityRequirements,
)
from intergrax.runtime.sandbox.execution_environment import FilesystemAccess, ProcessExecution
from intergrax.runtime.sandbox.network_egress import NetworkEgressAllowlist

from .constants import REFERENCE_PROVIDER_ID
from .endpoints import ReferenceEndpointServers
from .errors import ReferenceSubstrateLifecycleError
from .firewall import (
    ReferenceSubstratePolicyError,
    apply_egress_policy_netns,
    read_verified_egress_policy,
)
from .preflight import ReferenceSubstratePreflight, evaluate_reference_substrate_preflight
from .topology import NetnsSessionResources, create_netns_session, destroy_netns_session


class ReferenceSubstrateSecurityError(RuntimeError):
    """Security admission or attestation failure for reference substrate."""


@dataclass(slots=True)
class _ReferenceSessionState:
    resources: NetnsSessionResources
    security: SandboxSecurityCapabilities | None = None


class ReferenceSandboxBackend:
    """Linux netns backend — qualification harness only, not a production provider."""

    def __init__(
        self,
        *,
        preflight: ReferenceSubstratePreflight | None = None,
        endpoint_servers: ReferenceEndpointServers | None = None,
    ) -> None:
        self._preflight = preflight or evaluate_reference_substrate_preflight()
        if not self._preflight.ok:
            raise ReferenceSubstrateSecurityError(self._preflight.block_reason or "preflight blocked")
        self._use_nftables = self._preflight.firewall_backend == "nftables"
        self._endpoints = endpoint_servers or ReferenceEndpointServers()
        self._endpoints.start()
        self._sessions: dict[str, _ReferenceSessionState] = {}
        self._lifecycle_closed = False

    @property
    def preflight(self) -> ReferenceSubstratePreflight:
        return self._preflight

    def create_session(self) -> SandboxSession:
        resources = create_netns_session()
        session_id = resources.netns
        self._sessions[session_id] = _ReferenceSessionState(resources=resources)
        return SandboxSession(session_id=session_id, status="running")

    def create_session_with_security(
        self,
        requirements: SandboxSecurityRequirements,
    ) -> SandboxSession:
        if requirements.isolation_tier != "cloud":
            raise ReferenceSubstrateSecurityError(
                f"reference substrate expects cloud isolation tier for qualification parity; "
                f"got {requirements.isolation_tier}",
            )
        if requirements.network_egress != "allowlist":
            raise ReferenceSubstrateSecurityError(
                f"reference substrate supports allowlist egress only; got {requirements.network_egress}",
            )
        allowlist = requirements.network_egress_allowlist
        if allowlist is None or not allowlist.hosts:
            raise ReferenceSubstrateSecurityError(
                "empty or missing allowlist cannot admit unrestricted network egress",
            )
        resources = create_netns_session()
        session_id = resources.netns
        try:
            apply_egress_policy_netns(
                session_id,
                allowlist=allowlist,
                use_nftables=self._use_nftables,
            )
            verified = read_verified_egress_policy(session_id, use_nftables=self._use_nftables)
        except ReferenceSubstratePolicyError as exc:
            destroy_netns_session(resources)
            raise ReferenceSubstrateSecurityError(str(exc)) from exc
        capabilities = self._capabilities_from_verified(verified.enforced_hosts)
        self._sessions[session_id] = _ReferenceSessionState(
            resources=resources,
            security=capabilities,
        )
        return SandboxSession(session_id=session_id, status="running")

    def session_security_capabilities(self, session_id: str) -> SandboxSecurityCapabilities:
        state = self._sessions.get(session_id)
        if state is None:
            raise ReferenceSubstrateSecurityError(f"unknown reference session: {session_id}")
        try:
            verified = read_verified_egress_policy(session_id, use_nftables=self._use_nftables)
        except ReferenceSubstratePolicyError as exc:
            raise ReferenceSubstrateSecurityError(str(exc)) from exc
        return self._capabilities_from_verified(verified.enforced_hosts)

    def destroy_session(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        destroy_netns_session(state.resources)
        self._sessions.pop(session_id, None)

    def close(self) -> None:
        if self._lifecycle_closed:
            return
        cleanup_errors: list[str] = []
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            state = self._sessions.get(session_id)
            if state is None:
                continue
            try:
                destroy_netns_session(state.resources)
            except ReferenceSubstratePolicyError as exc:
                cleanup_errors.append(f"session {session_id}: {exc}")
                continue
            self._sessions.pop(session_id, None)
        try:
            self._endpoints.stop()
        except ReferenceSubstrateLifecycleError as exc:
            cleanup_errors.append(str(exc))
        if cleanup_errors:
            raise ReferenceSubstrateLifecycleError(
                "reference backend qualification cleanup incomplete: "
                + "; ".join(cleanup_errors),
            )
        self._lifecycle_closed = True

    def exec(self, session_id: str, command: str) -> SandboxExecResult:
        if session_id not in self._sessions:
            return SandboxExecResult(exit_code=1, stderr="unknown session", stdout="")
        completed = subprocess.run(
            ["ip", "netns", "exec", session_id, "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=45.0,
            check=False,
        )
        return SandboxExecResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        raise ReferenceSubstrateSecurityError("artifact upload not supported in reference substrate")

    def _capabilities_from_verified(
        self,
        enforced: NetworkEgressAllowlist,
    ) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier="cloud",
            provider_id=REFERENCE_PROVIDER_ID,
            network_egress_deny_enforced=False,
            network_egress_allowlist_enforced=True,
            enforced_network_hosts=enforced,
            supports_sandboxed_exec=True,
            supports_workspace_write=False,
            filesystem_access=FilesystemAccess.READ_ONLY,
            process_execution=ProcessExecution.SANDBOXED,
        )


def build_reference_backend_or_skip_reason() -> ReferenceSandboxBackend | str:
    preflight = evaluate_reference_substrate_preflight()
    if not preflight.ok:
        return preflight.skip_reason()
    return ReferenceSandboxBackend(preflight=preflight)
