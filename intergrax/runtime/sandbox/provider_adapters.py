# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral sandbox capability adapters (P1.8)."""

from __future__ import annotations

from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.runtime.sandbox.contracts import (
    IsolationTierEvidence,
    SandboxExecCapable,
    SandboxSecurityCapabilities,
    SandboxSecurityCapable,
)
from intergrax.runtime.sandbox.execution_environment import (
    ExecutionEnvironmentProviderKind,
    ExecutionEnvironmentProviderRef,
    FilesystemAccess,
    NetworkAccess,
    ProcessExecution,
    SandboxProviderCapabilities,
)
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.sandbox_runtime import DEFAULT_SANDBOX_OPERATIONS
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry.wiring import ToolWiringContext


def _resolve_sandbox_session(ctx: ToolWiringContext) -> SandboxExecCapable | None:
    raw_session = ctx.sandbox_session or ctx.extras.get("sandbox_session")
    if raw_session is None:
        return None
    if isinstance(raw_session, (SandboxSession, SandboxExecCapable)):
        return raw_session
    return None


def _network_access_from_egress_proof(
    *,
    network_egress_deny_enforced: bool | None,
    network_egress_allowlist_enforced: bool | None,
) -> NetworkAccess:
    if network_egress_deny_enforced is True:
        return NetworkAccess.RESTRICTED
    if network_egress_allowlist_enforced is True:
        return NetworkAccess.RESTRICTED
    return NetworkAccess.NONE


def _supports_network_isolation_from_egress_proof(
    *,
    network_egress_deny_enforced: bool | None,
    network_egress_allowlist_enforced: bool | None,
) -> bool | None:
    if network_egress_deny_enforced is True or network_egress_allowlist_enforced is True:
        return True
    if network_egress_deny_enforced is False or network_egress_allowlist_enforced is False:
        return False
    return None


def _provider_kind_for_isolation_tier(
    isolation_tier: IsolationTierEvidence,
) -> ExecutionEnvironmentProviderKind | None:
    if isolation_tier == "local":
        return ExecutionEnvironmentProviderKind.LOCAL
    if isolation_tier in ("container", "cloud"):
        return ExecutionEnvironmentProviderKind.HOSTED
    return None


def project_provider_capabilities_from_security(
    security: SandboxSecurityCapabilities,
) -> SandboxProviderCapabilities | None:
    """Pure projection from trusted ``SandboxSecurityCapabilities`` — no invented positive facts."""
    provider_id = security.provider_id.strip()
    if not provider_id:
        return None
    provider_kind = _provider_kind_for_isolation_tier(security.isolation_tier)
    if provider_kind is None:
        return None
    if security.supports_sandboxed_exec is None:
        return None
    if security.supports_workspace_write is None:
        return None
    if security.filesystem_access is None:
        return None
    if security.process_execution is None:
        return None
    return SandboxProviderCapabilities(
        provider_ref=ExecutionEnvironmentProviderRef(
            provider_id=provider_id,
            provider_kind=provider_kind,
        ),
        filesystem_access=security.filesystem_access,
        network_access=_network_access_from_egress_proof(
            network_egress_deny_enforced=security.network_egress_deny_enforced,
            network_egress_allowlist_enforced=security.network_egress_allowlist_enforced,
        ),
        process_execution=security.process_execution,
        supports_sandboxed_exec=security.supports_sandboxed_exec,
        supports_workspace_write=security.supports_workspace_write,
        supports_network_isolation=_supports_network_isolation_from_egress_proof(
            network_egress_deny_enforced=security.network_egress_deny_enforced,
            network_egress_allowlist_enforced=security.network_egress_allowlist_enforced,
        ),
    )


def capabilities_from_security_attestation(
    security: SandboxSecurityCapabilities,
) -> SandboxProviderCapabilities | None:
    """Translate trusted ``SandboxSecurityCapabilities`` into provider capabilities."""
    return project_provider_capabilities_from_security(security)


def capabilities_from_local_session(session: SandboxSession) -> SandboxProviderCapabilities | None:
    return project_provider_capabilities_from_security(session.security_capabilities())


def capabilities_from_hosted_session(session: HostedSandboxSession) -> SandboxProviderCapabilities | None:
    return project_provider_capabilities_from_security(session.security_capabilities())


def capabilities_from_attested_exec_session(
    session: SandboxExecCapable,
) -> SandboxProviderCapabilities | None:
    """Capabilities only when ``SandboxSecurityCapable`` attests complete substrate evidence."""
    if not isinstance(session, SandboxSecurityCapable):
        return None
    if isinstance(session, SandboxSession):
        return capabilities_from_local_session(session)
    if isinstance(session, HostedSandboxSession):
        return capabilities_from_hosted_session(session)
    return project_provider_capabilities_from_security(session.security_capabilities())


def capabilities_from_host_backend(backend: SandboxHostBackend) -> SandboxProviderCapabilities | None:
    if not isinstance(backend, SandboxSecurityCapable):
        return None
    return project_provider_capabilities_from_security(backend.security_capabilities())


def probe_provider_capabilities_from_wiring(
    ctx: ToolWiringContext,
) -> tuple[SandboxProviderCapabilities, ...]:
    """Deterministic provider ordering: local session, configured host backend."""
    providers: list[SandboxProviderCapabilities] = []
    seen_ids: set[str] = set()

    session = _resolve_sandbox_session(ctx)
    if isinstance(session, SandboxSession):
        caps = capabilities_from_local_session(session)
        if caps is not None:
            if caps.provider_ref.provider_id in seen_ids:
                raise ValueError(f"duplicate provider_id: {caps.provider_ref.provider_id}")
            seen_ids.add(caps.provider_ref.provider_id)
            providers.append(caps)
    elif isinstance(session, HostedSandboxSession):
        caps = capabilities_from_hosted_session(session)
        if caps is not None:
            if caps.provider_ref.provider_id in seen_ids:
                raise ValueError(f"duplicate provider_id: {caps.provider_ref.provider_id}")
            seen_ids.add(caps.provider_ref.provider_id)
            providers.append(caps)
    elif isinstance(session, SandboxExecCapable):
        caps = capabilities_from_attested_exec_session(session)
        if caps is not None:
            if caps.provider_ref.provider_id in seen_ids:
                raise ValueError(f"duplicate provider_id: {caps.provider_ref.provider_id}")
            seen_ids.add(caps.provider_ref.provider_id)
            providers.append(caps)

    if ctx.sandbox_host is not None and session is None:
        caps = capabilities_from_host_backend(ctx.sandbox_host)
        if caps is not None:
            if caps.provider_ref.provider_id in seen_ids:
                raise ValueError(f"duplicate provider_id: {caps.provider_ref.provider_id}")
            seen_ids.add(caps.provider_ref.provider_id)
            providers.append(caps)

    return tuple(providers)


def select_provider_capabilities(
    providers: tuple[SandboxProviderCapabilities, ...],
) -> SandboxProviderCapabilities | None:
    """Stable first-match selection — local before hosted when both listed."""
    if not providers:
        return None
    return providers[0]
