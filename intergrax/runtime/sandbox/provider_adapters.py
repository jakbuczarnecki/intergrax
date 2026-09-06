# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral sandbox capability adapters (P1.8)."""

from __future__ import annotations

from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.runtime.sandbox.contracts import SandboxExecCapable, SandboxSecurityCapable
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


def _exec_supported(allowed_operations: frozenset[str]) -> bool:
    return "run_python" in allowed_operations or "run_script" in allowed_operations or "echo" in allowed_operations


def _network_access_for_operations(allowed_operations: frozenset[str]) -> NetworkAccess:
    if "browser_fetch" in allowed_operations:
        return NetworkAccess.RESTRICTED
    return NetworkAccess.NONE


def _network_access_from_egress_proof(network_egress_deny_enforced: bool | None) -> NetworkAccess:
    if network_egress_deny_enforced is True:
        return NetworkAccess.RESTRICTED
    return NetworkAccess.NONE


def capabilities_from_local_session(session: SandboxSession) -> SandboxProviderCapabilities:
    ops = session.allowed_operations
    security = session.security_capabilities()
    return SandboxProviderCapabilities(
        provider_ref=ExecutionEnvironmentProviderRef(
            provider_id=security.provider_id,
            provider_kind=ExecutionEnvironmentProviderKind.LOCAL,
        ),
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        network_access=_network_access_for_operations(ops),
        process_execution=ProcessExecution.SANDBOXED if _exec_supported(ops) else ProcessExecution.DENIED,
        supports_sandboxed_exec=_exec_supported(ops),
        supports_workspace_write="write_file" in ops,
        supports_network_isolation=security.network_egress_deny_enforced,
    )


def capabilities_from_hosted_session(session: HostedSandboxSession) -> SandboxProviderCapabilities:
    security = session.security_capabilities()
    ops = DEFAULT_SANDBOX_OPERATIONS
    return SandboxProviderCapabilities(
        provider_ref=ExecutionEnvironmentProviderRef(
            provider_id=security.provider_id,
            provider_kind=ExecutionEnvironmentProviderKind.HOSTED,
        ),
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        network_access=_network_access_from_egress_proof(security.network_egress_deny_enforced),
        process_execution=ProcessExecution.SANDBOXED,
        supports_sandboxed_exec=True,
        supports_workspace_write=True,
        supports_network_isolation=security.network_egress_deny_enforced,
    )


def capabilities_from_host_backend(backend: SandboxHostBackend) -> SandboxProviderCapabilities:
    if isinstance(backend, SandboxSecurityCapable):
        security = backend.security_capabilities()
        provider_id = security.provider_id
        network_isolation = security.network_egress_deny_enforced
    else:
        provider_id = f"hosted:{type(backend).__name__}"
        network_isolation = None
    return SandboxProviderCapabilities(
        provider_ref=ExecutionEnvironmentProviderRef(
            provider_id=provider_id,
            provider_kind=ExecutionEnvironmentProviderKind.HOSTED,
        ),
        filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
        network_access=_network_access_from_egress_proof(network_isolation),
        process_execution=ProcessExecution.SANDBOXED,
        supports_sandboxed_exec=True,
        supports_workspace_write=True,
        supports_network_isolation=network_isolation,
    )


def probe_provider_capabilities_from_wiring(
    ctx: ToolWiringContext,
) -> tuple[SandboxProviderCapabilities, ...]:
    """Deterministic provider ordering: local session, configured host backend."""
    providers: list[SandboxProviderCapabilities] = []
    seen_ids: set[str] = set()

    session = _resolve_sandbox_session(ctx)
    if isinstance(session, SandboxSession):
        caps = capabilities_from_local_session(session)
        if caps.provider_ref.provider_id in seen_ids:
            raise ValueError(f"duplicate provider_id: {caps.provider_ref.provider_id}")
        seen_ids.add(caps.provider_ref.provider_id)
        providers.append(caps)
    elif isinstance(session, HostedSandboxSession):
        caps = capabilities_from_hosted_session(session)
        if caps.provider_ref.provider_id in seen_ids:
            raise ValueError(f"duplicate provider_id: {caps.provider_ref.provider_id}")
        seen_ids.add(caps.provider_ref.provider_id)
        providers.append(caps)

    if ctx.sandbox_host is not None and session is None:
        caps = capabilities_from_host_backend(ctx.sandbox_host)
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
