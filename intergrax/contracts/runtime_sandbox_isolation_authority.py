# © Artur Czarnecki. All rights reserved.

"""Explicit runtime-host sandbox isolation authority (composition-owned, not session-derived)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_environment_isolation import ProfileSandboxIsolationSource
from intergrax.contracts.sandbox_profile import SandboxProfile


@dataclass(frozen=True, slots=True)
class RuntimeSandboxIsolationAuthority:
    """
    Narrow sandbox isolation authority for UAEP/Nexus runtime hosts without Tier-3 profile.

    Implements :class:`~intergrax.contracts.execution_environment_isolation.ProfileSandboxIsolationSource`.
    Must be supplied explicitly by composition — never inferred from ``SandboxExecCapable`` availability.
    """

    sandbox: SandboxProfile

    @classmethod
    def local_sandbox_exec_enabled(cls) -> RuntimeSandboxIsolationAuthority:
        """Explicit composition preset: local sandbox manager may run sandbox.exec."""
        return cls(sandbox=SandboxProfile(enable_exec_tool=True))
