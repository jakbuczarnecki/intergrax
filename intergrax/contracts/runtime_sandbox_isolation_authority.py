# © Artur Czarnecki. All rights reserved.

"""Explicit runtime-host sandbox isolation authority (composition-owned, not session-derived)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from intergrax.contracts.sandbox_profile import SandboxProfile

RUNTIME_SANDBOX_ISOLATION_AUTHORITY_EXTRA_KEY = "runtime_sandbox_isolation_authority"


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


def extras_contain_sandbox_isolation_authority(extras: Mapping[str, object]) -> bool:
    """True when extras already carry pinned or legacy profile authority."""
    revision_raw = extras.get("effective_profile_revision")
    if revision_raw is not None and hasattr(revision_raw, "effective_profile"):
        profile = getattr(revision_raw, "effective_profile", None)
        if profile is not None and hasattr(profile, "sandbox"):
            return True
    legacy = extras.get("effective_environment_profile")
    if legacy is not None and hasattr(legacy, "sandbox"):
        return True
    authority = extras.get(RUNTIME_SANDBOX_ISOLATION_AUTHORITY_EXTRA_KEY)
    return isinstance(authority, RuntimeSandboxIsolationAuthority)


def apply_runtime_sandbox_isolation_authority_extras(
    extras: Mapping[str, object],
    authority: RuntimeSandboxIsolationAuthority,
) -> dict[str, object]:
    """Return a copy of ``extras`` with explicit runtime sandbox authority (no override of existing)."""
    merged = dict(extras)
    if extras_contain_sandbox_isolation_authority(merged):
        return merged
    merged[RUNTIME_SANDBOX_ISOLATION_AUTHORITY_EXTRA_KEY] = authority
    return merged
