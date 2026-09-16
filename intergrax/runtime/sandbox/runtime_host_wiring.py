# © Artur Czarnecki. All rights reserved.

"""Minimal sandbox authority projection for UAEP/Nexus runtime hosts without Tier-3 profile."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.sandbox_profile import SandboxProfile


@dataclass(frozen=True, slots=True)
class RuntimeHostSandboxIsolationProfile:
    """Narrow profile authority for local ``SandboxSessionManager`` sandbox tasks."""

    sandbox: SandboxProfile

    @classmethod
    def for_attached_session(cls) -> RuntimeHostSandboxIsolationProfile:
        return cls(sandbox=SandboxProfile(enable_exec_tool=True))


def runtime_host_sandbox_isolation_profile() -> RuntimeHostSandboxIsolationProfile:
    return RuntimeHostSandboxIsolationProfile.for_attached_session()
