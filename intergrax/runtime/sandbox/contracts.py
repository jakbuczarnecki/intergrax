# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox execution contracts shared by local and hosted sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from intergrax.runtime.sandbox.models import SandboxExecutionResult

IsolationTierEvidence = Literal["local", "container", "cloud"]


@dataclass(frozen=True, slots=True)
class SandboxSecurityCapabilities:
    """Trusted substrate security capability evidence (Sandbox-owned contract)."""

    isolation_tier: IsolationTierEvidence
    provider_id: str
    network_egress_deny_enforced: bool | None = None
    """``True`` when egress deny is proven; ``False`` when proven absent; ``None`` when unknown."""


@runtime_checkable
class SandboxSecurityCapable(Protocol):
    """Sessions and optional host backends that attest security capabilities."""

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        """Return trusted substrate security capability evidence."""
        ...


@runtime_checkable
class SandboxExecCapable(Protocol):
    """Minimal surface required by the ``sandbox.exec`` catalog tool."""

    session_id: str

    def execute(self, operation: str, payload: dict | None = None) -> SandboxExecutionResult:
        """Run an allowlisted sandbox operation."""
        ...
