# © Artur Czarnecki. All rights reserved.

"""Structural views for sandbox isolation authority without application package coupling."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.sandbox_profile import SandboxProfile


class ProfileSandboxIsolationSource(Protocol):
    """Environment profile fragment consulted by runtime sandbox resolver."""

    sandbox: SandboxProfile | None


class EffectiveProfileRevisionIsolationView(Protocol):
    """Pinned effective profile revision carrying an isolation-capable profile."""

    effective_profile: ProfileSandboxIsolationSource
