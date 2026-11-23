# © Artur Czarnecki. All rights reserved.

"""Shared posture predicate for production-wiring guards (T-IGX-043-B, DP-24)."""

from __future__ import annotations

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode


def wiring_fails_closed(env: ApplicationEnvironmentProfile) -> bool:
    """True when the composition must refuse in-memory or no-op stand-ins.

    Union of the two posture signals (DP-24): a ``PRODUCT`` application profile or a
    ``STRICT`` execution mode. Either signal alone is enough, because harness production
    hosts are built as ``LAB`` + ``STRICT``.
    """
    return env.application_profile is ApplicationProfile.PRODUCT or env.execution_mode is ExecutionMode.STRICT
