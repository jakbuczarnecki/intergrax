# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability acquisition policy port for AW-6A recovery decisions."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.autonomous_work.profile_reference import CapabilityProfileRef


class CapabilityAcquisitionPolicy(Protocol):
    """Resolve whether capability acquisition is permitted for a worker profile."""

    def is_acquisition_allowed(
        self,
        profile_ref: CapabilityProfileRef,
    ) -> bool:
        """Return True when the capability profile permits acquisition attempts."""
        ...


class StaticCapabilityAcquisitionPolicy:
    """Deterministic capability acquisition policy for tests and wiring."""

    def __init__(self, *, allowed: bool) -> None:
        self._allowed = allowed

    def is_acquisition_allowed(
        self,
        profile_ref: CapabilityProfileRef,
    ) -> bool:
        return self._allowed
