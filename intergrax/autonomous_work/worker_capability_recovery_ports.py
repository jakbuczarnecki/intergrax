# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ports for canonical discovery/UCA/qualification — AW core depends on abstractions only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityNeed,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletion,
)
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)


@dataclass(frozen=True, slots=True)
class CanonicalCapabilityDiscoveryRequest:
    """Discovery inputs — worker operations inform catalog query, not CapabilityNeed."""

    capability_need: CapabilityNeed
    worker_need: WorkerCapabilityNeed
    discovery_correlation_id: str
    requested_at: datetime


@runtime_checkable
class CanonicalCapabilityDiscoveryPort(Protocol):
    """Canonical capability discovery — produces authoritative DiscoveryCompletion."""

    def complete_discovery(
        self,
        request: CanonicalCapabilityDiscoveryRequest,
    ) -> DiscoveryCompletion: ...


@runtime_checkable
class CapabilityAcquisitionCoordinatorPort(Protocol):
    """Canonical UCA acquisition dispatch — strategy selection opaque to AW."""

    def acquire(
        self,
        request: CapabilityAcquisitionRequest,
    ) -> CapabilityAcquisitionResult: ...


@runtime_checkable
class CapabilityQualificationCoordinatorPort(Protocol):
    """Canonical UCA-4 qualification dispatch."""

    def qualify(
        self,
        request: CapabilityQualificationRequest,
    ) -> CapabilityQualificationResult: ...


__all__ = [
    "CanonicalCapabilityDiscoveryPort",
    "CanonicalCapabilityDiscoveryRequest",
    "CapabilityAcquisitionCoordinatorPort",
    "CapabilityQualificationCoordinatorPort",
]
