# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool-owned realization for known catalog capability identities (UCA-2)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol

from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.tools.known_capability_realization import (
    KnownToolCapabilityRealizationOutcome,
    KnownToolCapabilityRealizationPort,
    KnownToolCapabilityRealizationRequest,
    KnownToolCapabilityRealizationResult,
)
from intergrax.tools.catalog import ToolPackageResolution
from intergrax.tools.dynamic_acquisition import (
    ToolHostActivationMaterializer,
    ToolHostActivationPort,
)
from intergrax.tools.errors import (
    DynamicToolAcquisitionResolutionError,
    KnownToolCapabilityRealizationConflictError,
)
from intergrax.tools.identity import ToolPackageIdentity
from intergrax.tools.known_capability_resolution import (
    assert_exact_tool_package_resolution_for_identity,
)


@dataclass(frozen=True, slots=True)
class KnownToolCapabilityRealizationOperationBinding:
    """Immutable operation semantics for Tool realization idempotency."""

    host_profile_id: str
    capability_identity: CapabilityIdentityKey

    @classmethod
    def from_request(
        cls,
        request: KnownToolCapabilityRealizationRequest,
    ) -> KnownToolCapabilityRealizationOperationBinding:
        return cls(
            host_profile_id=request.host_profile_id,
            capability_identity=request.capability_identity,
        )

    def assert_matches_request(
        self,
        request: KnownToolCapabilityRealizationRequest,
    ) -> None:
        if self.host_profile_id != request.host_profile_id:
            raise KnownToolCapabilityRealizationConflictError(
                "operation_id replay host_profile_id conflict",
            )
        if self.capability_identity.sort_key != request.capability_identity.sort_key:
            raise KnownToolCapabilityRealizationConflictError(
                "operation_id replay capability_identity conflict",
            )


class ToolPackageResolutionForIdentityPort(Protocol):
    """Domain-owned exact resolution for a known capability identity — not semantic search."""

    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution: ...


class ToolKnownCapabilityRealizationService(KnownToolCapabilityRealizationPort):
    """Activate a known tool capability on a host profile via Tool lifecycle ports."""

    def __init__(
        self,
        *,
        activation: ToolHostActivationPort,
        materializer: ToolHostActivationMaterializer,
        resolver: ToolPackageResolutionForIdentityPort,
    ) -> None:
        self._activation = activation
        self._materializer = materializer
        self._resolver = resolver
        self._bindings: dict[str, KnownToolCapabilityRealizationOperationBinding] = {}
        self._completed: dict[str, KnownToolCapabilityRealizationResult] = {}
        self._operation_locks: dict[str, threading.Lock] = {}
        self._operation_locks_guard = threading.Lock()

    @contextmanager
    def _synchronize_operation(self, operation_id: str) -> Iterator[None]:
        with self._operation_locks_guard:
            lock = self._operation_locks.get(operation_id)
            if lock is None:
                lock = threading.Lock()
                self._operation_locks[operation_id] = lock
        with lock:
            yield

    def realize(
        self,
        request: KnownToolCapabilityRealizationRequest,
    ) -> KnownToolCapabilityRealizationResult:
        with self._synchronize_operation(request.operation_id):
            return self._realize_locked(request)

    def _realize_locked(
        self,
        request: KnownToolCapabilityRealizationRequest,
    ) -> KnownToolCapabilityRealizationResult:
        prior = self._completed.get(request.operation_id)
        if prior is not None:
            self._binding_for(request.operation_id).assert_matches_request(request)
            return prior

        binding = self._bindings.get(request.operation_id)
        if binding is not None:
            binding.assert_matches_request(request)
        else:
            self._bindings[request.operation_id] = (
                KnownToolCapabilityRealizationOperationBinding.from_request(request)
            )

        if request.host_profile_id != self._activation.host_profile_id:
            return self._reject(
                request,
                outcome=KnownToolCapabilityRealizationOutcome.BLOCKED,
                reason_detail="host_profile_id mismatch",
            )

        try:
            resolution = self._resolver.resolve_for_identity(
                request.capability_identity
            )
        except (LookupError, ValueError) as exc:
            return self._reject(
                request,
                outcome=KnownToolCapabilityRealizationOutcome.FAILED,
                reason_detail=f"resolution failed: {exc}",
            )

        try:
            assert_exact_tool_package_resolution_for_identity(
                capability_identity=request.capability_identity,
                resolution=resolution,
            )
        except DynamicToolAcquisitionResolutionError as exc:
            return self._reject(
                request,
                outcome=KnownToolCapabilityRealizationOutcome.FAILED,
                reason_detail=str(exc),
            )

        if resolution.package_candidate.package_digest is None:
            return self._reject(
                request,
                outcome=KnownToolCapabilityRealizationOutcome.FAILED,
                reason_detail="resolved package lacks digest",
            )

        package_identity = ToolPackageIdentity.from_candidate(
            resolution.package_candidate,
        )
        ack = self._activation.activate(
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            resolved=resolution,
            materializer=self._materializer,
        )
        if ack.disposition is DomainLifecycleHandoffDisposition.DEFERRED:
            return self._reject(
                request,
                outcome=KnownToolCapabilityRealizationOutcome.REQUIRES_HITL,
                reason_detail=ack.reason_detail or "activation deferred",
            )
        if ack.disposition is not DomainLifecycleHandoffDisposition.ACCEPTED:
            return self._reject(
                request,
                outcome=KnownToolCapabilityRealizationOutcome.BLOCKED,
                reason_detail=ack.reason_detail or "activation rejected",
            )

        outcome = (
            KnownToolCapabilityRealizationOutcome.ALREADY_REALIZED
            if "idempotent" in (ack.reason_detail or "")
            else KnownToolCapabilityRealizationOutcome.REALIZED
        )
        evidence = CapabilityDiscoveryAvailabilityEvidence(
            host_available_keys=(request.capability_identity,),
        )
        result = KnownToolCapabilityRealizationResult(
            outcome=outcome,
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            capability_identity=request.capability_identity,
            domain_reference=ack.domain_reference
            or (
                f"tool:{package_identity.logical_tool_id}@"
                f"{package_identity.package_version}:{package_identity.package_digest}"
            ),
            availability_evidence=evidence,
            reason_detail=ack.reason_detail,
        )
        self._completed[request.operation_id] = result
        return result

    def _binding_for(
        self,
        operation_id: str,
    ) -> KnownToolCapabilityRealizationOperationBinding:
        binding = self._bindings.get(operation_id)
        if binding is None:
            raise KnownToolCapabilityRealizationConflictError(
                "operation_id completed without binding",
            )
        return binding

    def _reject(
        self,
        request: KnownToolCapabilityRealizationRequest,
        *,
        outcome: KnownToolCapabilityRealizationOutcome,
        reason_detail: str,
    ) -> KnownToolCapabilityRealizationResult:
        return KnownToolCapabilityRealizationResult(
            outcome=outcome,
            operation_id=request.operation_id,
            host_profile_id=request.host_profile_id,
            capability_identity=request.capability_identity,
            reason_detail=reason_detail,
        )


__all__ = [
    "ToolKnownCapabilityRealizationService",
    "ToolPackageResolutionForIdentityPort",
]
