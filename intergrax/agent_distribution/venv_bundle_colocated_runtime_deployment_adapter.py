# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production co-located VENV_BUNDLE ``RuntimeDeploymentAdapter`` (ADR-01 A1)."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Sequence

from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    DrainPolicy,
    DrainStatus,
    PreparedDeployment,
    ServingUnitAttestation,
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
)
from intergrax.agent_distribution.activation_domain_coordinator import (
    CoordinatorVerifiedUnitServingAdmissionOpen,
    PhysicalExecutionReservation,
    ServingExecutionEligibilityEvidence,
)
from intergrax.agent_distribution.colocated_venv_unit_runtime_binding import (
    ColocatedVenvUnitRuntimeBinding,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_resource import (
    ColocatedVenvRuntimeResource,
)
from intergrax.agent_distribution.venv_bundle_deployment_artifact import (
    VenvBundleDeploymentArtifactError,
    discover_venv_bundle_site_packages,
    load_venv_bundle_runtime_lock,
    resolve_venv_bundle_artifact_directory,
    verify_venv_bundle_artifact_bytes,
    verify_venv_bundle_lock_matches_revision,
)

_UNIT_REF_PREFIX: Final = "venv-colocated:"


class VenvBundleColocatedDeploymentError(AgentDistributionError):
    """Co-located VENV deployment adapter operation failed."""


@dataclass(frozen=True, slots=True)
class VenvBundleCoordinatorVerifiedUnitServingAdmissionOpen:
    """Adapter-owned unit-bound admission open port for coordinator invocation."""

    _adapter: VenvBundleColocatedRuntimeDeploymentAdapter
    _bound_serving_unit_ref: str

    def apply_coordinator_verified_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        if evidence.serving_unit_ref != self._bound_serving_unit_ref:
            raise VenvBundleColocatedDeploymentError(
                "coordinator-verified admission open serving unit mismatch"
            )
        self._adapter._apply_coordinator_verified_unit_admission_open(evidence)

    def publish_coordinator_verified_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        if evidence.serving_unit_ref != self._bound_serving_unit_ref:
            raise VenvBundleColocatedDeploymentError(
                "coordinator-verified admission publish serving unit mismatch"
            )
        self._adapter._publish_coordinator_verified_unit_admission_open(evidence)

    def compensate_failed_coordination_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        if evidence.serving_unit_ref != self._bound_serving_unit_ref:
            raise VenvBundleColocatedDeploymentError(
                "coordinator-verified admission compensation serving unit mismatch"
            )
        self._adapter._compensate_failed_coordinator_verified_unit_admission_open(
            evidence
        )

    def reconcile_coordinator_verified_existing_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        if evidence.serving_unit_ref != self._bound_serving_unit_ref:
            raise VenvBundleColocatedDeploymentError(
                "coordinator-verified admission reopen serving unit mismatch"
            )
        self._adapter._reconcile_coordinator_verified_unit_admission_open(evidence)


class ColocatedUnitExecutionReservation:
    """One physical in-flight execution slot; release at most once."""

    __slots__ = ("_adapter", "_active", "_serving_unit_ref")

    def __init__(
        self,
        adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
        serving_unit_ref: str,
    ) -> None:
        self._adapter = adapter
        self._serving_unit_ref = serving_unit_ref
        self._active = True

    def release(self) -> None:
        if not self._active:
            raise VenvBundleColocatedDeploymentError(
                "execution release without matching admission"
            )
        self._active = False
        self._adapter.release_unit_execution(self._serving_unit_ref)


class _UnitAdmissionGate:
    """Adapter-owned execution admission boundary for one physical unit (A1)."""

    __slots__ = ("_admission_closed", "_in_flight", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._admission_closed = True
        self._in_flight = 0

    def admit_execution(self) -> None:
        with self._lock:
            if self._admission_closed:
                raise VenvBundleColocatedDeploymentError(
                    "serving unit is not admitting new execution work"
                )
            self._in_flight += 1

    def release_execution(self) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise VenvBundleColocatedDeploymentError(
                    "execution release without matching admission"
                )
            self._in_flight -= 1

    def begin_drain(self) -> None:
        with self._lock:
            self._admission_closed = True

    def open_admission(self) -> None:
        with self._lock:
            self._admission_closed = False

    def in_flight_count(self) -> int:
        with self._lock:
            return self._in_flight

    def admission_closed(self) -> bool:
        with self._lock:
            return self._admission_closed


@dataclass
class _ColocatedVenvPhysicalUnit:
    serving_unit_ref: str
    application_id: str
    application_environment_id: str
    runtime_revision_id: str
    materialization_artifact_digest: str
    artifact_locator: str
    artifact_root: Path
    site_packages: Path
    runtime_lock: MaterializedRuntimeLock
    lifecycle_state: DeploymentInstanceState
    runtime_resource: ColocatedVenvRuntimeResource
    admission: _UnitAdmissionGate = field(default_factory=_UnitAdmissionGate)
    readiness_evidence_ref: str | None = None
    serving_admission_authorized: bool = False
    coordinator_admission_open_staged: ServingExecutionEligibilityEvidence | None = None
    coordinator_verified_published_pointer_revision: int | None = None
    drain_started_monotonic: float | None = None
    binding_key: str = ""

    def invalidate_resources(self) -> None:
        self.runtime_resource.invalidate()
        self.serving_admission_authorized = False
        self.coordinator_admission_open_staged = None
        self.coordinator_verified_published_pointer_revision = None
        self.admission.begin_drain()


class VenvBundleColocatedRuntimeDeploymentAdapter:
    """In-process VENV_BUNDLE serving unit with real artifact binding and admission gate."""

    def __init__(
        self,
        *,
        trusted_artifact_parent_roots: Sequence[Path] | None = None,
        allow_qualification_locators: bool = False,
    ) -> None:
        self._trusted_roots = (
            tuple(root.resolve() for root in trusted_artifact_parent_roots)
            if trusted_artifact_parent_roots is not None
            else None
        )
        self._allow_qualification_locators = allow_qualification_locators
        self._lock = threading.RLock()
        self._units: dict[str, _ColocatedVenvPhysicalUnit] = {}
        self._binding_index: dict[str, str] = {}

    def coordinator_verified_unit_serving_admission_open_port(
        self, serving_unit_ref: str
    ) -> CoordinatorVerifiedUnitServingAdmissionOpen:
        """Return a unit-bound admission open port for sanctioned coordinator wiring."""
        self._require_unit(serving_unit_ref)
        return VenvBundleCoordinatorVerifiedUnitServingAdmissionOpen(
            _adapter=self,
            _bound_serving_unit_ref=serving_unit_ref,
        )

    def unit_serving_execution_admission_open_for_lifecycle_reopen(
        self,
        serving_unit_ref: str,
    ) -> bool:
        """Report whether physical admission matches coordinator lifecycle reopen preconditions."""
        with self._lock:
            unit = self._units.get(serving_unit_ref)
            if unit is None:
                return False
            if unit.coordinator_admission_open_staged is not None:
                return False
            if unit.lifecycle_state is not DeploymentInstanceState.SERVING:
                return False
            if not unit.serving_admission_authorized:
                return False
            if unit.admission.admission_closed():
                return False
            if not unit.runtime_resource.is_live:
                return False
            return unit.coordinator_verified_published_pointer_revision is not None

    def reserve_unit_execution(
        self, serving_unit_ref: str
    ) -> ColocatedUnitExecutionReservation:
        """Reserve one in-flight execution slot after coordinator authorization."""
        unit = self._require_live_unit(serving_unit_ref)
        if not unit.serving_admission_authorized:
            raise VenvBundleColocatedDeploymentError(
                "serving unit is not admitting new execution work"
            )
        unit.admission.admit_execution()
        return ColocatedUnitExecutionReservation(self, serving_unit_ref)

    def release_unit_execution(self, serving_unit_ref: str) -> None:
        unit = self._require_unit(serving_unit_ref)
        unit.admission.release_execution()

    def authorize_serving_execution_admission(
        self, evidence: ServingExecutionEligibilityEvidence
    ) -> None:
        raise VenvBundleColocatedDeploymentError(
            "serving execution admission requires activation domain coordinator"
        )

    def _evidence_matches_unit_identity(
        self,
        unit: _ColocatedVenvPhysicalUnit,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        if unit.application_id != evidence.application_id:
            raise VenvBundleColocatedDeploymentError(
                "serving execution admission evidence application mismatch"
            )
        if unit.application_environment_id != evidence.application_environment_id:
            raise VenvBundleColocatedDeploymentError(
                "serving execution admission evidence environment mismatch"
            )
        if unit.runtime_revision_id != evidence.runtime_revision_id:
            raise VenvBundleColocatedDeploymentError(
                "serving execution admission evidence revision mismatch"
            )
        if unit.materialization_artifact_digest != evidence.materialization_artifact_digest:
            raise VenvBundleColocatedDeploymentError(
                "serving execution admission evidence digest mismatch"
            )

    def _apply_coordinator_verified_unit_admission_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Stage coordinator-verified admission; gate stays closed until publish."""
        with self._lock:
            unit = self._require_live_unit(evidence.serving_unit_ref)
            self._evidence_matches_unit_identity(unit, evidence)
            if unit.serving_admission_authorized:
                raise VenvBundleColocatedDeploymentError(
                    "serving execution admission already authorized for unit"
                )
            if unit.coordinator_admission_open_staged is not None:
                raise VenvBundleColocatedDeploymentError(
                    "coordinator-verified admission open already staged for unit"
                )
            if unit.lifecycle_state not in {
                DeploymentInstanceState.READY,
                DeploymentInstanceState.SERVING,
            }:
                raise VenvBundleColocatedDeploymentError(
                    "serving unit is not eligible for execution admission"
                )
            if not unit.runtime_resource.is_live:
                raise VenvBundleColocatedDeploymentError(
                    "serving unit physical runtime is not initialized"
                )
            unit.coordinator_admission_open_staged = evidence

    def _publish_coordinator_verified_unit_admission_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Publish staged admission after canonical coordination succeeded."""
        with self._lock:
            unit = self._require_live_unit(evidence.serving_unit_ref)
            self._evidence_matches_unit_identity(unit, evidence)
            staged = unit.coordinator_admission_open_staged
            if staged is None:
                raise VenvBundleColocatedDeploymentError(
                    "coordinator-verified admission open is not staged for unit"
                )
            if staged != evidence:
                raise VenvBundleColocatedDeploymentError(
                    "coordinator-verified admission publish evidence mismatch"
                )
            if unit.serving_admission_authorized:
                raise VenvBundleColocatedDeploymentError(
                    "serving execution admission already authorized for unit"
                )
            prior_lifecycle = unit.lifecycle_state
            try:
                unit.serving_admission_authorized = True
                unit.admission.open_admission()
                unit.lifecycle_state = DeploymentInstanceState.SERVING
                unit.coordinator_admission_open_staged = None
                unit.coordinator_verified_published_pointer_revision = (
                    evidence.serving_pointer_revision
                )
            except Exception:
                unit.serving_admission_authorized = False
                unit.coordinator_admission_open_staged = None
                unit.coordinator_verified_published_pointer_revision = None
                unit.admission.begin_drain()
                unit.lifecycle_state = prior_lifecycle
                raise

    def _reconcile_coordinator_verified_unit_admission_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        """Confirm live physical admission matches coordinator evidence (no new authority)."""
        with self._lock:
            unit = self._require_live_unit(evidence.serving_unit_ref)
            self._evidence_matches_unit_identity(unit, evidence)
            if unit.coordinator_admission_open_staged is not None:
                raise VenvBundleColocatedDeploymentError(
                    "coordinator-verified admission open is staged during lifecycle reopen"
                )
            if unit.lifecycle_state is not DeploymentInstanceState.SERVING:
                raise VenvBundleColocatedDeploymentError(
                    "serving unit is not in serving lifecycle state for admission reopen"
                )
            if not unit.serving_admission_authorized:
                raise VenvBundleColocatedDeploymentError(
                    "serving execution admission is not authorized for lifecycle reopen"
                )
            if unit.admission.admission_closed():
                raise VenvBundleColocatedDeploymentError(
                    "serving unit admission gate is closed for lifecycle reopen"
                )
            if not unit.runtime_resource.is_live:
                raise VenvBundleColocatedDeploymentError(
                    "serving unit physical runtime is not initialized for lifecycle reopen"
                )
            published_pointer = unit.coordinator_verified_published_pointer_revision
            if published_pointer is None:
                raise VenvBundleColocatedDeploymentError(
                    "serving unit lacks coordinator-verified admission pointer evidence"
                )
            if published_pointer != evidence.serving_pointer_revision:
                raise VenvBundleColocatedDeploymentError(
                    "serving pointer revision mismatch for lifecycle admission reopen"
                )

    def _compensate_failed_coordinator_verified_unit_admission_open(
        self,
        evidence: ServingExecutionEligibilityEvidence,
    ) -> None:
        with self._lock:
            unit = self._require_unit(evidence.serving_unit_ref)
            self._evidence_matches_unit_identity(unit, evidence)
            staged = unit.coordinator_admission_open_staged
            if staged is not None:
                if staged != evidence:
                    raise VenvBundleColocatedDeploymentError(
                        "admission compensation staged evidence mismatch"
                    )
                unit.coordinator_admission_open_staged = None
                return
            if not unit.serving_admission_authorized:
                return
            if unit.admission.in_flight_count() > 0:
                raise VenvBundleColocatedDeploymentError(
                    "cannot compensate unit admission open while executions are in flight"
                )
            unit.serving_admission_authorized = False
            unit.coordinator_verified_published_pointer_revision = None
            unit.admission.begin_drain()
            if unit.lifecycle_state is DeploymentInstanceState.SERVING:
                unit.lifecycle_state = DeploymentInstanceState.READY

    def artifact_root_for_serving_unit(self, serving_unit_ref: str) -> Path:
        """Expose immutable artifact root for composition-side resolver binding."""
        return self._require_unit(serving_unit_ref).artifact_root

    def colocated_unit_runtime_binding(
        self,
        serving_unit_ref: str,
    ) -> ColocatedVenvUnitRuntimeBinding:
        """Return typed physical runtime binding for composition-side factory resolution."""
        unit = self._require_live_unit(serving_unit_ref)
        warmup = unit.runtime_resource.warmup_evidence
        if warmup is None:
            raise VenvBundleColocatedDeploymentError(
                "serving unit physical runtime is not initialized"
            )
        return ColocatedVenvUnitRuntimeBinding(
            serving_unit_ref=unit.serving_unit_ref,
            application_id=unit.application_id,
            application_environment_id=unit.application_environment_id,
            runtime_revision_id=unit.runtime_revision_id,
            materialization_artifact_digest=unit.materialization_artifact_digest,
            artifact_root=unit.artifact_root,
            site_packages=unit.site_packages,
            physical_runtime_warmup_evidence=warmup,
        )

    def _binding_key(self, revision: RuntimeRevision) -> str:
        digest = revision.materialization_artifact_digest or ""
        return (
            f"{revision.application_id}:{revision.application_environment_id}:"
            f"{revision.runtime_revision_id}:{digest}"
        )

    def _validate_revision(self, revision: RuntimeRevision) -> str:
        if revision.materialization_topology is not MaterializationTopology.VENV_BUNDLE:
            raise VenvBundleColocatedDeploymentError(
                "deployment adapter supports VENV_BUNDLE topology only"
            )
        digest = revision.materialization_artifact_digest
        if digest is None:
            raise VenvBundleColocatedDeploymentError(
                "runtime revision requires materialization_artifact_digest"
            )
        return digest

    def _validate_materialization_output(
        self,
        revision: RuntimeRevision,
        materialization_output: MaterializationOutput | None,
    ) -> None:
        if materialization_output is None:
            return
        if materialization_output.topology is not MaterializationTopology.VENV_BUNDLE:
            raise VenvBundleColocatedDeploymentError(
                "materialization output topology must be VENV_BUNDLE"
            )
        digest = revision.materialization_artifact_digest
        if digest is None:
            raise VenvBundleColocatedDeploymentError(
                "runtime revision requires materialization_artifact_digest"
            )
        if materialization_output.materialization_artifact_digest != digest:
            raise VenvBundleColocatedDeploymentError(
                "materialization output artifact digest mismatch with revision"
            )

    def _inspect_artifact(
        self,
        revision: RuntimeRevision,
        artifact_locator: str,
    ) -> tuple[Path, MaterializedRuntimeLock, Path]:
        digest = self._validate_revision(revision)
        if artifact_locator.strip() != artifact_locator:
            raise VenvBundleColocatedDeploymentError("artifact_locator must be normalized")
        artifact_root = resolve_venv_bundle_artifact_directory(
            artifact_locator,
            trusted_parent_roots=self._trusted_roots,
            allow_qualification_locators=self._allow_qualification_locators,
        )
        verify_venv_bundle_artifact_bytes(artifact_root, digest)
        runtime_lock = load_venv_bundle_runtime_lock(artifact_root)
        verify_venv_bundle_lock_matches_revision(revision, runtime_lock)
        site_packages = discover_venv_bundle_site_packages(artifact_root)
        return artifact_root, runtime_lock, site_packages

    def _remove_unit(self, unit_ref: str, unit: _ColocatedVenvPhysicalUnit) -> None:
        unit.invalidate_resources()
        self._units.pop(unit_ref, None)
        if unit.binding_key:
            self._binding_index.pop(unit.binding_key, None)

    def prepare(
        self,
        revision: RuntimeRevision,
        *,
        artifact_locator: str,
        materialization_output: MaterializationOutput | None = None,
    ) -> PreparedDeployment:
        digest = self._validate_revision(revision)
        self._validate_materialization_output(revision, materialization_output)
        binding = self._binding_key(revision)
        with self._lock:
            existing_ref = self._binding_index.get(binding)
            if existing_ref is not None:
                existing = self._units.get(existing_ref)
                if existing is not None and existing.lifecycle_state not in {
                    DeploymentInstanceState.STOPPED,
                    DeploymentInstanceState.FAILED,
                }:
                    if (
                        existing.artifact_locator == artifact_locator
                        and existing.materialization_artifact_digest == digest
                    ):
                        return PreparedDeployment(
                            serving_unit_ref=existing_ref,
                            materialization_artifact_digest=digest,
                        )

            try:
                artifact_root, runtime_lock, site_packages = self._inspect_artifact(
                    revision, artifact_locator
                )
            except VenvBundleDeploymentArtifactError as exc:
                raise VenvBundleColocatedDeploymentError(str(exc)) from exc

            unit_ref = f"{_UNIT_REF_PREFIX}{secrets.token_hex(12)}"
            runtime_resource = ColocatedVenvRuntimeResource(
                site_packages=site_packages,
                artifact_digest=digest,
            )
            unit = _ColocatedVenvPhysicalUnit(
                serving_unit_ref=unit_ref,
                application_id=revision.application_id,
                application_environment_id=revision.application_environment_id,
                runtime_revision_id=revision.runtime_revision_id,
                materialization_artifact_digest=digest,
                artifact_locator=artifact_locator,
                artifact_root=artifact_root,
                site_packages=site_packages,
                runtime_lock=runtime_lock,
                lifecycle_state=DeploymentInstanceState.PREPARING,
                runtime_resource=runtime_resource,
                binding_key=binding,
            )
            self._units[unit_ref] = unit
            self._binding_index[binding] = unit_ref

        return PreparedDeployment(
            serving_unit_ref=unit_ref,
            materialization_artifact_digest=digest,
        )

    def check_readiness(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        unit = self._require_unit(serving_unit_ref)
        if unit.lifecycle_state is DeploymentInstanceState.STOPPED:
            raise VenvBundleColocatedDeploymentError(
                "cannot mark stopped serving unit ready"
            )
        try:
            verify_venv_bundle_artifact_bytes(
                unit.artifact_root, unit.materialization_artifact_digest
            )
            lock = load_venv_bundle_runtime_lock(unit.artifact_root)
            verify_venv_bundle_lock_matches_revision(revision, lock)
            site_packages = discover_venv_bundle_site_packages(unit.artifact_root)
        except VenvBundleDeploymentArtifactError as exc:
            self._fail_unit(serving_unit_ref, unit)
            raise VenvBundleColocatedDeploymentError(str(exc)) from exc

        if site_packages != unit.site_packages:
            self._fail_unit(serving_unit_ref, unit)
            raise VenvBundleColocatedDeploymentError(
                "site-packages binding changed since unit allocation"
            )
        if revision.runtime_revision_id != unit.runtime_revision_id:
            self._fail_unit(serving_unit_ref, unit)
            raise VenvBundleColocatedDeploymentError("revision identity mismatch")

        try:
            warmup = unit.runtime_resource.initialize()
        except RuntimeError as exc:
            self._fail_unit(serving_unit_ref, unit)
            raise VenvBundleColocatedDeploymentError(str(exc)) from exc

        evidence = (
            f"venv-colocated-readiness:{unit.serving_unit_ref}:"
            f"digest={unit.materialization_artifact_digest}:"
            f"lock={unit.runtime_lock.lock_digest}:"
            f"site={site_packages.name}:{warmup}"
        )
        unit.readiness_evidence_ref = evidence
        unit.lifecycle_state = DeploymentInstanceState.READY
        return evidence

    def _fail_unit(self, serving_unit_ref: str, unit: _ColocatedVenvPhysicalUnit) -> None:
        with self._lock:
            unit.lifecycle_state = DeploymentInstanceState.FAILED
            self._remove_unit(serving_unit_ref, unit)

    def begin_drain_for_bound_serving_unit(self, serving_unit_ref: str) -> None:
        """
        Close physical execution admission for one bound unit (adapter-owned gate).

        Used when session-close drain must not depend on durable revision evidence.
        """
        with self._lock:
            unit = self._require_unit(serving_unit_ref)
            unit.coordinator_admission_open_staged = None
            unit.admission.begin_drain()
            unit.lifecycle_state = DeploymentInstanceState.DRAINING
            unit.drain_started_monotonic = time.monotonic()

    def begin_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        self.begin_drain_for_bound_serving_unit(serving_unit_ref)

    def check_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
        policy: DrainPolicy,
    ) -> DrainStatus:
        unit = self._require_unit(serving_unit_ref)
        in_flight = unit.admission.in_flight_count()
        if in_flight > 0:
            if unit.drain_started_monotonic is not None:
                elapsed = time.monotonic() - unit.drain_started_monotonic
                if elapsed > policy.timeout_seconds:
                    return DrainStatus(
                        completed=False,
                        evidence_ref=f"drain:timeout:in_flight={in_flight}",
                    )
            return DrainStatus(
                completed=False,
                evidence_ref=f"drain:in-flight:{in_flight}",
            )
        return DrainStatus(
            completed=True,
            evidence_ref=f"drain:complete:{serving_unit_ref}",
        )

    def stop(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        with self._lock:
            unit = self._units.get(serving_unit_ref)
            if unit is None:
                return
            unit.invalidate_resources()
            unit.lifecycle_state = DeploymentInstanceState.STOPPED
            if unit.binding_key:
                self._binding_index.pop(unit.binding_key, None)

    def resume_serving(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        unit = self._require_unit(serving_unit_ref)
        if unit.lifecycle_state is DeploymentInstanceState.STOPPED:
            raise VenvBundleColocatedDeploymentError(
                "stopped serving unit cannot be resumed"
            )
        if not unit.runtime_resource.is_live:
            raise VenvBundleColocatedDeploymentError(
                "serving unit physical runtime is not initialized"
            )
        if unit.lifecycle_state not in {
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
            DeploymentInstanceState.DRAINING,
        }:
            raise VenvBundleColocatedDeploymentError("serving unit cannot be resumed")
        unit.admission.begin_drain()
        unit.serving_admission_authorized = False
        unit.coordinator_admission_open_staged = None
        unit.coordinator_verified_published_pointer_revision = None
        unit.lifecycle_state = DeploymentInstanceState.READY
        unit.drain_started_monotonic = None
        evidence = unit.readiness_evidence_ref or self.check_readiness(
            revision, serving_unit_ref=serving_unit_ref
        )
        return f"readiness:resume:{evidence}"

    def attest_serving_unit(
        self,
        revision: RuntimeRevision,
        *,
        request: ServingUnitAttestationRequest,
    ) -> ServingUnitAttestation:
        unit = self._units.get(request.serving_unit_ref)
        if unit is None:
            raise ServingUnitAttestationError(
                "serving unit reference is unknown to deployment adapter",
                reason_code="unknown_serving_unit",
            )
        if request.application_id != unit.application_id:
            raise ServingUnitAttestationError(
                "application identity mismatch for serving unit",
                reason_code="application_identity_mismatch",
            )
        if request.application_environment_id != unit.application_environment_id:
            raise ServingUnitAttestationError(
                "application environment identity mismatch for serving unit",
                reason_code="environment_identity_mismatch",
            )
        if request.runtime_revision_id != unit.runtime_revision_id:
            raise ServingUnitAttestationError(
                "runtime revision identity mismatch for serving unit",
                reason_code="runtime_revision_mismatch",
            )
        if request.materialization_artifact_digest != unit.materialization_artifact_digest:
            raise ServingUnitAttestationError(
                "materialization artifact digest mismatch for serving unit",
                reason_code="artifact_digest_mismatch",
            )
        if request.artifact_locator != unit.artifact_locator:
            raise ServingUnitAttestationError(
                "artifact locator mismatch for serving unit",
                reason_code="artifact_locator_mismatch",
            )
        if revision.runtime_revision_id != unit.runtime_revision_id:
            raise ServingUnitAttestationError(
                "revision scope mismatch for serving unit attestation",
                reason_code="revision_scope_mismatch",
            )
        if unit.lifecycle_state not in {
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
        }:
            raise ServingUnitAttestationError(
                "serving unit is not live",
                reason_code="unit_not_live",
            )
        if not unit.runtime_resource.is_live:
            raise ServingUnitAttestationError(
                "serving unit physical resources are not live",
                reason_code="unit_not_live",
            )
        try:
            verify_venv_bundle_artifact_bytes(
                unit.artifact_root, unit.materialization_artifact_digest
            )
        except VenvBundleDeploymentArtifactError as exc:
            raise ServingUnitAttestationError(
                str(exc),
                reason_code="artifact_integrity_failure",
            ) from exc

        return ServingUnitAttestation(
            application_id=unit.application_id,
            application_environment_id=unit.application_environment_id,
            serving_unit_ref=request.serving_unit_ref,
            runtime_revision_id=unit.runtime_revision_id,
            materialization_artifact_digest=unit.materialization_artifact_digest,
            physical_unit_live=True,
            evidence_ref=f"attest:{unit.serving_unit_ref}:{unit.readiness_evidence_ref or 'unready'}",
            attested_at=datetime.now(UTC),
        )

    def _require_unit(self, serving_unit_ref: str) -> _ColocatedVenvPhysicalUnit:
        unit = self._units.get(serving_unit_ref)
        if unit is None:
            raise VenvBundleColocatedDeploymentError("unknown serving unit")
        return unit

    def _require_live_unit(self, serving_unit_ref: str) -> _ColocatedVenvPhysicalUnit:
        unit = self._require_unit(serving_unit_ref)
        if unit.lifecycle_state is DeploymentInstanceState.STOPPED:
            raise VenvBundleColocatedDeploymentError("serving unit is stopped")
        if not unit.runtime_resource.is_live:
            raise VenvBundleColocatedDeploymentError("serving unit resources are invalid")
        return unit
