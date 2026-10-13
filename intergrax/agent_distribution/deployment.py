# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Topology-neutral deployment adapter contracts (AGENT_DISTRIBUTION §20)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.runtime_revision import RuntimeRevision

_NON_EMPTY = Field(min_length=1)

SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1: Final = "deployment_instance_record.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class DeploymentInstanceState(StrEnum):
    """Ephemeral per-revision serving facts (§20.4)."""

    PREPARING = "preparing"
    READY = "ready"
    SERVING = "serving"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class DeploymentInstanceRecord(BaseModel):
    """Durable deployment instance record for one revision in one environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1
    runtime_revision_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    instance_state: DeploymentInstanceState
    readiness_evidence_ref: str | None = None
    serving_unit_ref: str | None = None
    prepared_at: datetime | None = None
    ready_at: datetime | None = None
    drain_started_at: datetime | None = None
    drain_completed_at: datetime | None = None
    failure_evidence_ref: str | None = None
    record_revision: int = Field(default=0, ge=0)

    @field_validator(
        "runtime_revision_id",
        "application_id",
        "application_environment_id",
        "readiness_evidence_ref",
        "serving_unit_ref",
        "failure_evidence_ref",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class PreparedDeployment(BaseModel):
    """Adapter output after serving unit allocation (§20.2 step 4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY

    @field_validator("serving_unit_ref", "materialization_artifact_digest")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class DrainActionOnTimeout(StrEnum):
    """Bounded drain policy action when timeout is reached (§20.6)."""

    STOP = "stop"
    MARK_FAILED = "mark_failed"


class DrainPolicy(BaseModel):
    """Bounded drain orchestration policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(gt=0)
    action_on_timeout: DrainActionOnTimeout = DrainActionOnTimeout.STOP


class DrainStatus(BaseModel):
    """Adapter-reported drain completion boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completed: bool
    evidence_ref: str | None = None

    @field_validator("evidence_ref")
    @classmethod
    def _strip_evidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class ServingUnitAttestationError(AgentDistributionError):
    """Serving unit liveness or identity binding could not be attested."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ServingUnitAttestationRequest(BaseModel):
    """Input to topology-neutral serving-unit liveness attestation (§4.5A)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    artifact_locator: str = _NON_EMPTY
    tenant_id: str | None = None

    @field_validator(
        "application_id",
        "application_environment_id",
        "runtime_revision_id",
        "serving_unit_ref",
        "materialization_artifact_digest",
        "artifact_locator",
        "tenant_id",
    )
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class ServingUnitAttestation(BaseModel):
    """Observed serving-unit liveness evidence — not execution or traffic permission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    physical_unit_live: bool
    evidence_ref: str = _NON_EMPTY
    attested_at: datetime

    @field_validator(
        "application_id",
        "application_environment_id",
        "serving_unit_ref",
        "runtime_revision_id",
        "materialization_artifact_digest",
        "evidence_ref",
    )
    @classmethod
    def _strip_attestation_fields(cls, value: str) -> str:
        return _strip_required(value)


class RuntimeDeploymentAdapter(Protocol):
    """Topology-neutral deployment port — OCI, venv bundle, or sidecar."""

    def prepare(
        self,
        revision: RuntimeRevision,
        *,
        artifact_locator: str,
        materialization_output: MaterializationOutput | None = None,
    ) -> PreparedDeployment:
        """Allocate and deploy serving unit without production traffic."""

    def check_readiness(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        """Return readiness evidence ref after health + certification checks."""

    def begin_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        """Signal serving unit to stop accepting new traffic."""

    def check_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
        policy: DrainPolicy,
    ) -> DrainStatus:
        """Report whether in-flight work completed under drain policy."""

    def stop(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        """Terminate serving unit after drain policy satisfied."""

    def resume_serving(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        """Reuse a prior serving unit for rollback when still viable."""

    def attest_serving_unit(
        self,
        revision: RuntimeRevision,
        *,
        request: ServingUnitAttestationRequest,
    ) -> ServingUnitAttestation:
        """Observe unit liveness and digest binding; evidence only — no traffic."""


class _FakeUnitRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: DeploymentInstanceState
    application_id: str
    application_environment_id: str
    runtime_revision_id: str
    materialization_artifact_digest: str
    artifact_locator: str


class FakeInMemoryRuntimeDeploymentAdapter:
    """Deterministic in-memory deployment adapter for orchestration tests."""

    def __init__(self) -> None:
        self._units: dict[str, _FakeUnitRecord] = {}
        self._fail_prepare_for: set[str] = set()
        self._fail_readiness_for: set[str] = set()
        self._fail_begin_drain_for: set[str] = set()
        self._drain_pending: set[str] = set()
        self._drain_timeout_for: set[str] = set()
        self._prepare_count: dict[str, int] = {}

    @property
    def prepare_count(self) -> dict[str, int]:
        return dict(self._prepare_count)

    def fail_prepare(self, runtime_revision_id: str) -> None:
        self._fail_prepare_for.add(runtime_revision_id)

    def fail_readiness(self, runtime_revision_id: str) -> None:
        self._fail_readiness_for.add(runtime_revision_id)

    def fail_begin_drain(self, serving_unit_ref: str) -> None:
        self._fail_begin_drain_for.add(serving_unit_ref)

    def force_drain_timeout(self, serving_unit_ref: str) -> None:
        self._drain_timeout_for.add(serving_unit_ref)

    def complete_drain(self, serving_unit_ref: str) -> None:
        self._drain_pending.discard(serving_unit_ref)
        record = self._units.get(serving_unit_ref)
        if record is not None:
            self._units[serving_unit_ref] = record.model_copy(
                update={"state": DeploymentInstanceState.DRAINING}
            )

    def _unit_key(self, revision: RuntimeRevision) -> str:
        digest = revision.materialization_artifact_digest or "missing"
        return f"{revision.application_environment_id}:{revision.runtime_revision_id}:{digest}"

    def prepare(
        self,
        revision: RuntimeRevision,
        *,
        artifact_locator: str,
        materialization_output: MaterializationOutput | None = None,
    ) -> PreparedDeployment:
        if revision.runtime_revision_id in self._fail_prepare_for:
            raise RuntimeError("simulated deployment prepare failure")
        unit_ref = self._unit_key(revision)
        digest = revision.materialization_artifact_digest
        if digest is None:
            raise RuntimeError("revision lacks materialization artifact digest")
        self._units[unit_ref] = _FakeUnitRecord(
            state=DeploymentInstanceState.PREPARING,
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            materialization_artifact_digest=digest,
            artifact_locator=artifact_locator,
        )
        self._prepare_count[revision.runtime_revision_id] = (
            self._prepare_count.get(revision.runtime_revision_id, 0) + 1
        )
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
        if revision.runtime_revision_id in self._fail_readiness_for:
            raise RuntimeError("simulated readiness failure")
        record = self._units.get(serving_unit_ref)
        if record is None:
            raise RuntimeError("unknown serving unit for readiness")
        self._units[serving_unit_ref] = record.model_copy(
            update={"state": DeploymentInstanceState.READY}
        )
        return f"readiness:{serving_unit_ref}"

    def begin_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        if serving_unit_ref in self._fail_begin_drain_for:
            raise RuntimeError("simulated begin_drain failure")
        record = self._units.get(serving_unit_ref)
        if record is None:
            raise RuntimeError("unknown serving unit for drain")
        self._units[serving_unit_ref] = record.model_copy(
            update={"state": DeploymentInstanceState.DRAINING}
        )
        self._drain_pending.add(serving_unit_ref)

    def check_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
        policy: DrainPolicy,
    ) -> DrainStatus:
        if serving_unit_ref in self._drain_timeout_for:
            return DrainStatus(completed=False, evidence_ref="drain:timeout")
        if serving_unit_ref in self._drain_pending:
            return DrainStatus(completed=False, evidence_ref="drain:in-flight")
        return DrainStatus(completed=True, evidence_ref=f"drain:complete:{serving_unit_ref}")

    def stop(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        record = self._units.get(serving_unit_ref)
        if record is not None:
            self._units[serving_unit_ref] = record.model_copy(
                update={"state": DeploymentInstanceState.STOPPED}
            )
        self._drain_pending.discard(serving_unit_ref)

    def resume_serving(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        record = self._units.get(serving_unit_ref)
        if record is None:
            raise RuntimeError("serving unit cannot be resumed")
        if record.state in {
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
            DeploymentInstanceState.DRAINING,
            DeploymentInstanceState.STOPPED,
        }:
            self._units[serving_unit_ref] = record.model_copy(
                update={"state": DeploymentInstanceState.READY}
            )
            self._drain_pending.discard(serving_unit_ref)
            return f"readiness:resume:{serving_unit_ref}"
        raise RuntimeError("serving unit cannot be resumed")

    def attest_serving_unit(
        self,
        revision: RuntimeRevision,
        *,
        request: ServingUnitAttestationRequest,
    ) -> ServingUnitAttestation:
        record = self._units.get(request.serving_unit_ref)
        if record is None:
            raise ServingUnitAttestationError(
                "serving unit reference is unknown to deployment adapter",
                reason_code="unknown_serving_unit",
            )
        if request.application_id != record.application_id:
            raise ServingUnitAttestationError(
                "application identity mismatch for serving unit",
                reason_code="application_identity_mismatch",
            )
        if request.application_environment_id != record.application_environment_id:
            raise ServingUnitAttestationError(
                "application environment identity mismatch for serving unit",
                reason_code="environment_identity_mismatch",
            )
        if request.runtime_revision_id != record.runtime_revision_id:
            raise ServingUnitAttestationError(
                "runtime revision identity mismatch for serving unit",
                reason_code="runtime_revision_mismatch",
            )
        if request.materialization_artifact_digest != record.materialization_artifact_digest:
            raise ServingUnitAttestationError(
                "materialization artifact digest mismatch for serving unit",
                reason_code="artifact_digest_mismatch",
            )
        if request.artifact_locator != record.artifact_locator:
            raise ServingUnitAttestationError(
                "artifact locator mismatch for serving unit",
                reason_code="artifact_locator_mismatch",
            )
        if revision.runtime_revision_id != record.runtime_revision_id:
            raise ServingUnitAttestationError(
                "revision scope mismatch for serving unit attestation",
                reason_code="revision_scope_mismatch",
            )
        if record.state not in {
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
        }:
            raise ServingUnitAttestationError(
                "serving unit is not live",
                reason_code="unit_not_live",
            )
        return ServingUnitAttestation(
            application_id=record.application_id,
            application_environment_id=record.application_environment_id,
            serving_unit_ref=request.serving_unit_ref,
            runtime_revision_id=record.runtime_revision_id,
            materialization_artifact_digest=record.materialization_artifact_digest,
            physical_unit_live=True,
            evidence_ref=f"attest:{request.serving_unit_ref}",
            attested_at=datetime.now(UTC),
        )
