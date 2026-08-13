# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Topology-neutral deployment adapter contracts (AGENT_DISTRIBUTION §20)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.runtime_revision import RuntimeRevision

_NON_EMPTY = Field(min_length=1)


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


class FakeInMemoryRuntimeDeploymentAdapter:
    """Deterministic in-memory deployment adapter for orchestration tests."""

    def __init__(self) -> None:
        self._units: dict[str, DeploymentInstanceState] = {}
        self._fail_prepare_for: set[str] = set()
        self._fail_readiness_for: set[str] = set()
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

    def force_drain_timeout(self, serving_unit_ref: str) -> None:
        self._drain_timeout_for.add(serving_unit_ref)

    def complete_drain(self, serving_unit_ref: str) -> None:
        self._drain_pending.discard(serving_unit_ref)
        self._units[serving_unit_ref] = DeploymentInstanceState.DRAINING

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
        self._units[unit_ref] = DeploymentInstanceState.PREPARING
        self._prepare_count[revision.runtime_revision_id] = (
            self._prepare_count.get(revision.runtime_revision_id, 0) + 1
        )
        digest = revision.materialization_artifact_digest
        if digest is None:
            raise RuntimeError("revision lacks materialization artifact digest")
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
        self._units[serving_unit_ref] = DeploymentInstanceState.READY
        return f"readiness:{serving_unit_ref}"

    def begin_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        self._units[serving_unit_ref] = DeploymentInstanceState.DRAINING
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
        self._units[serving_unit_ref] = DeploymentInstanceState.STOPPED
        self._drain_pending.discard(serving_unit_ref)

    def resume_serving(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        state = self._units.get(serving_unit_ref)
        if state in {
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
            DeploymentInstanceState.DRAINING,
            DeploymentInstanceState.STOPPED,
        }:
            self._units[serving_unit_ref] = DeploymentInstanceState.READY
            self._drain_pending.discard(serving_unit_ref)
            return f"readiness:resume:{serving_unit_ref}"
        raise RuntimeError("serving unit cannot be resumed")
