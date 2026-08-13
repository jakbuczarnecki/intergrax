# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Zero-downtime runtime revision activation orchestration (AGENT_DISTRIBUTION §20, AP-9)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
    DrainActionOnTimeout,
    DrainPolicy,
    DrainStatus,
    RuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    RuntimeActivationConflict,
    RuntimeActivationError,
    RuntimeDrainError,
    RuntimeReadinessError,
    RuntimeRollbackError,
)
from intergrax.agent_distribution.events import TransitionResult
from intergrax.agent_distribution.runtime_revision import RuntimeRevision, RuntimeRevisionState
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingRecord,
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)

_NON_EMPTY = Field(min_length=1)


class RuntimeServingProjectionCoordinator(Protocol):
    """AP-10 coordination boundary — registry projection aligned with traffic commit."""

    def prepare_projection(self, runtime_revision_id: str) -> None:
        """Prepare registry projection inputs for candidate revision."""

    def commit_projection(self, runtime_revision_id: str) -> None:
        """Publish registry projection for traffic-serving revision."""

    def rollback_projection(self, runtime_revision_id: str) -> None:
        """Restore registry projection for rollback target revision."""


class FakeRuntimeServingProjectionCoordinator:
    """Deterministic projection coordinator for orchestration tests."""

    def __init__(self) -> None:
        self.prepared: set[str] = set()
        self.committed: str | None = None
        self.rolled_back: str | None = None
        self.fail_prepare_for: set[str] = set()

    def fail_prepare(self, runtime_revision_id: str) -> None:
        self.fail_prepare_for.add(runtime_revision_id)

    def prepare_projection(self, runtime_revision_id: str) -> None:
        if runtime_revision_id in self.fail_prepare_for:
            raise RuntimeActivationError("simulated projection prepare failure")
        self.prepared.add(runtime_revision_id)

    def commit_projection(self, runtime_revision_id: str) -> None:
        self.committed = runtime_revision_id

    def rollback_projection(self, runtime_revision_id: str) -> None:
        self.rolled_back = runtime_revision_id


class ActivationCommitResult(BaseModel):
    """Outcome of a successful traffic commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_record: ApplicationEnvironmentServingRecord
    activated_revision: RuntimeRevision
    candidate_instance: DeploymentInstanceRecord
    prior_instance: DeploymentInstanceRecord | None = None


class RollbackResult(BaseModel):
    """Outcome of a successful rollback traffic commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_record: ApplicationEnvironmentServingRecord
    restored_revision: RuntimeRevision
    restored_instance: DeploymentInstanceRecord
    superseded_instance: DeploymentInstanceRecord | None = None


class DrainCompletionResult(BaseModel):
    """Outcome of drain orchestration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance: DeploymentInstanceRecord
    drain_status: DrainStatus


@dataclass(frozen=True, slots=True)
class ArtifactRevalidationHook:
    """Rollback trust / artifact presence boundary — injectable for tests."""

    validate: Callable[[RuntimeRevision], None]


def _default_artifact_revalidation(revision: RuntimeRevision) -> None:
    if revision.materialization_artifact_digest is None:
        raise RuntimeRollbackError("rollback target lacks materialization artifact digest")


class ActivationService:
    """PREPARE → READY → COMMIT activation orchestration for one environment."""

    def __init__(
        self,
        *,
        revision_store: RuntimeRevisionStore,
        deployment_instance_store: DeploymentInstanceStore,
        serving_store: ApplicationEnvironmentServingStore,
        deployment_adapter: RuntimeDeploymentAdapter,
        projection_coordinator: RuntimeServingProjectionCoordinator,
        artifact_revalidation: ArtifactRevalidationHook | None = None,
    ) -> None:
        self._revision_store = revision_store
        self._revision_service = RuntimeRevisionService(revision_store)
        self._deployment_instance_store = deployment_instance_store
        self._serving_store = serving_store
        self._deployment_adapter = deployment_adapter
        self._projection_coordinator = projection_coordinator
        self._artifact_revalidation = artifact_revalidation or ArtifactRevalidationHook(
            validate=_default_artifact_revalidation
        )
        self._environment_locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def prepare_candidate(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
        artifact_locator: str,
    ) -> TransitionResult[DeploymentInstanceRecord]:
        with self._environment_lock(application_environment_id):
            revision = self._require_revision(runtime_revision_id)
            self._assert_environment_match(
                revision,
                application_id=application_id,
                application_environment_id=application_environment_id,
            )
            if revision.revision_state is not RuntimeRevisionState.VALIDATED:
                raise RuntimeActivationError("prepare requires validated runtime revision")
            if revision.materialization_artifact_digest is None:
                raise RuntimeActivationError("validated revision lacks materialization artifact identity")

            existing = self._deployment_instance_store.get_instance(
                application_environment_id,
                runtime_revision_id,
            )
            if existing is not None and existing.instance_state is DeploymentInstanceState.READY:
                if existing.serving_unit_ref is None or existing.readiness_evidence_ref is None:
                    raise RuntimeActivationError("ready deployment instance lacks readiness evidence")
                return TransitionResult(value=existing)

            now = datetime.now(UTC)
            prepared = self._deployment_adapter.prepare(
                revision,
                artifact_locator=artifact_locator,
            )
            if prepared.materialization_artifact_digest != revision.materialization_artifact_digest:
                raise RuntimeActivationError("deployment artifact identity mismatch")

            preparing = DeploymentInstanceRecord(
                runtime_revision_id=runtime_revision_id,
                application_id=application_id,
                application_environment_id=application_environment_id,
                instance_state=DeploymentInstanceState.PREPARING,
                serving_unit_ref=prepared.serving_unit_ref,
                prepared_at=now,
                record_revision=existing.record_revision if existing else 0,
            )
            self._deployment_instance_store.persist_instance(preparing)

            try:
                readiness_ref = self._deployment_adapter.check_readiness(
                    revision,
                    serving_unit_ref=prepared.serving_unit_ref,
                )
            except Exception as exc:
                failed = preparing.model_copy(
                    update={
                        "instance_state": DeploymentInstanceState.FAILED,
                        "failure_evidence_ref": f"readiness:{exc}",
                        "record_revision": preparing.record_revision + 1,
                    }
                )
                self._deployment_instance_store.persist_instance(failed)
                raise RuntimeReadinessError("candidate readiness validation failed") from exc

            ready = preparing.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.READY,
                    "readiness_evidence_ref": readiness_ref,
                    "ready_at": datetime.now(UTC),
                    "record_revision": preparing.record_revision + 1,
                }
            )
            persisted = self._deployment_instance_store.persist_instance(ready)
            return TransitionResult(value=persisted)

    def commit_activation(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
        expected_prior_traffic_revision_id: str | None,
        expected_serving_pointer_revision: int,
        expected_artifact_digest: str,
    ) -> TransitionResult[ActivationCommitResult]:
        with self._environment_lock(application_environment_id):
            revision = self._require_revision(runtime_revision_id)
            self._assert_environment_match(
                revision,
                application_id=application_id,
                application_environment_id=application_environment_id,
            )
            if revision.materialization_artifact_digest != expected_artifact_digest:
                raise RuntimeActivationError("artifact identity mismatch at commit")

            serving = self._serving_store.get_serving_record(application_environment_id)
            if serving is not None:
                if serving.traffic_serving_revision_id == runtime_revision_id:
                    if serving.serving_pointer_revision > expected_serving_pointer_revision:
                        return self._idempotent_active_commit(
                            revision=revision,
                            serving=serving,
                            application_environment_id=application_environment_id,
                        )
                    if serving.serving_pointer_revision < expected_serving_pointer_revision:
                        raise RuntimeActivationConflict("stale serving pointer revision")

            if revision.revision_state is not RuntimeRevisionState.VALIDATED:
                raise RuntimeActivationError("commit requires validated runtime revision")

            instance = self._require_ready_instance(
                application_environment_id=application_environment_id,
                runtime_revision_id=runtime_revision_id,
            )

            self._projection_coordinator.prepare_projection(runtime_revision_id)

            now = datetime.now(UTC)
            try:
                serving_record = self._serving_store.atomic_swap_serving_revision(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                    expected_current_revision_id=expected_prior_traffic_revision_id,
                    expected_pointer_revision=expected_serving_pointer_revision,
                    new_revision_id=runtime_revision_id,
                    prior_revision_id=expected_prior_traffic_revision_id,
                    committed_at=now,
                )
            except RuntimeActivationConflict:
                raise
            except Exception as exc:
                raise RuntimeActivationConflict("serving pointer CAS failed") from exc

            try:
                activated = self._revision_service.activate_revision(
                    runtime_revision_id,
                    expected_prior_active_revision_id=expected_prior_traffic_revision_id,
                )
            except Exception as exc:
                raise RuntimeActivationError("revision activation failed after pointer swap") from exc

            serving_instance = instance.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.SERVING,
                    "record_revision": instance.record_revision + 1,
                }
            )
            self._deployment_instance_store.update_instance(
                serving_instance,
                expected_state=DeploymentInstanceState.READY,
                expected_record_revision=instance.record_revision,
            )

            prior_instance: DeploymentInstanceRecord | None = None
            if expected_prior_traffic_revision_id is not None:
                prior = self._deployment_instance_store.get_instance(
                    application_environment_id,
                    expected_prior_traffic_revision_id,
                )
                if prior is not None and prior.serving_unit_ref is not None:
                    prior_revision = self._require_revision(expected_prior_traffic_revision_id)
                    self._deployment_adapter.begin_drain(
                        prior_revision,
                        serving_unit_ref=prior.serving_unit_ref,
                    )
                    prior_instance = prior.model_copy(
                        update={
                            "instance_state": DeploymentInstanceState.DRAINING,
                            "drain_started_at": now,
                            "record_revision": prior.record_revision + 1,
                        }
                    )
                    self._deployment_instance_store.update_instance(
                        prior_instance,
                        expected_state=prior.instance_state,
                        expected_record_revision=prior.record_revision,
                    )

            self._projection_coordinator.commit_projection(runtime_revision_id)

            return TransitionResult(
                value=ActivationCommitResult(
                    serving_record=serving_record,
                    activated_revision=activated.value,
                    candidate_instance=serving_instance,
                    prior_instance=prior_instance,
                ),
                events=activated.events,
            )

    def complete_drain(
        self,
        *,
        application_environment_id: str,
        runtime_revision_id: str,
        policy: DrainPolicy,
    ) -> TransitionResult[DrainCompletionResult]:
        with self._environment_lock(application_environment_id):
            instance = self._deployment_instance_store.get_instance(
                application_environment_id,
                runtime_revision_id,
            )
            if instance is None:
                raise AgentDistributionNotFoundError("deployment instance was not found")
            if instance.instance_state is not DeploymentInstanceState.DRAINING:
                raise RuntimeDrainError("complete_drain requires draining instance")
            if instance.serving_unit_ref is None:
                raise RuntimeDrainError("draining instance lacks serving unit ref")

            revision = self._require_revision(runtime_revision_id)
            status = self._deployment_adapter.check_drain(
                revision,
                serving_unit_ref=instance.serving_unit_ref,
                policy=policy,
            )
            if not status.completed:
                if policy.action_on_timeout is DrainActionOnTimeout.MARK_FAILED:
                    failed = instance.model_copy(
                        update={
                            "instance_state": DeploymentInstanceState.FAILED,
                            "failure_evidence_ref": status.evidence_ref,
                            "record_revision": instance.record_revision + 1,
                        }
                    )
                    self._deployment_instance_store.update_instance(
                        failed,
                        expected_state=DeploymentInstanceState.DRAINING,
                        expected_record_revision=instance.record_revision,
                    )
                    raise RuntimeDrainError("drain timed out")
                raise RuntimeDrainError("drain still in progress")

            self._deployment_adapter.stop(revision, serving_unit_ref=instance.serving_unit_ref)
            stopped = instance.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.STOPPED,
                    "drain_completed_at": datetime.now(UTC),
                    "record_revision": instance.record_revision + 1,
                }
            )
            persisted = self._deployment_instance_store.update_instance(
                stopped,
                expected_state=DeploymentInstanceState.DRAINING,
                expected_record_revision=instance.record_revision,
            )
            return TransitionResult(
                value=DrainCompletionResult(instance=persisted, drain_status=status)
            )

    def rollback(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_traffic_revision_id: str,
        expected_serving_pointer_revision: int,
    ) -> TransitionResult[RollbackResult]:
        with self._environment_lock(application_environment_id):
            serving = self._serving_store.get_serving_record(application_environment_id)
            if serving is None or serving.traffic_serving_revision_id is None:
                raise RuntimeRollbackError("no current serving revision to rollback from")
            if serving.prior_traffic_revision_id is None:
                raise RuntimeRollbackError("no prior traffic revision available for rollback")

            prior_revision_id = serving.prior_traffic_revision_id
            if serving.traffic_serving_revision_id != expected_current_traffic_revision_id:
                if serving.prior_traffic_revision_id == expected_current_traffic_revision_id:
                    demoted = self._require_revision(expected_current_traffic_revision_id)
                    restored = self._require_revision(serving.traffic_serving_revision_id or prior_revision_id)
                    if (
                        demoted.revision_state is RuntimeRevisionState.SUPERSEDED
                        and restored.revision_state is RuntimeRevisionState.ACTIVE
                        and demoted.supersedes_revision_id == restored.runtime_revision_id
                    ):
                        return self._idempotent_rollback_result(
                            serving=serving,
                            application_environment_id=application_environment_id,
                            prior_revision_id=restored.runtime_revision_id,
                        )
                raise RuntimeActivationConflict("rollback serving pointer does not match expected current")
            if serving.serving_pointer_revision != expected_serving_pointer_revision:
                raise RuntimeActivationConflict("rollback serving pointer revision mismatch")

            prior_revision = self._require_revision(prior_revision_id)
            if prior_revision.revision_state is not RuntimeRevisionState.SUPERSEDED:
                raise RuntimeRollbackError("prior revision is not rollback-eligible")

            self._artifact_revalidation.validate(prior_revision)
            self._assert_identity_unchanged(prior_revision)

            current_revision_id = serving.traffic_serving_revision_id
            prior_instance = self._ensure_prior_instance_ready(
                application_id=application_id,
                application_environment_id=application_environment_id,
                prior_revision=prior_revision,
            )

            self._projection_coordinator.prepare_projection(prior_revision_id)

            now = datetime.now(UTC)
            serving_record = self._serving_store.atomic_swap_serving_revision(
                application_id=application_id,
                application_environment_id=application_environment_id,
                expected_current_revision_id=current_revision_id,
                expected_pointer_revision=expected_serving_pointer_revision,
                new_revision_id=prior_revision_id,
                prior_revision_id=current_revision_id,
                committed_at=now,
            )

            restored = prior_revision.model_copy(
                update={
                    "revision_state": RuntimeRevisionState.ACTIVE,
                    "activated_at": now,
                }
            )
            current_revision = self._require_revision(current_revision_id)
            demoted_current = current_revision.model_copy(
                update={"revision_state": RuntimeRevisionState.SUPERSEDED}
            )
            self._revision_store.atomic_activate_revision(
                application_environment_id=application_environment_id,
                promoted=restored,
                demoted_prior=demoted_current,
                expected_prior_active_revision_id=current_revision_id,
            )

            restored_instance = prior_instance.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.SERVING,
                    "record_revision": prior_instance.record_revision + 1,
                }
            )
            self._deployment_instance_store.update_instance(
                restored_instance,
                expected_state=prior_instance.instance_state,
                expected_record_revision=prior_instance.record_revision,
            )

            superseded_instance: DeploymentInstanceRecord | None = None
            current_instance = self._deployment_instance_store.get_instance(
                application_environment_id,
                current_revision_id,
            )
            if current_instance is not None and current_instance.serving_unit_ref is not None:
                self._deployment_adapter.begin_drain(
                    current_revision,
                    serving_unit_ref=current_instance.serving_unit_ref,
                )
                superseded_instance = current_instance.model_copy(
                    update={
                        "instance_state": DeploymentInstanceState.DRAINING,
                        "drain_started_at": now,
                        "record_revision": current_instance.record_revision + 1,
                    }
                )
                self._deployment_instance_store.update_instance(
                    superseded_instance,
                    expected_state=current_instance.instance_state,
                    expected_record_revision=current_instance.record_revision,
                )

            self._projection_coordinator.rollback_projection(prior_revision_id)
            self._projection_coordinator.commit_projection(prior_revision_id)

            return TransitionResult(
                value=RollbackResult(
                    serving_record=serving_record,
                    restored_revision=restored,
                    restored_instance=restored_instance,
                    superseded_instance=superseded_instance,
                )
            )

    def mark_post_cutover_failure(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
        failure_evidence_ref: str,
        attempt_rollback: bool = True,
    ) -> TransitionResult[RollbackResult | None]:
        with self._environment_lock(application_environment_id):
            serving = self._serving_store.get_serving_record(application_environment_id)
            if serving is None or serving.traffic_serving_revision_id != runtime_revision_id:
                raise RuntimeActivationError("post-cutover failure requires current serving revision")

            instance = self._deployment_instance_store.get_instance(
                application_environment_id,
                runtime_revision_id,
            )
            if instance is not None:
                failed = instance.model_copy(
                    update={
                        "instance_state": DeploymentInstanceState.FAILED,
                        "failure_evidence_ref": failure_evidence_ref,
                        "record_revision": instance.record_revision + 1,
                    }
                )
                self._deployment_instance_store.update_instance(
                    failed,
                    expected_state=instance.instance_state,
                    expected_record_revision=instance.record_revision,
                )

            if not attempt_rollback or serving.prior_traffic_revision_id is None:
                return TransitionResult(value=None)

            try:
                return self.rollback(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                    expected_current_traffic_revision_id=runtime_revision_id,
                    expected_serving_pointer_revision=serving.serving_pointer_revision,
                )
            except (RuntimeRollbackError, RuntimeActivationConflict) as exc:
                raise RuntimeRollbackError(
                    "post-cutover rollback failed; serving pointer remains authoritative"
                ) from exc

    def _idempotent_active_commit(
        self,
        *,
        revision: RuntimeRevision,
        serving: ApplicationEnvironmentServingRecord,
        application_environment_id: str,
    ) -> TransitionResult[ActivationCommitResult]:
        if revision.revision_state is not RuntimeRevisionState.ACTIVE:
            raise RuntimeActivationConflict("serving pointer already targets revision but state is not active")
        instance = self._deployment_instance_store.get_instance(
            application_environment_id,
            revision.runtime_revision_id,
        )
        if instance is None or instance.instance_state is not DeploymentInstanceState.SERVING:
            raise RuntimeActivationConflict("idempotent commit requires serving deployment instance")
        return TransitionResult(
            value=ActivationCommitResult(
                serving_record=serving,
                activated_revision=revision,
                candidate_instance=instance,
            )
        )

    def _idempotent_rollback_result(
        self,
        *,
        serving: ApplicationEnvironmentServingRecord,
        application_environment_id: str,
        prior_revision_id: str,
    ) -> TransitionResult[RollbackResult]:
        restored = self._require_revision(prior_revision_id)
        instance = self._deployment_instance_store.get_instance(
            application_environment_id,
            prior_revision_id,
        )
        if instance is None or instance.instance_state is not DeploymentInstanceState.SERVING:
            raise RuntimeRollbackError("idempotent rollback requires serving prior instance")
        return TransitionResult(
            value=RollbackResult(
                serving_record=serving,
                restored_revision=restored,
                restored_instance=instance,
            )
        )

    def _ensure_prior_instance_ready(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        prior_revision: RuntimeRevision,
    ) -> DeploymentInstanceRecord:
        instance = self._deployment_instance_store.get_instance(
            application_environment_id,
            prior_revision.runtime_revision_id,
        )
        if instance is not None and instance.instance_state in {
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
            DeploymentInstanceState.DRAINING,
            DeploymentInstanceState.STOPPED,
        }:
            if instance.serving_unit_ref is None:
                raise RuntimeRollbackError("prior deployment instance lacks serving unit ref")
            if instance.instance_state in {
                DeploymentInstanceState.DRAINING,
                DeploymentInstanceState.STOPPED,
            }:
                readiness_ref = self._deployment_adapter.resume_serving(
                    prior_revision,
                    serving_unit_ref=instance.serving_unit_ref,
                )
                ready = instance.model_copy(
                    update={
                        "instance_state": DeploymentInstanceState.READY,
                        "readiness_evidence_ref": readiness_ref,
                        "ready_at": datetime.now(UTC),
                        "record_revision": instance.record_revision + 1,
                    }
                )
                return self._deployment_instance_store.update_instance(
                    ready,
                    expected_state=instance.instance_state,
                    expected_record_revision=instance.record_revision,
                )
            return instance

        if prior_revision.materialization_artifact_digest is None:
            raise RuntimeRollbackError("prior revision lacks artifact identity for redeploy")

        return self._prepare_prior_serving_unit(
            application_id=application_id,
            application_environment_id=application_environment_id,
            prior_revision=prior_revision,
        )

    def _prepare_prior_serving_unit(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        prior_revision: RuntimeRevision,
    ) -> DeploymentInstanceRecord:
        """Reuse prior immutable artifact — no revision rebuild."""
        artifact_locator = f"artifact://{prior_revision.materialization_artifact_digest}"
        now = datetime.now(UTC)
        prepared = self._deployment_adapter.prepare(
            prior_revision,
            artifact_locator=artifact_locator,
        )
        if prepared.materialization_artifact_digest != prior_revision.materialization_artifact_digest:
            raise RuntimeRollbackError("rollback artifact identity mismatch")

        preparing = DeploymentInstanceRecord(
            runtime_revision_id=prior_revision.runtime_revision_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
            instance_state=DeploymentInstanceState.PREPARING,
            serving_unit_ref=prepared.serving_unit_ref,
            prepared_at=now,
        )
        self._deployment_instance_store.persist_instance(preparing)
        readiness_ref = self._deployment_adapter.check_readiness(
            prior_revision,
            serving_unit_ref=prepared.serving_unit_ref,
        )
        ready = preparing.model_copy(
            update={
                "instance_state": DeploymentInstanceState.READY,
                "readiness_evidence_ref": readiness_ref,
                "ready_at": datetime.now(UTC),
                "record_revision": 1,
            }
        )
        return self._deployment_instance_store.persist_instance(ready)

    def _require_ready_instance(
        self,
        *,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> DeploymentInstanceRecord:
        instance = self._deployment_instance_store.get_instance(
            application_environment_id,
            runtime_revision_id,
        )
        if instance is None:
            raise RuntimeActivationError("deployment instance does not exist")
        if instance.runtime_revision_id != runtime_revision_id:
            raise RuntimeActivationError("deployment instance revision mismatch")
        if instance.application_environment_id != application_environment_id:
            raise RuntimeActivationError("deployment instance environment mismatch")
        if instance.instance_state is not DeploymentInstanceState.READY:
            raise RuntimeActivationError("commit requires ready deployment instance")
        if instance.readiness_evidence_ref is None:
            raise RuntimeActivationError("ready deployment instance lacks readiness evidence")
        return instance

    @staticmethod
    def _assert_environment_match(
        revision: RuntimeRevision,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> None:
        if revision.application_environment_id != application_environment_id:
            raise RuntimeActivationError("runtime revision environment mismatch")

    @staticmethod
    def _assert_identity_unchanged(revision: RuntimeRevision) -> None:
        if revision.materialized_runtime_lock_digest is None:
            raise RuntimeRollbackError("rollback target lacks lock digest")
        if revision.runtime_graph_digest is None:
            raise RuntimeRollbackError("rollback target lacks graph digest")

    def _require_revision(self, runtime_revision_id: str) -> RuntimeRevision:
        revision = self._revision_store.get_revision(runtime_revision_id)
        if revision is None:
            raise AgentDistributionNotFoundError(
                f"runtime revision {runtime_revision_id} was not found"
            )
        return revision

    def _environment_lock(self, application_environment_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._environment_locks.get(application_environment_id)
            if lock is None:
                lock = threading.RLock()
                self._environment_locks[application_environment_id] = lock
            return lock
