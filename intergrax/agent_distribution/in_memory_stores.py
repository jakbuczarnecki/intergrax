# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference store implementations for Agent Distribution (AP-4)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.deployment import DeploymentInstanceRecord, DeploymentInstanceState
from intergrax.agent_distribution.errors import (
    BindingRevisionConflict,
    InstallationSlotConflict,
    MaterializedRuntimeLockConflict,
    RuntimeActivationConflict,
    RuntimeActivationError,
    RuntimeRevisionConflict,
    RuntimeRollbackError,
)
from intergrax.agent_distribution.installation import AgentInstallationRecord, InstallationState
from intergrax.agent_distribution.runtime_revision import RuntimeRevision, RuntimeRevisionState
from intergrax.agent_distribution.stores import (
    ActivationAtomicCommitResult,
    AgentArtifactMetadata,
    ApplicationEnvironmentServingRecord,
    RollbackAtomicCommitResult,
)


@dataclass
class AgentDistributionStoreState:
    """Shared durable backing state — survives store instance recreation."""

    installations: dict[str, AgentInstallationRecord] = field(default_factory=dict)
    active_installation_by_slot: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, ApplicationAgentBinding] = field(default_factory=dict)
    revisions: dict[str, RuntimeRevision] = field(default_factory=dict)
    active_revision_by_environment: dict[str, str] = field(default_factory=dict)
    artifact_metadata: dict[str, AgentArtifactMetadata] = field(default_factory=dict)
    locks: dict[str, MaterializedRuntimeLock] = field(default_factory=dict)
    deployment_instances: dict[tuple[str, str], DeploymentInstanceRecord] = field(default_factory=dict)
    serving_records: dict[str, ApplicationEnvironmentServingRecord] = field(default_factory=dict)
    _activation_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)


class InMemoryAgentInstallationStore:
    """Process-local installation store with slot-serialized mutations."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()
        self._fail_after_prior_demotion = False

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def get_installation(self, installation_id: str) -> AgentInstallationRecord | None:
        with self._lock:
            return self._state.installations.get(installation_id)

    def get_active_installation_for_slot(
        self,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        with self._lock:
            active_id = self._state.active_installation_by_slot.get(installation_slot_id)
            if active_id is None:
                return None
            return self._state.installations.get(active_id)

    def list_installations_for_slot(
        self,
        installation_slot_id: str,
    ) -> list[AgentInstallationRecord]:
        with self._lock:
            return [
                record
                for record in self._state.installations.values()
                if record.installation_slot_id == installation_slot_id
            ]

    def persist_installation(
        self,
        record: AgentInstallationRecord,
        *,
        expected_active_installation_id: str | None = None,
    ) -> AgentInstallationRecord:
        with self._lock:
            current_active_id = self._state.active_installation_by_slot.get(record.installation_slot_id)
            if expected_active_installation_id is not None and current_active_id != expected_active_installation_id:
                raise InstallationSlotConflict(
                    "installation slot active pointer does not match expected value"
                )
            if record.active_for_slot:
                if (
                    current_active_id is not None
                    and current_active_id != record.installation_id
                    and record.installation_state is InstallationState.INSTALLED_ACTIVE
                ):
                    raise InstallationSlotConflict(
                        "slot already has a different active installation"
                    )
                self._state.active_installation_by_slot[record.installation_slot_id] = record.installation_id
            elif current_active_id == record.installation_id:
                self._state.active_installation_by_slot.pop(record.installation_slot_id, None)
            self._state.installations[record.installation_id] = record
            return record

    def atomic_promote_active_installation(
        self,
        *,
        demoted_prior: AgentInstallationRecord | None,
        promoted: AgentInstallationRecord,
        expected_active_installation_id: str | None,
    ) -> tuple[AgentInstallationRecord, AgentInstallationRecord | None]:
        """Atomically demote prior active and promote verified installation."""
        with self._lock:
            current_active_id = self._state.active_installation_by_slot.get(promoted.installation_slot_id)
            if expected_active_installation_id is not None and current_active_id != expected_active_installation_id:
                raise InstallationSlotConflict(
                    "installation slot active pointer does not match expected value"
                )
            if demoted_prior is not None:
                if demoted_prior.active_for_slot:
                    raise InstallationSlotConflict("demoted prior record cannot remain active_for_slot")
                if self._fail_after_prior_demotion:
                    raise InstallationSlotConflict("simulated persistence failure during slot promotion")
                self._state.installations[demoted_prior.installation_id] = demoted_prior
            self._state.installations[promoted.installation_id] = promoted
            self._state.active_installation_by_slot[promoted.installation_slot_id] = promoted.installation_id
            return promoted, demoted_prior


class InMemoryApplicationAgentBindingStore:
    """Process-local binding store with optimistic revision concurrency."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def get_binding(self, application_binding_id: str) -> ApplicationAgentBinding | None:
        with self._lock:
            return self._state.bindings.get(application_binding_id)

    def list_bindings_for_environment(
        self,
        application_environment_id: str,
    ) -> list[ApplicationAgentBinding]:
        with self._lock:
            return [
                binding
                for binding in self._state.bindings.values()
                if binding.application_environment_id == application_environment_id
            ]

    def list_bindings_for_slot(self, installation_slot_id: str) -> list[ApplicationAgentBinding]:
        with self._lock:
            return [
                binding
                for binding in self._state.bindings.values()
                if binding.installation_slot_id == installation_slot_id
            ]

    def persist_binding(
        self,
        binding: ApplicationAgentBinding,
        *,
        expected_revision: int | None = None,
    ) -> ApplicationAgentBinding:
        with self._lock:
            current = self._state.bindings.get(binding.application_binding_id)
            if expected_revision is not None:
                if current is None:
                    if expected_revision != binding.binding_revision:
                        raise BindingRevisionConflict("binding revision conflict on create")
                elif current.binding_revision != expected_revision:
                    raise BindingRevisionConflict("binding revision conflict")
            self._state.bindings[binding.application_binding_id] = binding
            return binding


class InMemoryRuntimeRevisionStore:
    """Process-local runtime revision store with atomic activation swap."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()
        self._fail_after_prior_supersede = False

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def get_revision(self, runtime_revision_id: str) -> RuntimeRevision | None:
        with self._lock:
            return self._state.revisions.get(runtime_revision_id)

    def get_active_revision(
        self,
        application_environment_id: str,
    ) -> RuntimeRevision | None:
        with self._lock:
            active_id = self._state.active_revision_by_environment.get(application_environment_id)
            if active_id is None:
                return None
            return self._state.revisions.get(active_id)

    def persist_candidate_revision(
        self,
        revision: RuntimeRevision,
        *,
        expected_revision_state: RuntimeRevisionState | None = None,
    ) -> RuntimeRevision:
        with self._lock:
            current = self._state.revisions.get(revision.runtime_revision_id)
            if expected_revision_state is not None:
                if current is None or current.revision_state != expected_revision_state:
                    raise RuntimeRevisionConflict("runtime revision state does not match expected value")
            self._state.revisions[revision.runtime_revision_id] = revision
            return revision

    def swap_active_revision(
        self,
        *,
        application_environment_id: str,
        new_active_revision_id: str,
        prior_active_revision_id: str | None = None,
    ) -> RuntimeRevision:
        with self._lock:
            current_active_id = self._state.active_revision_by_environment.get(application_environment_id)
            if prior_active_revision_id is not None and current_active_id != prior_active_revision_id:
                raise RuntimeRevisionConflict("active runtime revision does not match expected prior")
            new_revision = self._state.revisions.get(new_active_revision_id)
            if new_revision is None:
                raise RuntimeRevisionConflict("new active runtime revision was not found")
            if new_revision.revision_state is not RuntimeRevisionState.VALIDATED:
                raise RuntimeRevisionConflict("only validated revisions may become active")
            if self._fail_after_prior_supersede:
                raise RuntimeRevisionConflict("simulated persistence failure during activation swap")
            self._state.active_revision_by_environment[application_environment_id] = new_active_revision_id
            return new_revision

    def atomic_activate_revision(
        self,
        *,
        application_environment_id: str,
        promoted: RuntimeRevision,
        demoted_prior: RuntimeRevision | None,
        expected_prior_active_revision_id: str | None,
    ) -> tuple[RuntimeRevision, RuntimeRevision | None]:
        with self._lock:
            current_active_id = self._state.active_revision_by_environment.get(application_environment_id)
            if (
                expected_prior_active_revision_id is not None
                and current_active_id != expected_prior_active_revision_id
            ):
                raise RuntimeRevisionConflict("active runtime revision does not match expected prior")
            if demoted_prior is not None:
                if self._fail_after_prior_supersede:
                    raise RuntimeRevisionConflict("simulated persistence failure during activation swap")
                self._state.revisions[demoted_prior.runtime_revision_id] = demoted_prior
            self._state.revisions[promoted.runtime_revision_id] = promoted
            self._state.active_revision_by_environment[application_environment_id] = promoted.runtime_revision_id
            return promoted, demoted_prior


class InMemoryAgentArtifactMetadataStore:
    """Process-local artifact metadata store."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()

    def get_by_digest(self, package_digest: str) -> AgentArtifactMetadata | None:
        with self._lock:
            return self._state.artifact_metadata.get(package_digest)

    def persist_metadata(self, metadata: AgentArtifactMetadata) -> AgentArtifactMetadata:
        with self._lock:
            self._state.artifact_metadata[metadata.package_digest] = metadata
            return metadata


class InMemoryMaterializedRuntimeLockStore:
    """Process-local immutable lock artifact store."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def get_lock(self, lock_id: str) -> MaterializedRuntimeLock | None:
        with self._lock:
            return self._state.locks.get(lock_id)

    def get_lock_by_digest(self, lock_digest: str) -> MaterializedRuntimeLock | None:
        with self._lock:
            return self._state.locks.get(lock_digest)

    def persist_lock(self, lock: MaterializedRuntimeLock) -> MaterializedRuntimeLock:
        with self._lock:
            canonical = lock.with_content_identity()
            if lock.lock_id is not None or lock.lock_digest is not None:
                if (
                    lock.lock_id != canonical.lock_id
                    or lock.lock_digest != canonical.lock_digest
                ):
                    raise MaterializedRuntimeLockConflict(
                        "lock identity does not match semantic content"
                    )
            identity = canonical
            existing = self._state.locks.get(identity.lock_id)
            if existing is not None:
                if existing.compute_lock_digest() != identity.compute_lock_digest():
                    raise MaterializedRuntimeLockConflict(
                        "lock_id collision with different semantic content"
                    )
                return existing
            self._state.locks[identity.lock_id] = identity
            return identity


def _deployment_instance_key(
    application_environment_id: str,
    runtime_revision_id: str,
) -> tuple[str, str]:
    return (application_environment_id, runtime_revision_id)


class InMemoryDeploymentInstanceStore:
    """Process-local deployment instance store with optimistic concurrency."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def get_instance(
        self,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> DeploymentInstanceRecord | None:
        with self._lock:
            return self._state.deployment_instances.get(
                _deployment_instance_key(application_environment_id, runtime_revision_id)
            )

    def persist_instance(self, instance: DeploymentInstanceRecord) -> DeploymentInstanceRecord:
        with self._lock:
            key = _deployment_instance_key(
                instance.application_environment_id,
                instance.runtime_revision_id,
            )
            self._state.deployment_instances[key] = instance
            return instance

    def update_instance(
        self,
        instance: DeploymentInstanceRecord,
        *,
        expected_state: DeploymentInstanceState | None = None,
        expected_record_revision: int | None = None,
    ) -> DeploymentInstanceRecord:
        with self._lock:
            key = _deployment_instance_key(
                instance.application_environment_id,
                instance.runtime_revision_id,
            )
            current = self._state.deployment_instances.get(key)
            if current is None:
                raise RuntimeActivationConflict("deployment instance was not found")
            if expected_state is not None and current.instance_state != expected_state:
                raise RuntimeActivationConflict("deployment instance state does not match expected value")
            if (
                expected_record_revision is not None
                and current.record_revision != expected_record_revision
            ):
                raise RuntimeActivationConflict("deployment instance revision does not match expected value")
            self._state.deployment_instances[key] = instance
            return instance


class InMemoryApplicationEnvironmentServingStore:
    """Process-local serving pointer store with CAS-protected swaps."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._lock = threading.RLock()

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def get_serving_record(
        self,
        application_environment_id: str,
    ) -> ApplicationEnvironmentServingRecord | None:
        with self._lock:
            return self._state.serving_records.get(application_environment_id)

    def atomic_swap_serving_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str | None,
        expected_pointer_revision: int,
        new_revision_id: str,
        prior_revision_id: str | None,
        committed_at: datetime,
    ) -> ApplicationEnvironmentServingRecord:
        with self._lock:
            current = self._state.serving_records.get(application_environment_id)
            if current is None:
                if expected_pointer_revision != 0 or expected_current_revision_id is not None:
                    raise RuntimeActivationConflict("serving pointer does not match expected value")
                record = ApplicationEnvironmentServingRecord(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                    traffic_serving_revision_id=new_revision_id,
                    serving_pointer_revision=1,
                    prior_traffic_revision_id=prior_revision_id,
                    committed_at=committed_at,
                )
            else:
                if current.traffic_serving_revision_id != expected_current_revision_id:
                    raise RuntimeActivationConflict("serving pointer does not match expected value")
                if current.serving_pointer_revision != expected_pointer_revision:
                    raise RuntimeActivationConflict("serving pointer revision does not match expected value")
                record = ApplicationEnvironmentServingRecord(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                    traffic_serving_revision_id=new_revision_id,
                    serving_pointer_revision=current.serving_pointer_revision + 1,
                    prior_traffic_revision_id=prior_revision_id,
                    committed_at=committed_at,
                )
            self._state.serving_records[application_environment_id] = record
            return record


class InMemoryApplicationEnvironmentActivationStore:
    """Shared-state activation commit store — one critical section per cutover."""

    def __init__(self, state: AgentDistributionStoreState | None = None) -> None:
        self._state = state or AgentDistributionStoreState()
        self._fail_atomic_commit_activation = False
        self._fail_atomic_commit_rollback = False

    @property
    def state(self) -> AgentDistributionStoreState:
        return self._state

    def atomic_commit_activation(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str | None,
        expected_pointer_revision: int,
        candidate_revision_id: str,
        expected_artifact_digest: str,
        committed_at: datetime,
    ) -> ActivationAtomicCommitResult:
        with self._state._activation_lock:
            serving = self._state.serving_records.get(application_environment_id)
            if (
                serving is not None
                and serving.traffic_serving_revision_id == candidate_revision_id
            ):
                return self._idempotent_activation_result(
                    application_environment_id=application_environment_id,
                    candidate_revision_id=candidate_revision_id,
                    serving=serving,
                )

            candidate = self._state.revisions.get(candidate_revision_id)
            if candidate is None:
                raise RuntimeActivationError("candidate runtime revision was not found")
            if candidate.application_environment_id != application_environment_id:
                raise RuntimeActivationError("candidate runtime revision environment mismatch")
            if candidate.revision_state is not RuntimeRevisionState.VALIDATED:
                raise RuntimeActivationError("commit requires validated runtime revision")
            if candidate.materialization_artifact_digest != expected_artifact_digest:
                raise RuntimeActivationError("artifact identity mismatch at commit")

            candidate_key = _deployment_instance_key(
                application_environment_id,
                candidate_revision_id,
            )
            candidate_instance = self._state.deployment_instances.get(candidate_key)
            if candidate_instance is None:
                raise RuntimeActivationError("deployment instance does not exist")
            if candidate_instance.instance_state is not DeploymentInstanceState.READY:
                raise RuntimeActivationError("commit requires ready deployment instance")
            if candidate_instance.readiness_evidence_ref is None:
                raise RuntimeActivationError("ready deployment instance lacks readiness evidence")

            if serving is None:
                if expected_pointer_revision != 0 or expected_current_revision_id is not None:
                    raise RuntimeActivationConflict("serving pointer does not match expected value")
            else:
                if serving.traffic_serving_revision_id != expected_current_revision_id:
                    raise RuntimeActivationConflict("serving pointer does not match expected value")
                if serving.serving_pointer_revision != expected_pointer_revision:
                    raise RuntimeActivationConflict("serving pointer revision does not match expected value")

            prior_revision: RuntimeRevision | None = None
            prior_instance: DeploymentInstanceRecord | None = None
            if expected_current_revision_id is not None:
                prior_revision = self._state.revisions.get(expected_current_revision_id)
                if prior_revision is None:
                    raise RuntimeActivationError("prior runtime revision was not found")
                if prior_revision.revision_state is not RuntimeRevisionState.ACTIVE:
                    raise RuntimeActivationError("prior runtime revision is not active")
                prior_key = _deployment_instance_key(
                    application_environment_id,
                    expected_current_revision_id,
                )
                prior_instance = self._state.deployment_instances.get(prior_key)
                if prior_instance is None:
                    raise RuntimeActivationError("prior deployment instance does not exist")
                if prior_instance.instance_state is not DeploymentInstanceState.SERVING:
                    raise RuntimeActivationError("prior deployment instance is not serving")

            if self._fail_atomic_commit_activation:
                raise RuntimeActivationConflict("simulated atomic activation commit failure")

            promoted = candidate.model_copy(
                update={
                    "revision_state": RuntimeRevisionState.ACTIVE,
                    "activated_at": committed_at,
                    "supersedes_revision_id": expected_current_revision_id,
                    "rollback_target_revision_id": expected_current_revision_id,
                }
            )
            self._state.revisions[candidate_revision_id] = promoted

            demoted_prior: RuntimeRevision | None = None
            if prior_revision is not None:
                demoted_prior = prior_revision.model_copy(
                    update={"revision_state": RuntimeRevisionState.SUPERSEDED}
                )
                self._state.revisions[expected_current_revision_id] = demoted_prior

            self._state.active_revision_by_environment[application_environment_id] = candidate_revision_id

            serving_instance = candidate_instance.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.SERVING,
                    "record_revision": candidate_instance.record_revision + 1,
                }
            )
            self._state.deployment_instances[candidate_key] = serving_instance

            draining_prior: DeploymentInstanceRecord | None = None
            if prior_instance is not None:
                draining_prior = prior_instance.model_copy(
                    update={
                        "instance_state": DeploymentInstanceState.DRAINING,
                        "drain_started_at": committed_at,
                        "record_revision": prior_instance.record_revision + 1,
                    }
                )
                prior_key = _deployment_instance_key(
                    application_environment_id,
                    expected_current_revision_id or "",
                )
                self._state.deployment_instances[prior_key] = draining_prior

            new_pointer_revision = 1 if serving is None else serving.serving_pointer_revision + 1
            serving_record = ApplicationEnvironmentServingRecord(
                application_id=application_id,
                application_environment_id=application_environment_id,
                traffic_serving_revision_id=candidate_revision_id,
                serving_pointer_revision=new_pointer_revision,
                prior_traffic_revision_id=expected_current_revision_id,
                committed_at=committed_at,
            )
            self._state.serving_records[application_environment_id] = serving_record

            return ActivationAtomicCommitResult(
                serving_record=serving_record,
                activated_revision=promoted,
                candidate_instance=serving_instance,
                prior_instance=draining_prior,
                demoted_prior_revision=demoted_prior,
            )

    def atomic_commit_rollback(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str,
        expected_pointer_revision: int,
        target_revision_id: str,
        committed_at: datetime,
    ) -> RollbackAtomicCommitResult:
        with self._state._activation_lock:
            serving = self._state.serving_records.get(application_environment_id)
            if serving is None:
                raise RuntimeActivationConflict("no serving record for rollback")

            if serving.traffic_serving_revision_id == target_revision_id:
                return self._idempotent_rollback_result(
                    application_environment_id=application_environment_id,
                    target_revision_id=target_revision_id,
                    serving=serving,
                )

            if serving.traffic_serving_revision_id != expected_current_revision_id:
                raise RuntimeActivationConflict("rollback serving pointer does not match expected current")
            if serving.serving_pointer_revision != expected_pointer_revision:
                raise RuntimeActivationConflict("rollback serving pointer revision mismatch")
            if serving.prior_traffic_revision_id != target_revision_id:
                raise RuntimeActivationError("rollback target revision is not prior traffic revision")

            target_revision = self._state.revisions.get(target_revision_id)
            if target_revision is None:
                raise RuntimeActivationError("rollback target runtime revision was not found")
            if target_revision.revision_state is not RuntimeRevisionState.SUPERSEDED:
                raise RuntimeActivationError("prior revision is not rollback-eligible")

            current_revision = self._state.revisions.get(expected_current_revision_id)
            if current_revision is None:
                raise RuntimeActivationError("current runtime revision was not found")
            if current_revision.revision_state is not RuntimeRevisionState.ACTIVE:
                raise RuntimeActivationError("current runtime revision is not active")

            target_key = _deployment_instance_key(application_environment_id, target_revision_id)
            target_instance = self._state.deployment_instances.get(target_key)
            if target_instance is None:
                raise RuntimeActivationError("rollback target deployment instance does not exist")
            if target_instance.instance_state is not DeploymentInstanceState.READY:
                raise RuntimeActivationError("rollback target deployment instance is not ready")

            current_key = _deployment_instance_key(
                application_environment_id,
                expected_current_revision_id,
            )
            current_instance = self._state.deployment_instances.get(current_key)
            if current_instance is None:
                raise RuntimeActivationError("current deployment instance does not exist")
            if current_instance.instance_state not in {
                DeploymentInstanceState.SERVING,
                DeploymentInstanceState.FAILED,
            }:
                raise RuntimeActivationError("current deployment instance is not serving")

            if self._fail_atomic_commit_rollback:
                raise RuntimeActivationConflict("simulated atomic rollback commit failure")

            restored = target_revision.model_copy(
                update={
                    "revision_state": RuntimeRevisionState.ACTIVE,
                    "activated_at": committed_at,
                }
            )
            demoted_current = current_revision.model_copy(
                update={"revision_state": RuntimeRevisionState.SUPERSEDED}
            )
            self._state.revisions[target_revision_id] = restored
            self._state.revisions[expected_current_revision_id] = demoted_current
            self._state.active_revision_by_environment[application_environment_id] = target_revision_id

            restored_instance = target_instance.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.SERVING,
                    "record_revision": target_instance.record_revision + 1,
                }
            )
            self._state.deployment_instances[target_key] = restored_instance

            draining_current = current_instance.model_copy(
                update={
                    "instance_state": DeploymentInstanceState.DRAINING,
                    "drain_started_at": committed_at,
                    "record_revision": current_instance.record_revision + 1,
                }
            )
            self._state.deployment_instances[current_key] = draining_current

            serving_record = ApplicationEnvironmentServingRecord(
                application_id=application_id,
                application_environment_id=application_environment_id,
                traffic_serving_revision_id=target_revision_id,
                serving_pointer_revision=serving.serving_pointer_revision + 1,
                prior_traffic_revision_id=expected_current_revision_id,
                committed_at=committed_at,
            )
            self._state.serving_records[application_environment_id] = serving_record

            return RollbackAtomicCommitResult(
                serving_record=serving_record,
                restored_revision=restored,
                restored_instance=restored_instance,
                demoted_current_revision=demoted_current,
                superseded_instance=draining_current,
            )

    def _idempotent_activation_result(
        self,
        *,
        application_environment_id: str,
        candidate_revision_id: str,
        serving: ApplicationEnvironmentServingRecord,
    ) -> ActivationAtomicCommitResult:
        candidate = self._state.revisions.get(candidate_revision_id)
        if candidate is None or candidate.revision_state is not RuntimeRevisionState.ACTIVE:
            raise RuntimeActivationConflict(
                "serving pointer already targets revision but state is not active"
            )
        candidate_key = _deployment_instance_key(application_environment_id, candidate_revision_id)
        candidate_instance = self._state.deployment_instances.get(candidate_key)
        if (
            candidate_instance is None
            or candidate_instance.instance_state is not DeploymentInstanceState.SERVING
        ):
            raise RuntimeActivationConflict("idempotent commit requires serving deployment instance")

        prior_instance: DeploymentInstanceRecord | None = None
        demoted_prior: RuntimeRevision | None = None
        if serving.prior_traffic_revision_id is not None:
            prior_key = _deployment_instance_key(
                application_environment_id,
                serving.prior_traffic_revision_id,
            )
            prior_instance = self._state.deployment_instances.get(prior_key)
            demoted_prior = self._state.revisions.get(serving.prior_traffic_revision_id)

        return ActivationAtomicCommitResult(
            serving_record=serving,
            activated_revision=candidate,
            candidate_instance=candidate_instance,
            prior_instance=prior_instance,
            demoted_prior_revision=demoted_prior,
        )

    def _idempotent_rollback_result(
        self,
        *,
        application_environment_id: str,
        target_revision_id: str,
        serving: ApplicationEnvironmentServingRecord,
    ) -> RollbackAtomicCommitResult:
        restored = self._state.revisions.get(target_revision_id)
        if restored is None or restored.revision_state is not RuntimeRevisionState.ACTIVE:
            raise RuntimeRollbackError("idempotent rollback requires active restored revision")

        current_revision_id = serving.prior_traffic_revision_id
        if current_revision_id is None:
            raise RuntimeRollbackError("idempotent rollback requires superseded current revision")
        demoted_current = self._state.revisions.get(current_revision_id)
        if demoted_current is None or demoted_current.revision_state is not RuntimeRevisionState.SUPERSEDED:
            raise RuntimeRollbackError("idempotent rollback requires superseded current revision")

        target_key = _deployment_instance_key(application_environment_id, target_revision_id)
        restored_instance = self._state.deployment_instances.get(target_key)
        if (
            restored_instance is None
            or restored_instance.instance_state is not DeploymentInstanceState.SERVING
        ):
            raise RuntimeRollbackError("idempotent rollback requires serving prior instance")

        superseded_instance = self._state.deployment_instances.get(
            _deployment_instance_key(application_environment_id, current_revision_id)
        )
        return RollbackAtomicCommitResult(
            serving_record=serving,
            restored_revision=restored,
            restored_instance=restored_instance,
            demoted_current_revision=demoted_current,
            superseded_instance=superseded_instance,
        )
