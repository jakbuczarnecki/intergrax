# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime revision lifecycle domain service (AGENT_DISTRIBUTION §18, AP-4)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.events import TransitionResult, distribution_event
from intergrax.agent_distribution.runtime_revision import (
    RUNTIME_REVISION_IMMUTABLE_IDENTITY_FIELDS,
    RUNTIME_REVISION_VALIDATION_TRANSITION_FIELDS,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.stores import RuntimeRevisionStore


_ALLOWED_TRANSITIONS: dict[RuntimeRevisionState, frozenset[RuntimeRevisionState]] = {
    RuntimeRevisionState.CANDIDATE: frozenset({RuntimeRevisionState.VALIDATED, RuntimeRevisionState.FAILED}),
    RuntimeRevisionState.VALIDATED: frozenset({RuntimeRevisionState.ACTIVE}),
    RuntimeRevisionState.ACTIVE: frozenset({RuntimeRevisionState.SUPERSEDED}),
}


class RuntimeRevisionService:
    """Transactional runtime revision persistence — no materialization."""

    def __init__(self, store: RuntimeRevisionStore) -> None:
        self._store = store

    def persist_candidate_revision(
        self,
        revision: RuntimeRevision,
    ) -> TransitionResult[RuntimeRevision]:
        if revision.revision_state is not RuntimeRevisionState.CANDIDATE:
            raise RuntimeRevisionLifecycleError("persist_candidate_revision requires candidate state")
        persisted = self._store.persist_candidate_revision(revision)
        return TransitionResult(value=persisted)

    def mark_validated(
        self,
        runtime_revision_id: str,
        *,
        validated_revision: RuntimeRevision,
    ) -> TransitionResult[RuntimeRevision]:
        current = self._require_revision(runtime_revision_id)
        self._require_transition(current.revision_state, RuntimeRevisionState.VALIDATED)
        if validated_revision.runtime_revision_id != runtime_revision_id:
            raise RuntimeRevisionLifecycleError("validated revision id mismatch")
        if validated_revision.revision_state is not RuntimeRevisionState.VALIDATED:
            raise RuntimeRevisionLifecycleError("validated revision must use validated state")
        self._assert_candidate_identity_preserved(current, validated_revision)
        persisted = self._store.persist_candidate_revision(
            validated_revision,
            expected_revision_state=RuntimeRevisionState.CANDIDATE,
        )
        return TransitionResult(value=persisted)

    def activate_revision(
        self,
        runtime_revision_id: str,
        *,
        expected_prior_active_revision_id: str | None = None,
    ) -> TransitionResult[RuntimeRevision]:
        revision = self._require_revision(runtime_revision_id)
        self._require_transition(revision.revision_state, RuntimeRevisionState.ACTIVE)
        if revision.revision_state is not RuntimeRevisionState.VALIDATED:
            raise RuntimeRevisionLifecycleError("only validated revisions may be activated")

        prior_active = self._store.get_active_revision(
            revision.application_id,
            revision.application_environment_id,
        )
        now = datetime.now(UTC)
        demoted_prior: RuntimeRevision | None = None
        rollback_target = prior_active.runtime_revision_id if prior_active is not None else None

        if prior_active is not None:
            demoted_prior = prior_active.model_copy(
                update={"revision_state": RuntimeRevisionState.SUPERSEDED}
            )

        promoted = revision.model_copy(
            update={
                "revision_state": RuntimeRevisionState.ACTIVE,
                "activated_at": now,
                "supersedes_revision_id": prior_active.runtime_revision_id if prior_active else None,
                "rollback_target_revision_id": rollback_target,
            }
        )

        persisted, _ = self._store.atomic_activate_revision(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            promoted=promoted,
            demoted_prior=demoted_prior,
            expected_prior_active_revision_id=expected_prior_active_revision_id,
        )

        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    "runtime_revision.activated",
                    runtime_revision_id,
                    application_environment_id=revision.application_environment_id,
                ),
            ),
        )

    def get_active_revision(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> RuntimeRevision | None:
        return self._store.get_active_revision(application_id, application_environment_id)

    def _require_revision(self, runtime_revision_id: str) -> RuntimeRevision:
        revision = self._store.get_revision(runtime_revision_id)
        if revision is None:
            raise AgentDistributionNotFoundError(
                f"runtime revision {runtime_revision_id} was not found"
            )
        return revision

    @staticmethod
    def _assert_candidate_identity_preserved(
        candidate: RuntimeRevision,
        validated: RuntimeRevision,
    ) -> None:
        candidate_data = candidate.model_dump()
        validated_data = validated.model_dump()
        for field_name in RUNTIME_REVISION_IMMUTABLE_IDENTITY_FIELDS:
            if candidate_data[field_name] != validated_data[field_name]:
                raise RuntimeRevisionLifecycleError(
                    f"immutable runtime revision field {field_name} mutated during validation"
                )
        for field_name in RuntimeRevision.model_fields:
            if field_name in RUNTIME_REVISION_IMMUTABLE_IDENTITY_FIELDS:
                continue
            if field_name in RUNTIME_REVISION_VALIDATION_TRANSITION_FIELDS:
                continue
            if candidate_data[field_name] != validated_data[field_name]:
                raise RuntimeRevisionLifecycleError(
                    f"unexpected runtime revision field {field_name} mutated during validation"
                )

    @staticmethod
    def _require_transition(current: RuntimeRevisionState, target: RuntimeRevisionState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise RuntimeRevisionLifecycleError(
                f"illegal runtime revision transition {current.value} -> {target.value}"
            )
