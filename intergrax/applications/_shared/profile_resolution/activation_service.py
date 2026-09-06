# © Artur Czarnecki. All rights reserved.

"""Effective profile revision activation orchestration (P1.6)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.applications._shared.capability_dependency import (
    validate_capability_dependencies_for_environment,
)
from intergrax.applications.contracts.profile_resolution.activation import (
    ActivateEffectiveProfileRevisionRequest,
    ActiveEffectiveProfileRevisionBinding,
    ActiveEffectiveProfileRevisionCasOutcome,
    ActiveEffectiveProfileRevisionStore,
    EffectiveProfileActivationResult,
)
from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileActivationConflictError,
    EffectiveProfileActivationRejectedError,
    EffectiveProfileActivationRevisionNotFoundError,
    EffectiveProfileActivationScopeMismatchError,
    EffectiveProfileRevisionError,
    MissingActiveEffectiveProfileRevisionError,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.skills.registry.runtime import SkillRegistry


def resolve_active_effective_profile_revision(
    *,
    active_store: ActiveEffectiveProfileRevisionStore,
    revision_store: EffectiveProfileRevisionStore,
    scope: EffectiveProfileRevisionScope,
) -> EffectiveProfileRevision:
    """Canonical read seam for future admission — one coherent snapshot."""
    binding = active_store.get_active(scope)
    if binding is None:
        raise MissingActiveEffectiveProfileRevisionError(scope=scope)
    if binding.scope != scope:
        raise EffectiveProfileRevisionError("active binding scope mismatch")
    revision = revision_store.get(binding.revision_id, scope=scope)
    if revision is None:
        raise EffectiveProfileActivationRevisionNotFoundError(
            revision_id=binding.revision_id.value,
            scope=scope,
        )
    if revision.fingerprint != binding.fingerprint:
        raise EffectiveProfileRevisionError("active binding fingerprint mismatch")
    if revision.scope != scope:
        raise EffectiveProfileActivationScopeMismatchError(
            "active revision scope does not match requested scope",
        )
    return revision


@dataclass(frozen=True, slots=True)
class EffectiveProfileActivationDependencies:
    """Immutable activation orchestration dependencies."""

    revision_store: EffectiveProfileRevisionStore
    active_store: ActiveEffectiveProfileRevisionStore
    skill_registry: SkillRegistry | None = None
    eligibility_checker: Callable[[EffectiveProfileRevision], None] | None = None


class EffectiveProfileActivationService:
    """Coordinates validation and atomic active pointer publication."""

    def __init__(self, dependencies: EffectiveProfileActivationDependencies) -> None:
        self._dependencies = dependencies

    @property
    def active_store(self) -> ActiveEffectiveProfileRevisionStore:
        return self._dependencies.active_store

    @property
    def revision_store(self) -> EffectiveProfileRevisionStore:
        return self._dependencies.revision_store

    def get_active_binding(
        self,
        scope: EffectiveProfileRevisionScope,
    ) -> ActiveEffectiveProfileRevisionBinding | None:
        return self._dependencies.active_store.get_active(scope)

    def resolve_active_revision(
        self,
        scope: EffectiveProfileRevisionScope,
    ) -> EffectiveProfileRevision:
        return resolve_active_effective_profile_revision(
            active_store=self._dependencies.active_store,
            revision_store=self._dependencies.revision_store,
            scope=scope,
        )

    def activate(
        self,
        request: ActivateEffectiveProfileRevisionRequest,
    ) -> EffectiveProfileActivationResult:
        candidate = self._load_and_validate_candidate(request)
        self._assert_eligible(candidate)
        current = self._dependencies.active_store.get_active(request.scope)
        current_revision_id = current.revision_id if current is not None else None
        if request.expected_active_revision_id != current_revision_id:
            raise EffectiveProfileActivationConflictError(
                "expected active revision does not match current active revision",
            )
        binding = ActiveEffectiveProfileRevisionBinding(
            scope=request.scope,
            revision_id=candidate.revision_id,
            fingerprint=candidate.fingerprint,
        )
        if current is not None and current == binding:
            return EffectiveProfileActivationResult(
                scope=request.scope,
                previous_revision_id=current.revision_id,
                active_revision_id=candidate.revision_id,
                active_fingerprint=candidate.fingerprint,
                changed=False,
            )
        cas = self._dependencies.active_store.compare_and_set_active(
            request.scope,
            expected_revision_id=request.expected_active_revision_id,
            new_binding=binding,
        )
        if cas.outcome is ActiveEffectiveProfileRevisionCasOutcome.CONFLICT:
            raise EffectiveProfileActivationConflictError(
                "active effective profile revision compare-and-set conflict",
            )
        changed = cas.outcome is ActiveEffectiveProfileRevisionCasOutcome.UPDATED
        return EffectiveProfileActivationResult(
            scope=request.scope,
            previous_revision_id=current_revision_id,
            active_revision_id=candidate.revision_id,
            active_fingerprint=candidate.fingerprint,
            changed=changed,
        )

    def rollback(
        self,
        *,
        scope: EffectiveProfileRevisionScope,
        target_revision_id: EffectiveProfileRevisionId,
        expected_active_revision_id: EffectiveProfileRevisionId,
    ) -> EffectiveProfileActivationResult:
        """Rollback is activation of an immutable historical revision."""
        return self.activate(
            ActivateEffectiveProfileRevisionRequest(
                scope=scope,
                candidate_revision_id=target_revision_id,
                expected_active_revision_id=expected_active_revision_id,
            ),
        )

    def _load_and_validate_candidate(
        self,
        request: ActivateEffectiveProfileRevisionRequest,
    ) -> EffectiveProfileRevision:
        candidate = self._dependencies.revision_store.get(
            request.candidate_revision_id,
            scope=request.scope,
        )
        if candidate is None:
            raise EffectiveProfileActivationRevisionNotFoundError(
                revision_id=request.candidate_revision_id.value,
                scope=request.scope,
            )
        if candidate.scope != request.scope:
            raise EffectiveProfileActivationScopeMismatchError(
                "candidate revision scope does not match activation scope",
            )
        return candidate

    def _assert_eligible(self, candidate: EffectiveProfileRevision) -> None:
        if self._dependencies.eligibility_checker is not None:
            try:
                self._dependencies.eligibility_checker(candidate)
            except EffectiveProfileActivationRejectedError:
                raise
            except Exception as exc:
                raise EffectiveProfileActivationRejectedError(str(exc)) from exc
            return
        if self._dependencies.skill_registry is not None:
            try:
                validate_capability_dependencies_for_environment(
                    candidate.effective_profile,
                    skill_registry=self._dependencies.skill_registry,
                )
            except Exception as exc:
                raise EffectiveProfileActivationRejectedError(str(exc)) from exc


def activate_materialized_revision(
    service: EffectiveProfileActivationService,
    *,
    scope: EffectiveProfileRevisionScope,
    candidate_revision_id: EffectiveProfileRevisionId,
) -> EffectiveProfileActivationResult:
    """Bootstrap helper: activate with expected-current read from active store."""
    current = service.get_active_binding(scope)
    expected = current.revision_id if current is not None else None
    return service.activate(
        ActivateEffectiveProfileRevisionRequest(
            scope=scope,
            candidate_revision_id=candidate_revision_id,
            expected_active_revision_id=expected,
        ),
    )
