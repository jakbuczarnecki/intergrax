# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Shared provider-neutral qualification execution coordinator (PROVIDER-QUAL-7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from intergrax.core.qualification.persistence import (
    ProviderQualificationPersistenceConflictError,
)
from intergrax.core.qualification.provider import (
    ProviderQualificationExecutor,
    ProviderQualificationRun,
    ProviderQualificationSubject,
)
from intergrax.core.qualification.requalification import ProviderRequalificationRunIdentity
from intergrax.core.qualification.suite import (
    ProviderQualificationDomainBinding,
    ProviderQualificationMaterializationHandle,
    ProviderQualificationSuite,
    ProviderQualificationSuiteIdentity,
)
from intergrax.core.qualification.validity import (
    QualificationEvidenceValidity,
    QualificationRunId,
    ValidityEvaluationId,
    new_qualification_run_id,
    validate_qualification_run_id,
)
from intergrax.core.qualification.observability import (
    NoOpProviderQualificationExecutionObservability,
    ProviderQualificationExecutionObservabilityPort,
    ProviderQualificationInfrastructurePhase,
    utc_now,
)
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationError,
)
from intergrax.integrations.registry.factory import resolve_slug
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.contracts import PlatformIntegrationContract


class ProviderQualificationExecutionError(Exception):
    """Base error for qualification infrastructure that could not complete execution."""


class ProviderQualificationResolutionError(ProviderQualificationExecutionError):
    """Provider could not be resolved from the integration profile."""


class ProviderQualificationMaterializationError(ProviderQualificationExecutionError):
    """Provider materialization failed before suite execution."""


class ProviderQualificationSuiteInfrastructureError(ProviderQualificationExecutionError):
    """Suite infrastructure failed; this is not a provider semantic rejection."""


class ProviderQualificationSubjectMismatchError(ProviderQualificationExecutionError):
    """Resolved provider facts do not match the qualification subject."""


class ProviderQualificationSuiteIdentityMismatchError(ProviderQualificationExecutionError):
    """Injected suite identity does not match the execution subject."""


class ProviderQualificationPersistenceExecutionError(ProviderQualificationExecutionError):
    """Qualification executed but durable persistence failed."""


class ProviderQualificationRunIdentityError(ProviderQualificationExecutionError):
    """Prepared or explicit qualification run identity is inconsistent."""


class ProviderQualificationExecutionConflictError(ProviderQualificationExecutionError):
    """Same qualification_run_id stores or requests incompatible qualification facts."""


class ProviderQualificationRequestIncompatibleError(ProviderQualificationExecutionConflictError):
    """Persisted qualification run is incompatible with the execution request."""


@dataclass(frozen=True, slots=True)
class ProviderQualificationExecutionCausality:
    """Optional provenance when execution follows a requalification decision."""

    prior_qualification_run_id: QualificationRunId
    new_qualification_run_id: QualificationRunId
    based_on_validity: QualificationEvidenceValidity
    basis_validity_evaluation_id: ValidityEvaluationId


@dataclass(frozen=True, slots=True)
class ProviderQualificationExecutionRequest:
    """Immutable qualification execution request (no secrets)."""

    subject: ProviderQualificationSubject
    executor: ProviderQualificationExecutor
    source_revision: str
    integration_profile: IntegrationProfile
    qualification_run_id: QualificationRunId | None = None
    requalification_identity: ProviderRequalificationRunIdentity | None = None
    causality: ProviderQualificationExecutionCausality | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ProviderQualificationSubject):
            raise TypeError("subject must be ProviderQualificationSubject")
        if not isinstance(self.executor, ProviderQualificationExecutor):
            raise TypeError("executor must be ProviderQualificationExecutor")
        if type(self.source_revision) is not str or not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if not isinstance(self.integration_profile, IntegrationProfile):
            raise TypeError("integration_profile must be IntegrationProfile")
        if self.qualification_run_id is not None:
            validate_qualification_run_id(self.qualification_run_id)
        if self.requalification_identity is not None and not isinstance(
            self.requalification_identity,
            ProviderRequalificationRunIdentity,
        ):
            raise TypeError("requalification_identity must be ProviderRequalificationRunIdentity")
        if self.causality is not None and not isinstance(
            self.causality,
            ProviderQualificationExecutionCausality,
        ):
            raise TypeError("causality must be ProviderQualificationExecutionCausality")


@runtime_checkable
class ProviderQualificationPersistencePort(Protocol):
    """Narrow persistence contract used by the shared qualification runner."""

    def persist(self, run: ProviderQualificationRun) -> ProviderQualificationRun:
        """Persist an immutable provider qualification run."""

    def get_by_qualification_run_id(
        self,
        qualification_run_id: QualificationRunId | str,
    ) -> ProviderQualificationRun | None:
        """Load a persisted run by authoritative qualification identity."""


@dataclass(frozen=True, slots=True)
class ProviderQualificationExecutionDependencies:
    """Typed wiring for synchronous qualification execution."""

    persistence: ProviderQualificationPersistencePort
    domain_binding: ProviderQualificationDomainBinding
    integration_category: IntegrationCategory = IntegrationCategory.RELATIONAL_STORE
    observability: ProviderQualificationExecutionObservabilityPort = (
        NoOpProviderQualificationExecutionObservability()
    )

    def __post_init__(self) -> None:
        if not isinstance(self.persistence, ProviderQualificationPersistencePort):
            raise TypeError("persistence must implement ProviderQualificationPersistencePort")
        if not isinstance(self.domain_binding, ProviderQualificationDomainBinding):
            raise TypeError("domain_binding must implement ProviderQualificationDomainBinding")
        if not isinstance(
            self.observability,
            ProviderQualificationExecutionObservabilityPort,
        ):
            raise TypeError(
                "observability must implement ProviderQualificationExecutionObservabilityPort",
            )


def causality_from_requalification_identity(
    identity: ProviderRequalificationRunIdentity,
) -> ProviderQualificationExecutionCausality:
    """Project requalification provenance for execution receipts."""
    decision = identity.decision
    return ProviderQualificationExecutionCausality(
        prior_qualification_run_id=identity.prior_qualification_run_id,
        new_qualification_run_id=identity.new_qualification_run_id,
        based_on_validity=decision.based_on_validity,
        basis_validity_evaluation_id=decision.basis_validity_evaluation_id,
    )


def resolve_integration_provider_id(
    profile: IntegrationProfile,
    category: IntegrationCategory,
) -> str:
    """Resolve canonical provider_id for a profile category without vendor dispatch."""
    instance = profile.instance_for_category(category)
    if instance is not None:
        if isinstance(instance, PlatformIntegrationContract):
            provider_id = instance.provider_id
            if type(provider_id) is not str or not provider_id.strip():
                raise ProviderQualificationResolutionError(
                    "pre-built integration instance did not expose provider_id",
                )
            return provider_id.strip().lower()
        raise ProviderQualificationResolutionError(
            "pre-built integration instance is not a PlatformIntegrationContract",
        )

    from_profile = profile.slug_for_category(category)
    if from_profile:
        return from_profile.strip().lower()

    try:
        return resolve_slug(category, profile=profile)
    except (IntegrationConfigurationError, IntegrationError) as exc:
        raise ProviderQualificationResolutionError(
            f"integration provider could not be resolved for category {category.value}",
        ) from exc


def _resolve_execution_run_id(request: ProviderQualificationExecutionRequest) -> QualificationRunId:
    if request.requalification_identity is not None:
        prepared = request.requalification_identity.new_qualification_run_id
        if request.qualification_run_id is not None and request.qualification_run_id != prepared:
            raise ProviderQualificationRunIdentityError(
                "qualification_run_id must match requalification prepared identity",
            )
        return prepared

    if request.qualification_run_id is not None:
        return request.qualification_run_id

    return new_qualification_run_id()


def _validate_suite_identity(
    suite: ProviderQualificationSuite,
    subject: ProviderQualificationSubject,
) -> None:
    identity = suite.identity
    if not isinstance(identity, ProviderQualificationSuiteIdentity):
        raise ProviderQualificationSuiteInfrastructureError(
            "domain suite returned invalid identity type",
        )
    mismatches: list[str] = []
    if identity.domain != subject.domain:
        mismatches.append("domain")
    if identity.capability_id != subject.capability_id:
        mismatches.append("capability_id")
    if identity.qualification_suite_id != subject.qualification_suite_id:
        mismatches.append("qualification_suite_id")
    if identity.qualification_suite_version != subject.qualification_suite_version:
        mismatches.append("qualification_suite_version")
    if mismatches:
        raise ProviderQualificationSuiteIdentityMismatchError(
            "suite identity does not match qualification subject: "
            + ", ".join(mismatches),
        )


def _validate_stored_run_compatible_with_request(
    stored: ProviderQualificationRun,
    request: ProviderQualificationExecutionRequest,
) -> None:
    if stored.qualification_run_id != _resolve_execution_run_id(request):
        raise ProviderQualificationRequestIncompatibleError(
            "stored qualification_run_id does not match execution request",
        )
    if stored.subject != request.subject:
        raise ProviderQualificationRequestIncompatibleError(
            "stored qualification subject does not match execution request",
        )
    if stored.source_revision != request.source_revision:
        raise ProviderQualificationRequestIncompatibleError(
            "stored source_revision does not match execution request",
        )
    if stored.executor != request.executor:
        raise ProviderQualificationRequestIncompatibleError(
            "stored executor does not match execution request",
        )


def _record_infrastructure_failure(
    dependencies: ProviderQualificationExecutionDependencies,
    *,
    request: ProviderQualificationExecutionRequest,
    run_id: QualificationRunId,
    phase: ProviderQualificationInfrastructurePhase,
    error: Exception,
    error_code: str,
) -> None:
    try:
        dependencies.observability.record_infrastructure_failure(
            qualification_run_id=run_id,
            subject=request.subject,
            executor=request.executor,
            source_revision=request.source_revision,
            phase=phase,
            error_type=type(error).__name__,
            error_code=error_code,
            occurred_at=utc_now(),
        )
    except Exception:
        return


def execute_provider_qualification(
    request: ProviderQualificationExecutionRequest,
    dependencies: ProviderQualificationExecutionDependencies,
) -> ProviderQualificationRun:
    """
    Synchronously execute provider qualification and return the canonical immutable run.

    Idempotency: when a persisted run already exists for the execution run id and the
    request is compatible, the same immutable historical fact is returned. Conflicting
    duplicate writes fail closed via persistence conditional semantics.
    """
    if not isinstance(request, ProviderQualificationExecutionRequest):
        raise TypeError("request must be ProviderQualificationExecutionRequest")
    if not isinstance(dependencies, ProviderQualificationExecutionDependencies):
        raise TypeError("dependencies must be ProviderQualificationExecutionDependencies")

    run_id = _resolve_execution_run_id(request)
    existing = dependencies.persistence.get_by_qualification_run_id(run_id)
    if existing is not None:
        _validate_stored_run_compatible_with_request(existing, request)
        dependencies.observability.record_execution_recovered(
            existing,
            recovery_kind="persisted_run",
            occurred_at=utc_now(),
        )
        return existing

    started_at = utc_now()
    dependencies.observability.record_execution_started(
        qualification_run_id=run_id,
        subject=request.subject,
        executor=request.executor,
        source_revision=request.source_revision,
        occurred_at=started_at,
    )

    binding = dependencies.domain_binding
    suite = binding.suite
    if not isinstance(suite, ProviderQualificationSuite):
        error = ProviderQualificationSuiteInfrastructureError(
            "domain binding did not provide a ProviderQualificationSuite",
        )
        _record_infrastructure_failure(
            dependencies,
            request=request,
            run_id=run_id,
            phase=ProviderQualificationInfrastructurePhase.SUITE,
            error=error,
            error_code="suite_contract_invalid",
        )
        raise error

    _validate_suite_identity(suite, request.subject)

    try:
        resolved_provider_id = resolve_integration_provider_id(
            request.integration_profile,
            dependencies.integration_category,
        )
    except ProviderQualificationResolutionError as exc:
        _record_infrastructure_failure(
            dependencies,
            request=request,
            run_id=run_id,
            phase=ProviderQualificationInfrastructurePhase.RESOLUTION,
            error=exc,
            error_code="provider_resolution_failed",
        )
        raise

    try:
        binding.validate_resolved_provider(
            request.subject,
            resolved_provider_id=resolved_provider_id,
        )
    except ProviderQualificationSubjectMismatchError:
        raise
    except Exception as exc:
        wrapped = ProviderQualificationResolutionError(
            "provider subject validation failed",
        )
        _record_infrastructure_failure(
            dependencies,
            request=request,
            run_id=run_id,
            phase=ProviderQualificationInfrastructurePhase.RESOLUTION,
            error=wrapped,
            error_code="provider_subject_validation_failed",
        )
        raise wrapped from exc

    capability: object | None = None
    handle: ProviderQualificationMaterializationHandle | None = None
    try:
        try:
            capability, handle = binding.materialize(
                request.integration_profile,
                resolved_provider_id=resolved_provider_id,
            )
        except ProviderQualificationMaterializationError as exc:
            _record_infrastructure_failure(
                dependencies,
                request=request,
                run_id=run_id,
                phase=ProviderQualificationInfrastructurePhase.MATERIALIZATION,
                error=exc,
                error_code="provider_materialization_failed",
            )
            raise
        except Exception as exc:
            wrapped = ProviderQualificationMaterializationError(
                "provider materialization failed",
            )
            _record_infrastructure_failure(
                dependencies,
                request=request,
                run_id=run_id,
                phase=ProviderQualificationInfrastructurePhase.MATERIALIZATION,
                error=wrapped,
                error_code="provider_materialization_failed",
            )
            raise wrapped from exc

        if not isinstance(handle, ProviderQualificationMaterializationHandle):
            error = ProviderQualificationMaterializationError(
                "domain binding returned invalid materialization handle",
            )
            _record_infrastructure_failure(
                dependencies,
                request=request,
                run_id=run_id,
                phase=ProviderQualificationInfrastructurePhase.MATERIALIZATION,
                error=error,
                error_code="materialization_handle_invalid",
            )
            raise error

        try:
            outcome = suite.execute(capability)
        except ProviderQualificationSuiteInfrastructureError as exc:
            _record_infrastructure_failure(
                dependencies,
                request=request,
                run_id=run_id,
                phase=ProviderQualificationInfrastructurePhase.SUITE,
                error=exc,
                error_code="suite_infrastructure_failed",
            )
            raise
        except Exception as exc:
            wrapped = ProviderQualificationSuiteInfrastructureError(
                "qualification suite infrastructure failed",
            )
            _record_infrastructure_failure(
                dependencies,
                request=request,
                run_id=run_id,
                phase=ProviderQualificationInfrastructurePhase.SUITE,
                error=wrapped,
                error_code="suite_infrastructure_failed",
            )
            raise wrapped from exc
    finally:
        if handle is not None:
            handle.close()

    executed_at = datetime.now(UTC)
    run = ProviderQualificationRun(
        qualification_run_id=run_id,
        subject=request.subject,
        status=outcome.status,
        executed_at=executed_at,
        executor=request.executor,
        result_summary=outcome.result_summary,
        evidence=outcome.evidence,
        reproducibility=outcome.reproducibility,
        limitations=outcome.limitations,
        source_revision=request.source_revision,
        environment_metadata=outcome.environment_metadata,
    )

    try:
        persisted = dependencies.persistence.persist(run)
    except ProviderQualificationPersistenceConflictError as exc:
        stored = dependencies.persistence.get_by_qualification_run_id(run_id)
        if stored is None:
            persistence_error = ProviderQualificationPersistenceExecutionError(
                "provider qualification persistence conflict could not be resolved",
            )
            _record_infrastructure_failure(
                dependencies,
                request=request,
                run_id=run_id,
                phase=ProviderQualificationInfrastructurePhase.PERSISTENCE,
                error=persistence_error,
                error_code="persistence_conflict_unresolved",
            )
            raise persistence_error from exc
        if stored == run:
            dependencies.observability.record_execution_recovered(
                stored,
                recovery_kind="persist_conflict_duplicate",
                occurred_at=utc_now(),
            )
            return stored
        conflict_error = ProviderQualificationExecutionConflictError(
            "conflicting provider qualification run for qualification_run_id",
        )
        _record_infrastructure_failure(
            dependencies,
            request=request,
            run_id=run_id,
            phase=ProviderQualificationInfrastructurePhase.PERSISTENCE,
            error=conflict_error,
            error_code="persistence_conflict",
        )
        raise conflict_error from exc
    except Exception as exc:
        persistence_error = ProviderQualificationPersistenceExecutionError(
            "provider qualification persistence failed after execution",
        )
        _record_infrastructure_failure(
            dependencies,
            request=request,
            run_id=run_id,
            phase=ProviderQualificationInfrastructurePhase.PERSISTENCE,
            error=persistence_error,
            error_code="persistence_failed",
        )
        raise persistence_error from exc
    else:
        dependencies.observability.record_execution_completed(
            persisted,
            occurred_at=utc_now(),
        )
        return persisted
