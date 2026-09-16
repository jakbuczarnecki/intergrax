# © Artur Czarnecki. All rights reserved.

"""Memory provider qualification runner (MEM-ENT-13)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticComponent,
    MemoryDiagnosticCounts,
    MemoryDiagnosticEvent,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
    MemoryDiagnosticPhase,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCapabilityQualification,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderDescriptor,
    MemoryProviderQualificationCheck,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationResult,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.provider_qualification.bindings import MemoryProviderCapabilityFactories
from intergrax.memory.provider_qualification.checks import default_checks_for_capability
from intergrax.memory.provider_qualification.factory import MemoryProviderInstanceFactory

__all__ = ["MemoryProviderQualificationRunner"]


def _sorted_capabilities(
    capabilities: tuple[MemoryProviderCapabilityKind, ...],
) -> tuple[MemoryProviderCapabilityKind, ...]:
    return tuple(sorted(capabilities, key=lambda item: item.value))


def _factory_for_capability(
    factories: MemoryProviderCapabilityFactories,
    capability: MemoryProviderCapabilityKind,
) -> MemoryProviderInstanceFactory[object] | None:
    if capability is MemoryProviderCapabilityKind.USER_PROFILE_STORE:
        return factories.user_profile_store
    if capability is MemoryProviderCapabilityKind.SESSION_STORAGE:
        return factories.session_storage
    if capability is MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE:
        return factories.session_turn_index_store
    if capability is MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE:
        return factories.entity_temporal_memory_store
    if capability is MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE:
        return factories.procedure_memory_store
    if capability is MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE:
        return factories.long_horizon_memory_store
    return None


def _merge_checks(
    defaults: tuple[MemoryProviderQualificationCheck, ...],
    extra: tuple[MemoryProviderQualificationCheck, ...],
) -> tuple[MemoryProviderQualificationCheck, ...]:
    by_id: dict[str, MemoryProviderQualificationCheck] = {
        check.check_id: check for check in defaults
    }
    for check in extra:
        by_id[check.check_id] = check
    return tuple(by_id[key] for key in sorted(by_id))


def _capability_status(
    *,
    capability: MemoryProviderCapabilityKind,
    check_results: tuple[MemoryProviderCheckResult, ...],
    materialization_blocked: bool,
    factory_missing: bool,
    required_capability: bool,
) -> MemoryProviderCapabilityQualification:
    if materialization_blocked:
        return MemoryProviderCapabilityQualification(
            capability=capability,
            status=MemoryProviderQualificationStatus.BLOCKED,
            checks_executed=0,
            checks_passed=0,
            checks_failed=0,
            check_results=(),
            reason_codes=(MemoryProviderQualificationFailureReason.MATERIALIZATION_FAILURE,),
        )
    if factory_missing:
        status = (
            MemoryProviderQualificationStatus.NOT_QUALIFIED
            if required_capability
            else MemoryProviderQualificationStatus.NOT_SUPPORTED
        )
        reason = (
            MemoryProviderQualificationFailureReason.UNSUPPORTED_CAPABILITY
            if not required_capability
            else MemoryProviderQualificationFailureReason.UNSUPPORTED_CAPABILITY
        )
        return MemoryProviderCapabilityQualification(
            capability=capability,
            status=status,
            checks_executed=0,
            checks_passed=0,
            checks_failed=0,
            check_results=(),
            reason_codes=(reason,),
        )
    checks_executed = len(check_results)
    checks_passed = sum(1 for item in check_results if item.passed)
    checks_failed = checks_executed - checks_passed
    reason_codes = tuple(
        sorted(
            {
                item.reason_code
                for item in check_results
                if not item.passed and item.reason_code is not None
            },
            key=lambda code: code.value,
        )
    )
    required_failures = any(
        not item.passed and item.severity is MemoryProviderCheckSeverity.REQUIRED
        for item in check_results
    )
    if not check_results and required_capability:
        status = MemoryProviderQualificationStatus.NOT_SUPPORTED
    elif required_failures:
        status = MemoryProviderQualificationStatus.NOT_QUALIFIED
    else:
        status = MemoryProviderQualificationStatus.QUALIFIED
    return MemoryProviderCapabilityQualification(
        capability=capability,
        status=status,
        checks_executed=checks_executed,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        check_results=check_results,
        reason_codes=reason_codes,
    )


def _aggregate_status(
    capability_results: tuple[MemoryProviderCapabilityQualification, ...],
    request: MemoryProviderQualificationRequest,
) -> MemoryProviderQualificationStatus:
    required = set(request.required_capabilities)
    if any(
        item.capability in required
        and item.status is MemoryProviderQualificationStatus.BLOCKED
        for item in capability_results
    ):
        return MemoryProviderQualificationStatus.BLOCKED
    if any(
        item.capability in required
        and item.status
        in {
            MemoryProviderQualificationStatus.NOT_QUALIFIED,
            MemoryProviderQualificationStatus.NOT_SUPPORTED,
        }
        for item in capability_results
    ):
        return MemoryProviderQualificationStatus.NOT_QUALIFIED
    if any(
        item.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
        for item in capability_results
        if item.capability in required
    ):
        return MemoryProviderQualificationStatus.NOT_QUALIFIED
    qualified_required = all(
        item.status is MemoryProviderQualificationStatus.QUALIFIED
        for item in capability_results
        if item.capability in required
    )
    if not qualified_required:
        return MemoryProviderQualificationStatus.NOT_QUALIFIED
    return MemoryProviderQualificationStatus.QUALIFIED


@dataclass(slots=True)
class MemoryProviderQualificationRunner:
    extra_checks: tuple[MemoryProviderQualificationCheck, ...] = ()
    observability: MemoryDiagnosticEmitter = field(
        default_factory=default_memory_diagnostic_emitter
    )

    async def qualify(
        self,
        *,
        descriptor: MemoryProviderDescriptor,
        context: MemoryProviderQualificationContext,
        request: MemoryProviderQualificationRequest,
        factories: MemoryProviderCapabilityFactories,
    ) -> MemoryProviderQualificationResult:
        capabilities = _sorted_capabilities(
            request.required_capabilities + request.optional_capabilities
        )
        required_set = set(request.required_capabilities)
        capability_results: list[MemoryProviderCapabilityQualification] = []

        for capability in capabilities:
            factory = _factory_for_capability(factories, capability)
            required_capability = capability in required_set
            if factory is None:
                capability_results.append(
                    _capability_status(
                        capability=capability,
                        check_results=(),
                        materialization_blocked=False,
                        factory_missing=True,
                        required_capability=required_capability,
                    )
                )
                continue

            instance: object | None = None
            materialization_blocked = False
            check_results: list[MemoryProviderCheckResult] = []
            cleanup_failure: MemoryProviderCheckResult | None = None

            try:
                instance = await factory.create()
            except Exception as exc:
                materialization_blocked = True
                check_results.append(
                    MemoryProviderCheckResult(
                        check_id="materialization.create",
                        capability=capability,
                        severity=MemoryProviderCheckSeverity.REQUIRED,
                        passed=False,
                        reason_code=MemoryProviderQualificationFailureReason.MATERIALIZATION_FAILURE,
                        detail=type(exc).__name__,
                    )
                )
            else:
                checks = _merge_checks(
                    default_checks_for_capability(capability),
                    tuple(
                        check
                        for check in self.extra_checks
                        if check.capability is capability
                    ),
                )
                for check in checks:
                    result = await check.run(instance, context)
                    check_results.append(result)

            if instance is not None:
                try:
                    await factory.dispose(instance)
                except Exception as exc:
                    cleanup_failure = MemoryProviderCheckResult(
                        check_id="materialization.dispose",
                        capability=capability,
                        severity=MemoryProviderCheckSeverity.REQUIRED,
                        passed=False,
                        reason_code=MemoryProviderQualificationFailureReason.CLEANUP_FAILURE,
                        detail=type(exc).__name__,
                    )
                    check_results.append(cleanup_failure)

            if materialization_blocked:
                capability_results.append(
                    _capability_status(
                        capability=capability,
                        check_results=tuple(check_results),
                        materialization_blocked=True,
                        factory_missing=False,
                        required_capability=required_capability,
                    )
                )
            else:
                capability_results.append(
                    _capability_status(
                        capability=capability,
                        check_results=tuple(
                            sorted(check_results, key=lambda item: item.check_id)
                        ),
                        materialization_blocked=False,
                        factory_missing=False,
                        required_capability=required_capability,
                    )
                )

            self._emit_capability_terminal(
                descriptor=descriptor,
                context=context,
                capability_result=capability_results[-1],
            )

        ordered_results = tuple(
            sorted(capability_results, key=lambda item: item.capability.value)
        )
        reason_codes = tuple(
            sorted(
                {code for item in ordered_results for code in item.reason_codes},
                key=lambda code: code.value,
            )
        )
        status = _aggregate_status(ordered_results, request)
        return MemoryProviderQualificationResult(
            descriptor=descriptor,
            qualification_run_id=context.qualification_run_id,
            reference_time_iso=context.reference_time_iso,
            status=status,
            capability_results=ordered_results,
            reason_codes=reason_codes,
        )

    def _emit_capability_terminal(
        self,
        *,
        descriptor: MemoryProviderDescriptor,
        context: MemoryProviderQualificationContext,
        capability_result: MemoryProviderCapabilityQualification,
    ) -> None:
        outcome = MemoryDiagnosticOutcome.SUCCESS
        if capability_result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED:
            outcome = MemoryDiagnosticOutcome.FAILED
        elif capability_result.status is MemoryProviderQualificationStatus.BLOCKED:
            outcome = MemoryDiagnosticOutcome.FAILED
        elif capability_result.status is MemoryProviderQualificationStatus.NOT_SUPPORTED:
            outcome = MemoryDiagnosticOutcome.UNSUPPORTED

        event = MemoryDiagnosticEvent(
            event_id=self.observability.new_event_id(),
            reference_time_iso=context.reference_time_iso,
            phase=MemoryDiagnosticPhase.TERMINAL,
            component=MemoryDiagnosticComponent.LIFECYCLE,
            operation=MemoryDiagnosticOperation.PROVIDER_QUALIFICATION,
            outcome=outcome,
            provider_id=descriptor.provider_id,
            projection_id=capability_result.capability.value,
            counts=MemoryDiagnosticCounts(
                sources_requested=capability_result.checks_executed,
                sources_resolved=capability_result.checks_passed,
                failures=capability_result.checks_failed,
            ),
        )
        self.observability.emit(event)
