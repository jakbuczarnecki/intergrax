# © Artur Czarnecki. All rights reserved.

"""Memory provider qualification runner (MEM-ENT-13)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.contracts.long_horizon_memory import LongHorizonMemoryStore
from intergrax.memory.contracts.procedural_memory import ProcedureMemoryStore
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticComponent,
    MemoryDiagnosticCounts,
    MemoryDiagnosticEvent,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
    MemoryDiagnosticPhase,
)
from intergrax.memory.contracts.provider_qualification import (
    EntityTemporalMemoryStoreQualificationCheck,
    LongHorizonMemoryStoreQualificationCheck,
    MemoryProviderCapabilityKind,
    MemoryProviderCapabilityQualification,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderDescriptor,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationResult,
    MemoryProviderQualificationStatus,
    ProcedureMemoryStoreQualificationCheck,
    UserProfileStoreQualificationCheck,
    validate_memory_provider_descriptor,
    validate_memory_provider_qualification_request,
)
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.provider_qualification.bindings import MemoryProviderCapabilityFactories
from intergrax.memory.provider_qualification.checks import (
    default_entity_temporal_checks,
    default_long_horizon_checks,
    default_procedure_checks,
    default_user_profile_checks,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.provider_qualification.factory import MemoryProviderInstanceFactory
from intergrax.memory.user_profile_store import UserProfileStore
__all__ = ["MemoryProviderQualificationRunner"]

T = TypeVar("T")


def _sorted_capabilities(
    capabilities: tuple[MemoryProviderCapabilityKind, ...],
) -> tuple[MemoryProviderCapabilityKind, ...]:
    return tuple(sorted(capabilities, key=lambda item: item.value))


def _merge_checks_by_id(
    defaults: tuple[T, ...],
    extra: tuple[T, ...],
    *,
    id_of: Callable[[T], str],
) -> tuple[T, ...]:
    by_id: dict[str, T] = {id_of(item): item for item in defaults}
    for item in extra:
        by_id[id_of(item)] = item
    return tuple(by_id[key] for key in sorted(by_id))


def _capability_status(
    *,
    capability: MemoryProviderCapabilityKind,
    check_results: tuple[MemoryProviderCheckResult, ...],
    materialization_blocked: bool,
    factory_missing: bool,
    descriptor_unsupported: bool,
    required_capability: bool,
) -> MemoryProviderCapabilityQualification:
    if descriptor_unsupported or factory_missing:
        status = (
            MemoryProviderQualificationStatus.NOT_QUALIFIED
            if required_capability
            else MemoryProviderQualificationStatus.NOT_SUPPORTED
        )
        return MemoryProviderCapabilityQualification(
            capability=capability,
            status=status,
            checks_executed=0,
            checks_passed=0,
            checks_failed=0,
            check_results=(),
            reason_codes=(MemoryProviderQualificationFailureReason.UNSUPPORTED_CAPABILITY,),
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
    if materialization_blocked:
        status = MemoryProviderQualificationStatus.BLOCKED
    elif not check_results and required_capability:
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


async def _qualify_typed_capability(
    *,
    capability: MemoryProviderCapabilityKind,
    factory: MemoryProviderInstanceFactory[T],
    context: MemoryProviderQualificationContext,
    run_checks: Callable[[T], Awaitable[list[MemoryProviderCheckResult]]],
) -> tuple[tuple[MemoryProviderCheckResult, ...], bool]:
    """Materialize, run checks, dispose. Returns (check_results, materialization_blocked)."""
    instance: T | None = None
    materialization_blocked = False
    check_results: list[MemoryProviderCheckResult] = []

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
        check_results.extend(await run_checks(instance))

    if instance is not None:
        try:
            await factory.dispose(instance)
        except Exception as exc:
            check_results.append(
                MemoryProviderCheckResult(
                    check_id="materialization.dispose",
                    capability=capability,
                    severity=MemoryProviderCheckSeverity.REQUIRED,
                    passed=False,
                    reason_code=MemoryProviderQualificationFailureReason.CLEANUP_FAILURE,
                    detail=type(exc).__name__,
                )
            )

    ordered = tuple(sorted(check_results, key=lambda item: item.check_id))
    return ordered, materialization_blocked


@dataclass(slots=True)
class MemoryProviderQualificationRunner:
    extra_user_profile_checks: tuple[UserProfileStoreQualificationCheck, ...] = ()
    extra_entity_temporal_checks: tuple[EntityTemporalMemoryStoreQualificationCheck, ...] = ()
    extra_procedure_checks: tuple[ProcedureMemoryStoreQualificationCheck, ...] = ()
    extra_long_horizon_checks: tuple[LongHorizonMemoryStoreQualificationCheck, ...] = ()
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
        validate_memory_provider_descriptor(descriptor)
        validate_memory_provider_qualification_request(request)

        capabilities = _sorted_capabilities(
            request.required_capabilities + request.optional_capabilities
        )
        required_set = set(request.required_capabilities)
        declared = set(descriptor.capabilities)
        capability_results: list[MemoryProviderCapabilityQualification] = []

        for capability in capabilities:
            required_capability = capability in required_set

            if capability not in declared:
                capability_results.append(
                    _capability_status(
                        capability=capability,
                        check_results=(),
                        materialization_blocked=False,
                        factory_missing=False,
                        descriptor_unsupported=True,
                        required_capability=required_capability,
                    )
                )
                self._emit_capability_terminal(
                    descriptor=descriptor,
                    context=context,
                    capability_result=capability_results[-1],
                )
                continue

            cap_result = await self._qualify_capability(
                capability=capability,
                factories=factories,
                context=context,
                required_capability=required_capability,
            )
            if cap_result is None:
                capability_results.append(
                    _capability_status(
                        capability=capability,
                        check_results=(),
                        materialization_blocked=False,
                        factory_missing=True,
                        descriptor_unsupported=False,
                        required_capability=required_capability,
                    )
                )
            else:
                capability_results.append(cap_result)

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

    async def _qualify_capability(
        self,
        *,
        capability: MemoryProviderCapabilityKind,
        factories: MemoryProviderCapabilityFactories,
        context: MemoryProviderQualificationContext,
        required_capability: bool,
    ) -> MemoryProviderCapabilityQualification | None:
        if capability is MemoryProviderCapabilityKind.USER_PROFILE_STORE:
            factory = factories.user_profile_store
            if factory is None:
                return None
            checks = _merge_checks_by_id(
                default_user_profile_checks(),
                self.extra_user_profile_checks,
                id_of=lambda c: c.check_id,
            )

            async def _run_user_profile(inst: UserProfileStore) -> list[MemoryProviderCheckResult]:
                return [await check.run(inst, context) for check in checks]

            check_results, materialization_blocked = await _qualify_typed_capability(
                capability=capability,
                factory=factory,
                context=context,
                run_checks=_run_user_profile,
            )
            return _capability_status(
                capability=capability,
                check_results=check_results,
                materialization_blocked=materialization_blocked,
                factory_missing=False,
                descriptor_unsupported=False,
                required_capability=required_capability,
            )

        if capability is MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE:
            factory = factories.entity_temporal_memory_store
            if factory is None:
                return None
            checks = _merge_checks_by_id(
                default_entity_temporal_checks(),
                self.extra_entity_temporal_checks,
                id_of=lambda c: c.check_id,
            )

            async def _run_entity(
                inst: EntityTemporalMemoryStore,
            ) -> list[MemoryProviderCheckResult]:
                return [await check.run(inst, context) for check in checks]

            check_results, materialization_blocked = await _qualify_typed_capability(
                capability=capability,
                factory=factory,
                context=context,
                run_checks=_run_entity,
            )
            return _capability_status(
                capability=capability,
                check_results=check_results,
                materialization_blocked=materialization_blocked,
                factory_missing=False,
                descriptor_unsupported=False,
                required_capability=required_capability,
            )

        if capability is MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE:
            factory = factories.procedure_memory_store
            if factory is None:
                return None
            checks = _merge_checks_by_id(
                default_procedure_checks(),
                self.extra_procedure_checks,
                id_of=lambda c: c.check_id,
            )

            async def _run_procedure(
                inst: ProcedureMemoryStore,
            ) -> list[MemoryProviderCheckResult]:
                return [await check.run(inst, context) for check in checks]

            check_results, materialization_blocked = await _qualify_typed_capability(
                capability=capability,
                factory=factory,
                context=context,
                run_checks=_run_procedure,
            )
            return _capability_status(
                capability=capability,
                check_results=check_results,
                materialization_blocked=materialization_blocked,
                factory_missing=False,
                descriptor_unsupported=False,
                required_capability=required_capability,
            )

        if capability is MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE:
            factory = factories.long_horizon_memory_store
            if factory is None:
                return None
            checks = _merge_checks_by_id(
                default_long_horizon_checks(),
                self.extra_long_horizon_checks,
                id_of=lambda c: c.check_id,
            )

            async def _run_long_horizon(
                inst: LongHorizonMemoryStore,
            ) -> list[MemoryProviderCheckResult]:
                return [await check.run(inst, context) for check in checks]

            check_results, materialization_blocked = await _qualify_typed_capability(
                capability=capability,
                factory=factory,
                context=context,
                run_checks=_run_long_horizon,
            )
            return _capability_status(
                capability=capability,
                check_results=check_results,
                materialization_blocked=materialization_blocked,
                factory_missing=False,
                descriptor_unsupported=False,
                required_capability=required_capability,
            )

        if capability is MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE:
            factory = factories.session_turn_index_store
            if factory is None:
                return None

            async def _run_session_turn_index(
                _inst: SessionTurnIndexStore,
            ) -> list[MemoryProviderCheckResult]:
                return []

            check_results, materialization_blocked = await _qualify_typed_capability(
                capability=capability,
                factory=factory,
                context=context,
                run_checks=_run_session_turn_index,
            )
            return _capability_status(
                capability=capability,
                check_results=check_results,
                materialization_blocked=materialization_blocked,
                factory_missing=False,
                descriptor_unsupported=False,
                required_capability=required_capability,
            )

        return None

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
