# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision domain adapter for Platform Plugin discovery and registry composition (DS-PLUGIN).

Platform discovers entry points; Decision validates semantics and composes immutable
domain registries. Installation alone does not activate plugins — explicit composition
with ``discover_entry_points=True`` is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from intergrax.contracts.decision_artifact_registry import (
    DecisionArtifactKindAlreadyRegisteredError,
    DecisionArtifactKindRegistry,
    register_decision_artifact_kind,
)
from intergrax.contracts.decision_record import (
    DecisionArtifactKind,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_strategy import (
    DecisionStrategy,
    DecisionStrategyAlreadyRegisteredError,
    DecisionStrategyRegistration,
    DecisionStrategyRegistry,
    register_decision_strategy,
)
from intergrax.contracts.decision_verification_stage import (
    T,
    VerificationStage,
    VerificationStageAlreadyRegisteredError,
    VerificationStageRegistration,
    VerificationStageRegistry,
    register_verification_stage,
)
from intergrax.contracts.decision.integration.composition import (
    DecisionIntegrationCompositionProvider,
)
from intergrax.contracts.decision.integration.engine import (
    DecisionSystemIntegrationEngine,
)
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EP_DECISION_ARTIFACT_KINDS,
    EP_DECISION_STRATEGIES,
    EP_DECISION_VERIFICATION_STAGES,
    EntryPointLoadResult,
    EntryPointSpec,
    instantiate_entry_point_target,
    load_entry_point_targets_for_specs,
)
from intergrax.runtime.decision_plugin_policy import DecisionPluginLoadPolicy
from intergrax.runtime.decision_plugin_pre_load import (
    DecisionPluginAdmissionPlan,
    plan_decision_plugin_admission,
)

DECISION_PLUGIN_DOMAIN = "decision"
DECISION_STRATEGY_CAPABILITY_ID = "decision.strategy"
DECISION_VERIFICATION_STAGE_CAPABILITY_ID = "decision.verification_stage"
DECISION_ARTIFACT_KIND_CAPABILITY_ID = "decision.artifact_kind"


@dataclass(frozen=True, slots=True)
class DecisionArtifactKindContribution:
    """Semantic metadata contribution for one Decision Artifact kind."""

    kind: DecisionArtifactKind


@dataclass(frozen=True, slots=True)
class DecisionStrategyPluginLoadOutcome:
    registry: DecisionStrategyRegistry
    report: DomainPluginLoadReport


@dataclass(frozen=True, slots=True)
class VerificationStagePluginLoadOutcome(Generic[T]):
    registry: VerificationStageRegistry[T]
    report: DomainPluginLoadReport


@dataclass(frozen=True, slots=True)
class DecisionArtifactKindPluginLoadOutcome:
    registry: DecisionArtifactKindRegistry
    report: DomainPluginLoadReport


def _runtime_identity_mismatch_rejection(
    spec: EntryPointSpec,
    *,
    expected_plugin_id: str,
    runtime_kind: str,
) -> PluginAdmissionRejection:
    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.PLUGIN_IDENTITY_MISMATCH,
        reason=(
            f"Admitted plugin_id {expected_plugin_id!r} does not match runtime kind "
            f"{runtime_kind!r} for entry point {spec.name!r} in group {spec.group!r}."
        ),
        plugin_id=expected_plugin_id,
        fail_closed=True,
    )


def _admitted_entry_point_specs(
    plan: DecisionPluginAdmissionPlan,
) -> tuple[EntryPointSpec, ...]:
    return tuple(item.spec for item in plan.admitted)


def _build_report(
    *,
    group: str,
    accepted: list[EntryPointSpec],
    rejected: list[PluginAdmissionRejection],
    failed: list[EntryPointLoadResult],
) -> DomainPluginLoadReport:
    accepted_tuple = tuple(sorted(accepted, key=lambda spec: (spec.name, spec.value)))
    return DomainPluginLoadReport(
        group=group,
        accepted=accepted_tuple,
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))
        ),
        failed=tuple(
            sorted(failed, key=lambda item: (item.spec.name, item.spec.value))
        ),
        registered_count=len(accepted_tuple),
    )


def _resolve_strategy_registration(
    target: object,
) -> DecisionStrategyRegistration:
    resolved = instantiate_entry_point_target(target)
    if isinstance(resolved, DecisionStrategyRegistration):
        return resolved
    if isinstance(resolved, DecisionStrategy):
        return DecisionStrategyRegistration(
            kind=resolved.kind,
            strategy=resolved,
        )
    raise TypeError(
        "Decision strategy entry point must return DecisionStrategy or "
        "DecisionStrategyRegistration",
    )


def _resolve_artifact_kind_contribution(
    target: object,
) -> DecisionArtifactKind:
    resolved = instantiate_entry_point_target(target)
    if isinstance(resolved, DecisionArtifactKindContribution):
        return validate_decision_artifact_kind(resolved.kind)
    if isinstance(resolved, str):
        return validate_decision_artifact_kind(resolved)
    raise TypeError(
        "Decision artifact kind entry point must return "
        "DecisionArtifactKindContribution or DecisionArtifactKind",
    )


def load_decision_strategy_plugins(
    registry: DecisionStrategyRegistry,
    *,
    policy: DecisionPluginLoadPolicy | None = None,
    discover_entry_points: bool = False,
) -> DecisionStrategyPluginLoadOutcome:
    """Compose strategy plugins into a new immutable DecisionStrategyRegistry."""
    chosen = policy if policy is not None else DecisionPluginLoadPolicy()
    if not discover_entry_points:
        return DecisionStrategyPluginLoadOutcome(
            registry=registry,
            report=DomainPluginLoadReport.empty(EP_DECISION_STRATEGIES),
        )

    plan = plan_decision_plugin_admission(
        EP_DECISION_STRATEGIES,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_STRATEGY_CAPABILITY_ID,
        policy=chosen,
        requested_plugins=chosen.requested_strategy_plugins,
    )
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(plan.rejected)
    failed: list[EntryPointLoadResult] = []
    current = registry

    for result in load_entry_point_targets_for_specs(
        _admitted_entry_point_specs(plan),
        on_load_failure=chosen.on_load_failure,
    ):
        if result.error is not None:
            failed.append(result)
            continue

        try:
            registration = _resolve_strategy_registration(result.target)
        except (TypeError, ValueError) as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=str(exc),
                    fail_closed=True,
                )
            )
            continue
        except Exception as exc:
            if chosen.on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue

        kind_value = str(registration.kind)
        expected_plugin_id = plan.expected_plugin_id_for(result.spec)
        if expected_plugin_id is not None and kind_value != expected_plugin_id:
            rejected.append(
                _runtime_identity_mismatch_rejection(
                    result.spec,
                    expected_plugin_id=expected_plugin_id,
                    runtime_kind=kind_value,
                ),
            )
            continue

        try:
            current = register_decision_strategy(current, registration)
        except DecisionStrategyAlreadyRegisteredError as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                    reason=str(exc),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=str(exc),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
            )
            continue

        accepted.append(result.spec)

    return DecisionStrategyPluginLoadOutcome(
        registry=current,
        report=_build_report(
            group=EP_DECISION_STRATEGIES,
            accepted=accepted,
            rejected=rejected,
            failed=failed,
        ),
    )


def load_verification_stage_plugins(
    registry: VerificationStageRegistry[T],
    *,
    policy: DecisionPluginLoadPolicy | None = None,
    discover_entry_points: bool = False,
) -> VerificationStagePluginLoadOutcome[T]:
    """Compose verification stage plugins into a new immutable registry."""
    chosen = policy if policy is not None else DecisionPluginLoadPolicy()
    if not discover_entry_points:
        return VerificationStagePluginLoadOutcome(
            registry=registry,
            report=DomainPluginLoadReport.empty(EP_DECISION_VERIFICATION_STAGES),
        )

    plan = plan_decision_plugin_admission(
        EP_DECISION_VERIFICATION_STAGES,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
        policy=chosen,
        requested_plugins=chosen.requested_verification_stage_plugins,
    )
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(plan.rejected)
    failed: list[EntryPointLoadResult] = []
    current: VerificationStageRegistry[T] = registry

    def _resolve_verification_registration(
        target: object,
    ) -> VerificationStageRegistration[T]:
        resolved = instantiate_entry_point_target(target)
        if isinstance(resolved, VerificationStageRegistration):
            return VerificationStageRegistration(
                kind=resolved.kind,
                stage=resolved.stage,
                required=resolved.required,
            )
        if isinstance(resolved, VerificationStage):
            return VerificationStageRegistration(
                kind=resolved.kind,
                stage=resolved,
                required=True,
            )
        raise TypeError(
            "Decision verification stage entry point must return VerificationStage or "
            "VerificationStageRegistration",
        )

    for result in load_entry_point_targets_for_specs(
        _admitted_entry_point_specs(plan),
        on_load_failure=chosen.on_load_failure,
    ):
        if result.error is not None:
            failed.append(result)
            continue

        try:
            registration = _resolve_verification_registration(result.target)
        except (TypeError, ValueError) as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=str(exc),
                    fail_closed=True,
                )
            )
            continue
        except Exception as exc:
            if chosen.on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue

        kind_value = str(registration.kind)
        expected_plugin_id = plan.expected_plugin_id_for(result.spec)
        if expected_plugin_id is not None and kind_value != expected_plugin_id:
            rejected.append(
                _runtime_identity_mismatch_rejection(
                    result.spec,
                    expected_plugin_id=expected_plugin_id,
                    runtime_kind=kind_value,
                ),
            )
            continue

        try:
            current = register_verification_stage(current, registration)
        except VerificationStageAlreadyRegisteredError as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                    reason=str(exc),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=str(exc),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
            )
            continue

        accepted.append(result.spec)

    return VerificationStagePluginLoadOutcome(
        registry=current,
        report=_build_report(
            group=EP_DECISION_VERIFICATION_STAGES,
            accepted=accepted,
            rejected=rejected,
            failed=failed,
        ),
    )


def load_decision_artifact_kind_plugins(
    registry: DecisionArtifactKindRegistry,
    *,
    policy: DecisionPluginLoadPolicy | None = None,
    discover_entry_points: bool = False,
) -> DecisionArtifactKindPluginLoadOutcome:
    """Compose artifact kind plugins into a new immutable registry."""
    chosen = policy if policy is not None else DecisionPluginLoadPolicy()
    if not discover_entry_points:
        return DecisionArtifactKindPluginLoadOutcome(
            registry=registry,
            report=DomainPluginLoadReport.empty(EP_DECISION_ARTIFACT_KINDS),
        )

    plan = plan_decision_plugin_admission(
        EP_DECISION_ARTIFACT_KINDS,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
        policy=chosen,
        requested_plugins=chosen.requested_artifact_plugins,
    )
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(plan.rejected)
    failed: list[EntryPointLoadResult] = []
    current = registry

    for result in load_entry_point_targets_for_specs(
        _admitted_entry_point_specs(plan),
        on_load_failure=chosen.on_load_failure,
    ):
        if result.error is not None:
            failed.append(result)
            continue

        try:
            kind = _resolve_artifact_kind_contribution(result.target)
        except (TypeError, ValueError) as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=str(exc),
                    fail_closed=True,
                )
            )
            continue
        except Exception as exc:
            if chosen.on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue

        kind_value = str(kind)
        expected_plugin_id = plan.expected_plugin_id_for(result.spec)
        if expected_plugin_id is not None and kind_value != expected_plugin_id:
            rejected.append(
                _runtime_identity_mismatch_rejection(
                    result.spec,
                    expected_plugin_id=expected_plugin_id,
                    runtime_kind=kind_value,
                ),
            )
            continue

        try:
            current = register_decision_artifact_kind(current, kind)
        except DecisionArtifactKindAlreadyRegisteredError as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                    reason=str(exc),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=str(exc),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
            )
            continue

        accepted.append(result.spec)

    return DecisionArtifactKindPluginLoadOutcome(
        registry=current,
        report=_build_report(
            group=EP_DECISION_ARTIFACT_KINDS,
            accepted=accepted,
            rejected=rejected,
            failed=failed,
        ),
    )


def compose_decision_system_integration_from_platform(
    *,
    composition: DecisionIntegrationCompositionProvider | None = None,
) -> DecisionSystemIntegrationEngine:
    """Official Decision domain platform entry for Integration Boundary composition."""
    from intergrax.runtime.decision_integration_composition import (
        compose_decision_system_integration_engine,
    )

    return compose_decision_system_integration_engine(composition)
