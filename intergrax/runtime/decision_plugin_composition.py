# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision domain adapter for Platform Plugin discovery and registry composition (DS-PLUGIN).

Platform discovers entry points; Decision validates semantics and composes immutable
domain registries. Installation alone does not activate plugins — explicit composition
with ``discover_entry_points=True`` is required.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
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
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    ConflictPolicy,
    EP_DECISION_ARTIFACT_KINDS,
    EP_DECISION_STRATEGIES,
    EP_DECISION_VERIFICATION_STAGES,
    EntryPointLoadResult,
    EntryPointSpec,
    LoadIsolation,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_targets,
)
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationResult,
    evaluate_external_package_entry_point_production_admission,
    resolve_host_platform_version,
)

DECISION_PLUGIN_DOMAIN = "decision"
DECISION_STRATEGY_CAPABILITY_ID = "decision.strategy"
DECISION_VERIFICATION_STAGE_CAPABILITY_ID = "decision.verification_stage"
DECISION_ARTIFACT_KIND_CAPABILITY_ID = "decision.artifact_kind"


class ManifestCapabilityBindingDisposition(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ManifestCapabilityBindingResult:
    disposition: ManifestCapabilityBindingDisposition
    rejection: PluginAdmissionRejection | None


@dataclass(frozen=True, slots=True)
class DecisionPluginLoadPolicy:
    """Shared Decision plugin load governance for all capability groups."""

    ep_name_conflict: ConflictPolicy = "error"
    on_load_failure: LoadIsolation = "isolate"
    require_production_admission: bool = False
    require_manifest_capability_binding: bool = False
    package_qualification_lookup: (
        Callable[[EntryPointSpec], PluginQualificationResult | None] | None
    ) = None
    platform_version: str | None = None
    allowed_strategy_kinds: frozenset[str] | None = None
    allowed_verification_stage_kinds: frozenset[str] | None = None
    allowed_artifact_kinds: frozenset[str] | None = None


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


def _production_admission_rejections(
    group: str,
    policy: DecisionPluginLoadPolicy,
) -> tuple[frozenset[str], list[PluginAdmissionRejection]]:
    if not policy.require_production_admission:
        return frozenset(), []

    platform_version = policy.platform_version or resolve_host_platform_version()
    lookup = policy.package_qualification_lookup
    rejected_names: set[str] = set()
    rejected: list[PluginAdmissionRejection] = []

    for spec in iter_entry_point_specs(group):
        qualification = lookup(spec) if lookup is not None else None
        admission = evaluate_external_package_entry_point_production_admission(
            spec,
            qualification,
            platform_version=platform_version,
        )
        if admission.admitted:
            continue
        rejected_names.add(spec.name)
        rejected.append(
            PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED,
                reason=admission.reason,
                fail_closed=True,
            )
        )

    return frozenset(rejected_names), rejected


def _manifest_binding_rejected(
    spec: EntryPointSpec,
    *,
    reason_code: PluginAdmissionReasonCode,
    reason: str,
) -> ManifestCapabilityBindingResult:
    return ManifestCapabilityBindingResult(
        disposition=ManifestCapabilityBindingDisposition.REJECTED,
        rejection=PluginAdmissionRejection(
            spec=spec,
            reason_code=reason_code,
            reason=reason,
            fail_closed=True,
        ),
    )


def _manifest_binding_valid() -> ManifestCapabilityBindingResult:
    return ManifestCapabilityBindingResult(
        disposition=ManifestCapabilityBindingDisposition.VALID,
        rejection=None,
    )


def _validate_manifest_capability_binding(
    spec: EntryPointSpec,
    *,
    domain: str,
    capability_id: str,
) -> ManifestCapabilityBindingResult:
    if spec.distribution is None:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.UNRESOLVED_PACKAGE_IDENTITY,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: entry-point distribution identity "
                "is missing."
            ),
        )

    from importlib.metadata import PackageNotFoundError, distribution

    from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
    from intergrax.core.plugins.manifest_io import parse_platform_plugin_pyproject_toml

    try:
        installed = distribution(spec.distribution)
    except PackageNotFoundError:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: distribution "
                f"{spec.distribution!r} is not installed or resolvable."
            ),
        )

    if installed.files is None:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: distribution "
                f"{spec.distribution!r} has no inspectable file metadata."
            ),
        )

    manifest = None
    try:
        source = installed.read_text("pyproject.toml")
    except (FileNotFoundError, OSError, TypeError) as exc:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: Platform Plugin manifest for "
                f"distribution {spec.distribution!r} is unavailable ({type(exc).__name__})."
            ),
        )

    if source is None:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: Platform Plugin manifest for "
                f"distribution {spec.distribution!r} is unavailable."
            ),
        )

    try:
        manifest = parse_platform_plugin_pyproject_toml(source)
    except PlatformPluginManifestValidationError:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_INVALID,
            reason=(
                f"Platform plugin manifest for distribution {spec.distribution!r} "
                f"is invalid or incomplete for entry point {spec.name!r} in group "
                f"{spec.group!r}."
            ),
        )

    if not manifest.capabilities:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_CAPABILITY_BINDING_MISSING,
            reason=(
                f"Platform plugin manifest for distribution {spec.distribution!r} "
                f"declares no capabilities for required binding "
                f"{capability_id!r} on entry point {spec.name!r} in group "
                f"{spec.group!r}."
            ),
        )

    for descriptor in manifest.capabilities:
        if (
            descriptor.domain == domain
            and descriptor.entry_point_group == spec.group
            and descriptor.entry_point_name == spec.name
        ):
            if capability_id not in descriptor.capability_ids:
                return _manifest_binding_rejected(
                    spec,
                    reason_code=PluginAdmissionReasonCode.CAPABILITY_ID_MISMATCH,
                    reason=(
                        f"Manifest capability_ids for entry point {spec.name!r} in group "
                        f"{spec.group!r} do not declare required capability "
                        f"{capability_id!r}."
                    ),
                )
            return _manifest_binding_valid()

    return _manifest_binding_rejected(
        spec,
        reason_code=PluginAdmissionReasonCode.MANIFEST_CAPABILITY_BINDING_MISSING,
        reason=(
            f"Entry point {spec.name!r} in group {spec.group!r} is not declared "
            f"in the installed package Platform Plugin manifest capabilities for "
            f"required capability {capability_id!r}."
        ),
    )


def _decision_plugin_pre_admission_rejections(
    group: str,
    *,
    required_capability_id: str,
    policy: DecisionPluginLoadPolicy,
) -> tuple[frozenset[str], list[PluginAdmissionRejection]]:
    skip_names: set[str] = set()
    rejected: list[PluginAdmissionRejection] = []

    production_skip, production_rejected = _production_admission_rejections(
        group,
        policy,
    )
    skip_names.update(production_skip)
    rejected.extend(production_rejected)

    if not policy.require_manifest_capability_binding:
        return frozenset(skip_names), rejected

    for spec in iter_entry_point_specs(group):
        if spec.name in skip_names:
            continue
        binding = _validate_manifest_capability_binding(
            spec,
            domain=DECISION_PLUGIN_DOMAIN,
            capability_id=required_capability_id,
        )
        if binding.disposition is ManifestCapabilityBindingDisposition.REJECTED:
            if binding.rejection is None:
                raise RuntimeError("manifest binding rejection missing structured evidence")
            skip_names.add(spec.name)
            rejected.append(binding.rejection)

    return frozenset(skip_names), rejected


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

    skip_names, pre_rejected = _decision_plugin_pre_admission_rejections(
        EP_DECISION_STRATEGIES,
        required_capability_id=DECISION_STRATEGY_CAPABILITY_ID,
        policy=chosen,
    )
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(pre_rejected)
    failed: list[EntryPointLoadResult] = []
    current = registry

    for result in load_entry_point_targets(
        EP_DECISION_STRATEGIES,
        on_conflict=chosen.ep_name_conflict,
        on_load_failure=chosen.on_load_failure,
        skip_names=skip_names,
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
        if (
            chosen.allowed_strategy_kinds is not None
            and kind_value not in chosen.allowed_strategy_kinds
        ):
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.NOT_IN_ALLOWLIST,
                    reason=(
                        f"DecisionStrategyKind {kind_value!r} is not in configured "
                        "allowlist."
                    ),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
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

    skip_names, pre_rejected = _decision_plugin_pre_admission_rejections(
        EP_DECISION_VERIFICATION_STAGES,
        required_capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
        policy=chosen,
    )
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(pre_rejected)
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

    for result in load_entry_point_targets(
        EP_DECISION_VERIFICATION_STAGES,
        on_conflict=chosen.ep_name_conflict,
        on_load_failure=chosen.on_load_failure,
        skip_names=skip_names,
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
        if (
            chosen.allowed_verification_stage_kinds is not None
            and kind_value not in chosen.allowed_verification_stage_kinds
        ):
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.NOT_IN_ALLOWLIST,
                    reason=(
                        f"VerificationStageKind {kind_value!r} is not in configured "
                        "allowlist."
                    ),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
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

    skip_names, pre_rejected = _decision_plugin_pre_admission_rejections(
        EP_DECISION_ARTIFACT_KINDS,
        required_capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
        policy=chosen,
    )
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(pre_rejected)
    failed: list[EntryPointLoadResult] = []
    current = registry

    for result in load_entry_point_targets(
        EP_DECISION_ARTIFACT_KINDS,
        on_conflict=chosen.ep_name_conflict,
        on_load_failure=chosen.on_load_failure,
        skip_names=skip_names,
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
        if (
            chosen.allowed_artifact_kinds is not None
            and kind_value not in chosen.allowed_artifact_kinds
        ):
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.NOT_IN_ALLOWLIST,
                    reason=(
                        f"DecisionArtifactKind {kind_value!r} is not in configured "
                        "allowlist."
                    ),
                    plugin_id=kind_value,
                    fail_closed=True,
                )
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
