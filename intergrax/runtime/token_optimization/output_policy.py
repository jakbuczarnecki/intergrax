# © Artur Czarnecki. All rights reserved.

"""Output policy runtime resolver (Phase TOKEN-2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    OutputPolicy,
    OutputProfile,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenCategory,
)

_PROFILE_TO_OUTPUT: dict[TokenOptimizationProfile, OutputProfile] = {
    TokenOptimizationProfile.OFF: OutputProfile.STANDARD,
    TokenOptimizationProfile.MEASURE_ONLY: OutputProfile.STANDARD,
    TokenOptimizationProfile.CONSERVATIVE: OutputProfile.TERSE,
    TokenOptimizationProfile.BALANCED: OutputProfile.STANDARD,
    TokenOptimizationProfile.AGGRESSIVE: OutputProfile.MINIMAL,
    TokenOptimizationProfile.EXPERIMENTAL: OutputProfile.MINIMAL,
}

_SAFE_OUTPUT_PROFILES: frozenset[OutputProfile] = frozenset(
    {
        OutputProfile.STANDARD,
        OutputProfile.FULL,
        OutputProfile.AUDIT,
        OutputProfile.MACHINE_RECEIPT,
        OutputProfile.DEBUG_VERBOSE,
    }
)

_AGGRESSIVE_COMPRESSION_LEVELS: frozenset[CompressionLevel] = frozenset(
    {
        CompressionLevel.HIGH,
        CompressionLevel.SEMANTIC,
    }
)

_CONTENT_OPTIMIZATION_SOURCE_TYPES: frozenset[TokenOptimizationSourceType] = frozenset(
    {
        TokenOptimizationSourceType.TOOL_OUTPUT,
        TokenOptimizationSourceType.LOG_OUTPUT,
        TokenOptimizationSourceType.TERMINAL_OUTPUT,
        TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
        TokenOptimizationSourceType.TOOL_CATALOG,
        TokenOptimizationSourceType.MEMORY,
        TokenOptimizationSourceType.STRUCTURED_DATA,
    }
)


class OutputPolicyResolutionStatus(StrEnum):
    """Outcome status for output policy resolution."""

    RESOLVED = "resolved"
    DEFAULTED = "defaulted"
    DISABLED = "disabled"


class OutputPolicyResolutionReason(StrEnum):
    """Deterministic reason for the resolved output policy."""

    NO_POLICY = "no_policy"
    TOKEN_OPTIMIZATION_DISABLED = "token_optimization_disabled"
    EXPLICIT_OUTPUT_POLICY = "explicit_output_policy"
    DERIVED_FROM_TOKEN_POLICY = "derived_from_token_policy"
    CONTEXT_REQUESTED_PROFILE = "context_requested_profile"
    SOURCE_TYPE_CONSTRAINT = "source_type_constraint"


class OutputPolicyResolutionValidationStatus(StrEnum):
    """Outcome of resolved output policy validation."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OutputPolicyResolutionContext:
    """Typed resolver input; not a runtime hook."""

    source_type: TokenOptimizationSourceType | None = None
    token_category: TokenCategory | None = None
    runtime_profile: str | None = None
    requested_output_profile: OutputProfile | None = None
    model: str | None = None
    provider: str | None = None
    tenant_id: str | None = None
    agent_id: str | None = None
    workflow_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedOutputPolicy:
    """Deterministic resolved output shaping decision."""

    status: OutputPolicyResolutionStatus
    reason: OutputPolicyResolutionReason
    enabled: bool
    profile: OutputProfile
    compression_level: CompressionLevel
    allow_lossy: bool
    require_validation: bool
    fallback_on_validation_failure: bool
    emit_receipts: bool
    emit_observability: bool
    max_output_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputPolicyResolutionValidationResult:
    """Validation outcome for a resolved output policy."""

    status: OutputPolicyResolutionValidationStatus
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputPolicyResolver:
    """Deterministic runtime output policy resolver."""

    def resolve(
        self,
        *,
        token_policy: TokenOptimizationPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        context: OutputPolicyResolutionContext | None = None,
    ) -> ResolvedOutputPolicy:
        return resolve_output_policy(
            token_policy=token_policy,
            output_policy=output_policy,
            context=context,
        )


def resolve_output_policy(
    *,
    token_policy: TokenOptimizationPolicy | None = None,
    output_policy: OutputPolicy | None = None,
    context: OutputPolicyResolutionContext | None = None,
) -> ResolvedOutputPolicy:
    """Resolve an explicit output shaping decision from policies and context."""
    ctx = context or OutputPolicyResolutionContext()
    source_type = ctx.source_type

    if token_policy is None and output_policy is None:
        return _default_disabled_resolution(source_type=source_type)

    if token_policy is not None and not token_policy.enabled:
        return _disabled_token_policy_resolution(token_policy=token_policy, source_type=source_type)

    profile, profile_reason = _resolve_profile(
        token_policy=token_policy,
        output_policy=output_policy,
        context=ctx,
    )
    compression_level = _resolve_compression_level(token_policy=token_policy)
    max_output_tokens = output_policy.max_output_tokens if output_policy is not None else None

    allow_lossy = token_policy.allow_lossy if token_policy is not None else False
    require_validation = token_policy.require_validation if token_policy is not None else True
    fallback_on_validation_failure = (
        token_policy.fallback_on_validation_failure if token_policy is not None else True
    )
    emit_receipts = token_policy.emit_receipts if token_policy is not None else False
    emit_observability = token_policy.emit_observability if token_policy is not None else False

    enabled = _resolve_enabled(
        token_policy=token_policy,
        output_policy=output_policy,
        source_type=source_type,
    )

    profile, compression_level, allow_lossy, require_validation, source_constrained = (
        _apply_source_type_constraints(
            source_type=source_type,
            profile=profile,
            compression_level=compression_level,
            allow_lossy=allow_lossy,
            require_validation=require_validation,
        )
    )

    reason = profile_reason
    if source_constrained:
        reason = OutputPolicyResolutionReason.SOURCE_TYPE_CONSTRAINT

    metadata = _merge_resolution_metadata(
        token_policy=token_policy,
        output_policy=output_policy,
        context=ctx,
        source_constrained=source_constrained,
    )

    return ResolvedOutputPolicy(
        status=OutputPolicyResolutionStatus.RESOLVED,
        reason=reason,
        enabled=enabled,
        profile=profile,
        compression_level=compression_level,
        allow_lossy=allow_lossy,
        require_validation=require_validation,
        fallback_on_validation_failure=fallback_on_validation_failure,
        emit_receipts=emit_receipts,
        emit_observability=emit_observability,
        max_output_tokens=max_output_tokens,
        metadata=metadata,
    )


def validate_resolved_output_policy(
    resolved: ResolvedOutputPolicy,
) -> OutputPolicyResolutionValidationResult:
    """Validate a resolved output policy deterministically without raw content."""
    failures: list[str] = []

    if resolved.profile is None:
        failures.append("profile must be present")

    if not resolved.enabled:
        if resolved.allow_lossy:
            failures.append("disabled resolution must not allow lossy behavior")
        if resolved.compression_level in _AGGRESSIVE_COMPRESSION_LEVELS:
            failures.append("disabled resolution must not use aggressive compression")

    source_type = resolved.metadata.get("source_type")
    if source_type == TokenOptimizationSourceType.SYSTEM_POLICY.value and resolved.allow_lossy:
        failures.append("system_policy resolution must not allow lossy behavior")

    if resolved.allow_lossy and not resolved.require_validation:
        failures.append("allow_lossy requires require_validation to be true")

    if resolved.require_validation and not resolved.fallback_on_validation_failure:
        failures.append(
            "require_validation requires fallback_on_validation_failure to be true"
        )

    if resolved.max_output_tokens is not None and resolved.max_output_tokens <= 0:
        failures.append("max_output_tokens must be greater than zero when present")

    if failures:
        return OutputPolicyResolutionValidationResult(
            status=OutputPolicyResolutionValidationStatus.FAILED,
            failures=tuple(failures),
        )

    return OutputPolicyResolutionValidationResult(
        status=OutputPolicyResolutionValidationStatus.PASSED,
    )


def _default_disabled_resolution(
    *,
    source_type: TokenOptimizationSourceType | None,
) -> ResolvedOutputPolicy:
    return ResolvedOutputPolicy(
        status=OutputPolicyResolutionStatus.DEFAULTED,
        reason=OutputPolicyResolutionReason.NO_POLICY,
        enabled=False,
        profile=OutputProfile.STANDARD,
        compression_level=CompressionLevel.OFF,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=False,
        emit_observability=False,
        metadata=_context_metadata(source_type=source_type),
    )


def _disabled_token_policy_resolution(
    *,
    token_policy: TokenOptimizationPolicy,
    source_type: TokenOptimizationSourceType | None,
) -> ResolvedOutputPolicy:
    return ResolvedOutputPolicy(
        status=OutputPolicyResolutionStatus.DISABLED,
        reason=OutputPolicyResolutionReason.TOKEN_OPTIMIZATION_DISABLED,
        enabled=False,
        profile=OutputProfile.STANDARD,
        compression_level=CompressionLevel.OFF,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=token_policy.emit_receipts,
        emit_observability=token_policy.emit_observability,
        metadata=_context_metadata(source_type=source_type),
    )


def _resolve_profile(
    *,
    token_policy: TokenOptimizationPolicy | None,
    output_policy: OutputPolicy | None,
    context: OutputPolicyResolutionContext,
) -> tuple[OutputProfile, OutputPolicyResolutionReason]:
    if output_policy is not None:
        return output_policy.profile, OutputPolicyResolutionReason.EXPLICIT_OUTPUT_POLICY

    if context.requested_output_profile is not None:
        return (
            context.requested_output_profile,
            OutputPolicyResolutionReason.CONTEXT_REQUESTED_PROFILE,
        )

    if token_policy is not None:
        return (
            _PROFILE_TO_OUTPUT[token_policy.profile],
            OutputPolicyResolutionReason.DERIVED_FROM_TOKEN_POLICY,
        )

    return OutputProfile.STANDARD, OutputPolicyResolutionReason.NO_POLICY


def _resolve_compression_level(
    *,
    token_policy: TokenOptimizationPolicy | None,
) -> CompressionLevel:
    if token_policy is None:
        return CompressionLevel.OFF
    return token_policy.compression_level


def _resolve_enabled(
    *,
    token_policy: TokenOptimizationPolicy | None,
    output_policy: OutputPolicy | None,
    source_type: TokenOptimizationSourceType | None,
) -> bool:
    if token_policy is not None:
        if not token_policy.enabled:
            return False
        if source_type in _CONTENT_OPTIMIZATION_SOURCE_TYPES:
            return True
        if source_type is TokenOptimizationSourceType.OUTPUT:
            return True
        if source_type in {
            TokenOptimizationSourceType.PROMPT,
            TokenOptimizationSourceType.SYSTEM_POLICY,
        }:
            return False
        if output_policy is not None:
            return True
        return True

    return output_policy is not None


def _apply_source_type_constraints(
    *,
    source_type: TokenOptimizationSourceType | None,
    profile: OutputProfile,
    compression_level: CompressionLevel,
    allow_lossy: bool,
    require_validation: bool,
) -> tuple[OutputProfile, CompressionLevel, bool, bool, bool]:
    constrained = False

    if source_type is TokenOptimizationSourceType.SYSTEM_POLICY:
        if allow_lossy:
            allow_lossy = False
            constrained = True
        if not require_validation:
            require_validation = True
            constrained = True
        if profile not in _SAFE_OUTPUT_PROFILES:
            profile = OutputProfile.STANDARD
            constrained = True
        if compression_level is not CompressionLevel.OFF:
            compression_level = CompressionLevel.OFF
            constrained = True

    elif source_type is TokenOptimizationSourceType.PROMPT:
        if allow_lossy:
            allow_lossy = False
            constrained = True
        if compression_level in _AGGRESSIVE_COMPRESSION_LEVELS:
            compression_level = CompressionLevel.MEDIUM
            constrained = True

    elif source_type is TokenOptimizationSourceType.OUTPUT:
        pass

    return profile, compression_level, allow_lossy, require_validation, constrained


def _merge_resolution_metadata(
    *,
    token_policy: TokenOptimizationPolicy | None,
    output_policy: OutputPolicy | None,
    context: OutputPolicyResolutionContext,
    source_constrained: bool,
) -> Mapping[str, Any]:
    metadata: dict[str, Any] = dict(context.metadata)
    if context.source_type is not None:
        metadata["source_type"] = context.source_type.value
    if context.token_category is not None:
        metadata["token_category"] = context.token_category.value
    if context.runtime_profile is not None:
        metadata["runtime_profile"] = context.runtime_profile
    if context.model is not None:
        metadata["model"] = context.model
    if context.provider is not None:
        metadata["provider"] = context.provider
    if context.tenant_id is not None:
        metadata["tenant_id"] = context.tenant_id
    if context.agent_id is not None:
        metadata["agent_id"] = context.agent_id
    if context.workflow_id is not None:
        metadata["workflow_id"] = context.workflow_id
    if token_policy is not None:
        metadata["token_policy_profile"] = token_policy.profile.value
    if output_policy is not None:
        metadata["output_policy_profile"] = output_policy.profile.value
    if source_constrained:
        metadata["source_type_constrained"] = True
    return metadata


def _context_metadata(
    *,
    source_type: TokenOptimizationSourceType | None,
) -> Mapping[str, Any]:
    if source_type is None:
        return {}
    return {"source_type": source_type.value}
