# © Artur Czarnecki. All rights reserved.

"""TOKEN-2: Output policy resolver tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    OutputPolicy,
    OutputProfile,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.output_policy import (
    OutputPolicyResolutionContext,
    OutputPolicyResolutionReason,
    OutputPolicyResolutionStatus,
    OutputPolicyResolutionValidationStatus,
    OutputPolicyResolver,
    ResolvedOutputPolicy,
    resolve_output_policy,
    validate_resolved_output_policy,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OUTPUT_POLICY_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "output_policy.py"
)


def test_resolve_without_policies_returns_safe_disabled_default() -> None:
    resolved = resolve_output_policy()

    assert resolved.status is OutputPolicyResolutionStatus.DEFAULTED
    assert resolved.reason is OutputPolicyResolutionReason.NO_POLICY
    assert resolved.enabled is False
    assert resolved.profile is OutputProfile.STANDARD
    assert resolved.compression_level is CompressionLevel.OFF
    assert resolved.allow_lossy is False
    assert resolved.require_validation is True
    assert resolved.fallback_on_validation_failure is True
    assert resolved.emit_receipts is False
    assert resolved.emit_observability is False


def test_disabled_token_policy_disables_output_shaping() -> None:
    token_policy = TokenOptimizationPolicy(
        enabled=False,
        profile=TokenOptimizationProfile.AGGRESSIVE,
        compression_level=CompressionLevel.HIGH,
        allow_lossy=True,
        emit_receipts=True,
        emit_observability=False,
    )

    resolved = resolve_output_policy(token_policy=token_policy)

    assert resolved.status is OutputPolicyResolutionStatus.DISABLED
    assert resolved.reason is OutputPolicyResolutionReason.TOKEN_OPTIMIZATION_DISABLED
    assert resolved.enabled is False
    assert resolved.profile is OutputProfile.STANDARD
    assert resolved.compression_level is CompressionLevel.OFF
    assert resolved.allow_lossy is False


def test_explicit_output_policy_wins_over_context_requested_profile() -> None:
    token_policy = TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.BALANCED)
    output_policy = OutputPolicy(profile=OutputProfile.TERSE)
    context = OutputPolicyResolutionContext(
        requested_output_profile=OutputProfile.MINIMAL,
        source_type=TokenOptimizationSourceType.OUTPUT,
    )

    resolved = resolve_output_policy(
        token_policy=token_policy,
        output_policy=output_policy,
        context=context,
    )

    assert resolved.profile is OutputProfile.TERSE
    assert resolved.reason is OutputPolicyResolutionReason.EXPLICIT_OUTPUT_POLICY


def test_context_requested_profile_used_without_explicit_output_policy() -> None:
    token_policy = TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.BALANCED)
    context = OutputPolicyResolutionContext(
        requested_output_profile=OutputProfile.FULL,
        source_type=TokenOptimizationSourceType.OUTPUT,
    )

    resolved = resolve_output_policy(token_policy=token_policy, context=context)

    assert resolved.profile is OutputProfile.FULL
    assert resolved.reason is OutputPolicyResolutionReason.CONTEXT_REQUESTED_PROFILE


def test_system_policy_forces_non_lossy_and_requires_validation() -> None:
    token_policy = TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.AGGRESSIVE,
        compression_level=CompressionLevel.HIGH,
        allow_lossy=True,
        require_validation=False,
        fallback_on_validation_failure=False,
    )
    output_policy = OutputPolicy(profile=OutputProfile.MINIMAL)
    context = OutputPolicyResolutionContext(
        source_type=TokenOptimizationSourceType.SYSTEM_POLICY,
    )

    resolved = resolve_output_policy(
        token_policy=token_policy,
        output_policy=output_policy,
        context=context,
    )

    assert resolved.allow_lossy is False
    assert resolved.require_validation is True
    assert resolved.profile is OutputProfile.STANDARD
    assert resolved.compression_level is CompressionLevel.OFF
    assert resolved.reason is OutputPolicyResolutionReason.SOURCE_TYPE_CONSTRAINT


def test_prompt_source_type_remains_conservative_and_non_lossy() -> None:
    token_policy = TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.AGGRESSIVE,
        compression_level=CompressionLevel.SEMANTIC,
        allow_lossy=True,
    )
    context = OutputPolicyResolutionContext(source_type=TokenOptimizationSourceType.PROMPT)

    resolved = resolve_output_policy(token_policy=token_policy, context=context)

    assert resolved.enabled is False
    assert resolved.allow_lossy is False
    assert resolved.compression_level is CompressionLevel.MEDIUM
    assert resolved.reason is OutputPolicyResolutionReason.SOURCE_TYPE_CONSTRAINT


def test_output_source_type_allows_shaping_when_enabled() -> None:
    token_policy = TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.AGGRESSIVE,
        compression_level=CompressionLevel.HIGH,
        allow_lossy=False,
    )
    output_policy = OutputPolicy(profile=OutputProfile.MINIMAL, max_output_tokens=512)
    context = OutputPolicyResolutionContext(source_type=TokenOptimizationSourceType.OUTPUT)

    resolved = resolve_output_policy(
        token_policy=token_policy,
        output_policy=output_policy,
        context=context,
    )

    assert resolved.enabled is True
    assert resolved.profile is OutputProfile.MINIMAL
    assert resolved.compression_level is CompressionLevel.HIGH
    assert resolved.max_output_tokens == 512


@pytest.mark.parametrize(
    "source_type",
    [
        TokenOptimizationSourceType.TOOL_OUTPUT,
        TokenOptimizationSourceType.LOG_OUTPUT,
        TokenOptimizationSourceType.TERMINAL_OUTPUT,
    ],
)
def test_content_source_types_preserve_validation_and_fallback(
    source_type: TokenOptimizationSourceType,
) -> None:
    token_policy = TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        require_validation=True,
        fallback_on_validation_failure=True,
    )
    context = OutputPolicyResolutionContext(source_type=source_type)

    resolved = resolve_output_policy(token_policy=token_policy, context=context)

    assert resolved.enabled is True
    assert resolved.require_validation is True
    assert resolved.fallback_on_validation_failure is True


def test_resolver_copies_emit_flags_from_token_policy() -> None:
    token_policy = TokenOptimizationPolicy(
        enabled=True,
        emit_receipts=False,
        emit_observability=True,
    )
    context = OutputPolicyResolutionContext(source_type=TokenOptimizationSourceType.OUTPUT)

    resolved = resolve_output_policy(token_policy=token_policy, context=context)

    assert resolved.emit_receipts is False
    assert resolved.emit_observability is True


def test_resolver_does_not_mutate_input_policies_or_context() -> None:
    token_policy = TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.BALANCED)
    output_policy = OutputPolicy(profile=OutputProfile.STANDARD)
    context = OutputPolicyResolutionContext(
        requested_output_profile=OutputProfile.TERSE,
        source_type=TokenOptimizationSourceType.OUTPUT,
        metadata={"trace": "abc"},
    )
    token_snapshot = (
        token_policy.enabled,
        token_policy.profile,
        token_policy.compression_level,
        token_policy.allow_lossy,
    )
    output_snapshot = (output_policy.profile, output_policy.max_output_tokens)
    context_snapshot = (
        context.requested_output_profile,
        context.source_type,
        dict(context.metadata),
    )

    resolve_output_policy(
        token_policy=token_policy,
        output_policy=output_policy,
        context=context,
    )

    assert (
        token_policy.enabled,
        token_policy.profile,
        token_policy.compression_level,
        token_policy.allow_lossy,
    ) == token_snapshot
    assert (output_policy.profile, output_policy.max_output_tokens) == output_snapshot
    assert (
        context.requested_output_profile,
        context.source_type,
        dict(context.metadata),
    ) == context_snapshot


def test_output_policy_resolver_class_delegates_to_helper() -> None:
    resolver = OutputPolicyResolver()
    resolved = resolver.resolve(
        token_policy=TokenOptimizationPolicy(enabled=True),
        context=OutputPolicyResolutionContext(source_type=TokenOptimizationSourceType.OUTPUT),
    )
    assert resolved.enabled is True


def test_validate_resolved_output_policy_passes_valid_resolution() -> None:
    resolved = ResolvedOutputPolicy(
        status=OutputPolicyResolutionStatus.RESOLVED,
        reason=OutputPolicyResolutionReason.EXPLICIT_OUTPUT_POLICY,
        enabled=True,
        profile=OutputProfile.STANDARD,
        compression_level=CompressionLevel.LIGHT,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=False,
        emit_observability=False,
        max_output_tokens=100,
    )

    result = validate_resolved_output_policy(resolved)

    assert result.status is OutputPolicyResolutionValidationStatus.PASSED
    assert result.failures == ()


def test_validate_fails_lossy_resolution_without_validation() -> None:
    resolved = ResolvedOutputPolicy(
        status=OutputPolicyResolutionStatus.RESOLVED,
        reason=OutputPolicyResolutionReason.DERIVED_FROM_TOKEN_POLICY,
        enabled=True,
        profile=OutputProfile.MINIMAL,
        compression_level=CompressionLevel.HIGH,
        allow_lossy=True,
        require_validation=False,
        fallback_on_validation_failure=True,
        emit_receipts=False,
        emit_observability=False,
    )

    result = validate_resolved_output_policy(resolved)

    assert result.status is OutputPolicyResolutionValidationStatus.FAILED
    assert "allow_lossy requires require_validation to be true" in result.failures


def test_validate_fails_non_positive_max_output_tokens() -> None:
    resolved = ResolvedOutputPolicy(
        status=OutputPolicyResolutionStatus.RESOLVED,
        reason=OutputPolicyResolutionReason.EXPLICIT_OUTPUT_POLICY,
        enabled=True,
        profile=OutputProfile.STANDARD,
        compression_level=CompressionLevel.OFF,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=False,
        emit_observability=False,
        max_output_tokens=0,
    )

    result = validate_resolved_output_policy(resolved)

    assert result.status is OutputPolicyResolutionValidationStatus.FAILED
    assert "max_output_tokens must be greater than zero when present" in result.failures


def test_output_policy_module_has_no_runtime_model_or_telemetry_imports() -> None:
    tree = ast.parse(_OUTPUT_POLICY_MODULE.read_text(encoding="utf-8"))
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    forbidden_roots = {
        "openai",
        "anthropic",
        "tiktoken",
        "transformers",
        "httpx",
        "requests",
    }
    assert imported_roots.isdisjoint(forbidden_roots)

    source = _OUTPUT_POLICY_MODULE.read_text(encoding="utf-8")
    assert "telemetry" not in source
    assert "intergrax.runtime.token_optimization.telemetry" not in source
