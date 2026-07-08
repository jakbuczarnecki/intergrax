# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-4A: ExtractiveFilteringLayer unit tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerRequest,
    TokenOptimizationMechanism,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from intergrax.runtime.token_optimization.layers.extractive_filtering import (
    ExtractiveFilteringLayer,
    ExtractiveFilteringLayerConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_LAYER_ID = "builtin.extractive_filtering"


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.BALANCED,
    )


def _request(
    content: str,
    *,
    source_type: TokenOptimizationSourceType = TokenOptimizationSourceType.TOOL_OUTPUT,
    policy: TokenOptimizationPolicy | None = None,
    metadata: dict | None = None,
) -> TokenOptimizationLayerRequest:
    return TokenOptimizationLayerRequest(
        original_content=content,
        current_content=content,
        source_type=source_type,
        policy=policy or _enabled_policy(),
        metadata=metadata or {},
    )


def _filtering_config(**overrides: object) -> ExtractiveFilteringLayerConfig:
    defaults = {
        "min_lines_before_filtering": 10,
        "head_lines": 3,
        "tail_lines": 3,
        "max_output_chars": 4000,
    }
    defaults.update(overrides)
    return ExtractiveFilteringLayerConfig(**defaults)


def _noisy_long_output(*, middle_error: str | None = None) -> str:
    lines = [f"INFO: progress step {index}" for index in range(150)]
    if middle_error is not None:
        lines[75] = middle_error
    lines.append("INFO: final cleanup")
    return "\n".join(lines) + "\n"


def test_descriptor_uses_extractive_filtering_strategy() -> None:
    layer = ExtractiveFilteringLayer()
    descriptor = layer.descriptor

    assert descriptor.layer_id == _LAYER_ID
    assert descriptor.name == "Extractive Filtering"
    assert descriptor.version == "1"
    assert descriptor.strategy.kind is TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING
    assert descriptor.strategy.mechanism is TokenOptimizationMechanism.TERMINAL_LOG_FILTERING
    assert descriptor.safety_class is StrategySafetyClass.LOSSY
    assert descriptor.built_in is True
    assert descriptor.requires_validation is True


def test_descriptor_supports_tool_terminal_and_log_output() -> None:
    descriptor = ExtractiveFilteringLayer().descriptor
    supported = descriptor.supported_source_types

    assert TokenOptimizationSourceType.TOOL_OUTPUT in supported
    assert TokenOptimizationSourceType.TERMINAL_OUTPUT in supported
    assert TokenOptimizationSourceType.LOG_OUTPUT in supported


def test_bypasses_unsupported_source_type() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    result = layer.optimize(
        _request("line\n" * 20, source_type=TokenOptimizationSourceType.PROMPT),
    )

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.bypass_reason is TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE


def test_bypasses_when_disabled() -> None:
    layer = ExtractiveFilteringLayer(
        config=_filtering_config(enabled=False),
    )
    result = layer.optimize(_request(_noisy_long_output()))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.bypass_reason is TokenOptimizationBypassReason.DISABLED


def test_bypasses_short_clean_output_with_no_savings() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    content = "alpha\nbeta\ngamma\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.bypass_reason is TokenOptimizationBypassReason.NO_SAVINGS
    assert result.output_content == content


def test_keeps_head_and_tail_windows() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config(head_lines=2, tail_lines=2))
    lines = [f"line-{index}" for index in range(30)]
    content = "\n".join(lines) + "\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert "line-0" in result.output_content
    assert "line-1" in result.output_content
    assert "line-28" in result.output_content
    assert "line-29" in result.output_content
    assert "line-15" not in result.output_content


def test_preserves_error_lines_from_middle_of_long_output() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    content = _noisy_long_output(middle_error="ERROR: module compile failed")
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert "ERROR: module compile failed" in result.output_content


def test_preserves_warning_lines_from_middle_of_long_output() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    content = _noisy_long_output(middle_error="WARNING: deprecated API usage detected")
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert "WARNING: deprecated API usage detected" in result.output_content


def test_preserves_traceback_block() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    filler = [f"INFO: step {index}" for index in range(40)]
    traceback_lines = [
        "Traceback (most recent call last):",
        '  File "app.py", line 10, in main',
        "    raise ValueError('boom')",
        "ValueError: boom",
        "",
    ]
    content = "\n".join(filler + traceback_lines + filler) + "\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert "Traceback (most recent call last):" in result.output_content
    assert "ValueError: boom" in result.output_content
    assert result.metadata["traceback_block_count"] >= 1


def test_collapses_repeated_lines_deterministically() -> None:
    layer = ExtractiveFilteringLayer(
        config=_filtering_config(
            head_lines=20,
            repeated_line_threshold=3,
            collapse_repeated_lines=True,
        ),
    )
    repeated = "WARNING: dependency already satisfied\n"
    content = repeated * 8 + "\n".join(f"INFO: tail {index}" for index in range(20)) + "\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    warning_lines = [
        line
        for line in result.output_content.splitlines()
        if line.strip() == "WARNING: dependency already satisfied"
    ]
    assert len(warning_lines) == 1
    assert "repeated 7x" in result.output_content
    assert len(result.metadata["repeated_line_groups"]) >= 1


def test_emits_omission_marker() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    result = layer.optimize(_request(_noisy_long_output()))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert "omitted" in result.output_content
    assert "intergrax extractive filtering" in result.output_content
    assert result.metadata["omitted_line_count"] > 0


def test_output_is_shorter_than_input_for_noisy_long_output() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    content = _noisy_long_output()
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert len(result.output_content) < len(content)


def test_receipt_metadata_uses_char_level_fields_only() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    result = layer.optimize(_request(_noisy_long_output()))

    metadata = result.metadata
    assert metadata["budget_unit"] == "chars"
    assert isinstance(metadata["original_chars"], int)
    assert isinstance(metadata["output_chars"], int)
    assert isinstance(metadata["saved_chars"], int)
    assert metadata["saved_chars"] == metadata["original_chars"] - metadata["output_chars"]


def test_receipt_metadata_does_not_include_token_named_savings_fields() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    result = layer.optimize(_request(_noisy_long_output()))

    metadata_keys = set(result.metadata.keys())
    forbidden = {
        key
        for key in metadata_keys
        if "token" in key.lower() and key not in {"strategy"}
    }
    assert forbidden == set()
    assert result.measurement is None


def test_receipt_metadata_does_not_include_raw_full_input_or_output() -> None:
    secret = "SECRET_FULL_LOG_BODY_SHOULD_NOT_APPEAR_IN_METADATA"
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    content = _noisy_long_output() + secret + "\n"
    result = layer.optimize(_request(content))

    metadata_blob = str(result.metadata)
    assert secret not in metadata_blob
    assert content not in metadata_blob
    assert result.output_content not in metadata_blob


def test_protected_region_removal_risk_triggers_fallback() -> None:
    secret = "PROTECTED-SECRET-VALUE-12345"
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    lines = [f"INFO: progress {index}" for index in range(40)]
    lines[20] = f"marker before {secret}"
    content = "\n".join(lines) + "\n"
    protected = ProtectedRegion(
        kind=ProtectedRegionKind.IDENTIFIER,
        value=secret,
    )
    result = layer.optimize(
        _request(content, metadata={"protected_regions": (protected,)}),
    )

    assert result.decision is TokenOptimizationLayerDecision.FALLBACK
    assert result.fallback_used is True
    assert result.bypass_reason is TokenOptimizationBypassReason.PROTECTED_REGION_RISK
    assert result.output_content == content


def test_no_protected_region_risk_when_protected_value_remains_in_output() -> None:
    secret = "PROTECTED-HEAD-VALUE-999"
    layer = ExtractiveFilteringLayer(config=_filtering_config(head_lines=5))
    lines = [f"INFO: head {secret}"] + [f"INFO: progress {index}" for index in range(40)]
    content = "\n".join(lines) + "\n"
    protected = ProtectedRegion(
        kind=ProtectedRegionKind.IDENTIFIER,
        value=secret,
    )
    result = layer.optimize(
        _request(content, metadata={"protected_regions": (protected,)}),
    )

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert secret in result.output_content


def test_char_budget_satisfied_false_when_important_content_exceeds_max_output_chars() -> None:
    layer = ExtractiveFilteringLayer(
        config=_filtering_config(
            max_output_chars=200,
            head_lines=2,
            tail_lines=2,
        ),
    )
    errors = [f"ERROR: failure detail {index} " + ("x" * 40) for index in range(10)]
    filler = [f"INFO: progress {index}" for index in range(40)]
    content = "\n".join(filler + errors + filler) + "\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.metadata["char_budget_satisfied"] is False
    assert len(result.output_content) > 200


def test_savings_attribution_is_extractive_filtering_only() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    result = layer.optimize(_request(_noisy_long_output()))

    assert result.metadata["strategy"] == "extractive_filtering"
    assert "dedupe_saved_chars" not in result.metadata
    assert "packing_decisions" not in result.metadata
    assert result.strategy is not None
    assert result.strategy.kind is TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING


def test_crlf_line_endings_do_not_crash() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    content = "\r\n".join(f"INFO: step {index}" for index in range(30)) + "\r\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert len(result.output_content) < len(content)


def test_repeated_line_detection_is_case_sensitive_by_default() -> None:
    layer = ExtractiveFilteringLayer(
        config=_filtering_config(head_lines=10, repeated_line_threshold=2),
    )
    content = "WARN: same\nwarn: same\nWARN: same\n" + "\n".join(
        f"INFO: tail {index}" for index in range(20)
    ) + "\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert len(result.metadata["repeated_line_groups"]) == 0


def test_repeated_line_detection_can_be_case_insensitive_via_config() -> None:
    layer = ExtractiveFilteringLayer(
        config=_filtering_config(
            head_lines=10,
            repeated_line_threshold=2,
            case_sensitive_repeated_lines=False,
        ),
    )
    content = "WARN: same\nwarn: same\nWARN: same\n" + "\n".join(
        f"INFO: tail {index}" for index in range(20)
    ) + "\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert len(result.metadata["repeated_line_groups"]) >= 1


def test_empty_content_bypasses_safely() -> None:
    layer = ExtractiveFilteringLayer(config=_filtering_config())
    result = layer.optimize(_request(""))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.bypass_reason is TokenOptimizationBypassReason.NOT_APPLICABLE
    assert result.output_content == ""
