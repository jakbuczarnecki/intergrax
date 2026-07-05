# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-3C-B: ExactDeduplicationLayer unit tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
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
from intergrax.runtime.token_optimization.layers.exact_deduplication import (
    ExactDeduplicationLayer,
    ExactDeduplicationLayerConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_LAYER_ID = "builtin.exact_deduplication"


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
    )


def _request(
    content: str,
    *,
    source_type: TokenOptimizationSourceType = TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    policy: TokenOptimizationPolicy | None = None,
    original_content: str | None = None,
) -> TokenOptimizationLayerRequest:
    return TokenOptimizationLayerRequest(
        original_content=original_content if original_content is not None else content,
        current_content=content,
        source_type=source_type,
        policy=policy or _enabled_policy(),
    )


def test_exact_deduplication_layer_exposes_descriptor() -> None:
    layer = ExactDeduplicationLayer()
    descriptor = layer.descriptor

    assert descriptor.layer_id == _LAYER_ID
    assert descriptor.name == "Exact Deduplication"
    assert descriptor.version == "1"
    assert descriptor.strategy.mechanism is TokenOptimizationMechanism.DEDUPLICATION
    assert descriptor.strategy.kind is TokenOptimizationStrategyKind.DEDUPLICATION
    assert descriptor.safety_class is StrategySafetyClass.LOSSLESS
    assert descriptor.built_in is True
    assert descriptor.requires_validation is True
    assert TokenOptimizationSourceType.PROMPT in descriptor.supported_source_types
    assert TokenOptimizationSourceType.RAG_CONTEXT_PACK in descriptor.supported_source_types
    assert TokenOptimizationSourceType.RETRIEVED_EVIDENCE in descriptor.supported_source_types
    assert TokenOptimizationSourceType.CONVERSATION_HISTORY in descriptor.supported_source_types
    assert TokenOptimizationSourceType.TOOL_OUTPUT in descriptor.supported_source_types


def test_default_config_validates() -> None:
    config = ExactDeduplicationLayerConfig()
    assert config.case_sensitive is True
    assert config.normalize_whitespace is True
    assert config.preserve_first_occurrence is True
    assert config.min_duplicate_length == 1


def test_config_rejects_min_duplicate_length_below_one() -> None:
    with pytest.raises(ValueError, match="min_duplicate_length must be >= 1"):
        ExactDeduplicationLayerConfig(min_duplicate_length=0)


def test_empty_content_returns_bypass() -> None:
    layer = ExactDeduplicationLayer()
    result = layer.optimize(_request(""))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.output_content == ""
    assert result.bypass_reason is TokenOptimizationBypassReason.NOT_APPLICABLE


def test_content_with_no_duplicate_lines_returns_bypass() -> None:
    layer = ExactDeduplicationLayer()
    content = "alpha\nbeta\ngamma"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.output_content == content
    assert result.bypass_reason is TokenOptimizationBypassReason.NO_SAVINGS
    assert result.metadata["duplicates_removed"] == 0


def test_exact_duplicate_lines_return_apply() -> None:
    layer = ExactDeduplicationLayer()
    content = "line one\nline two\nline two\nline three"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "line one\nline two\nline three"
    assert result.metadata["duplicates_removed"] == 1


def test_first_occurrence_is_preserved() -> None:
    layer = ExactDeduplicationLayer()
    content = "first variant\nsecond line\nfirst variant"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "first variant\nsecond line\n"
    groups = result.metadata["duplicate_groups"]
    assert len(groups) == 1
    assert groups[0]["representative_line_index"] == 0
    assert groups[0]["duplicate_line_indices"] == [2]


def test_case_sensitive_behavior_is_default() -> None:
    layer = ExactDeduplicationLayer()
    content = "Hello\nhello\nHELLO"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.output_content == content
    assert result.metadata["duplicates_removed"] == 0


def test_case_insensitive_behavior_works_only_when_configured() -> None:
    layer = ExactDeduplicationLayer(
        config=ExactDeduplicationLayerConfig(case_sensitive=False),
    )
    content = "Hello\nhello\nHELLO"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "Hello\n"
    assert result.metadata["duplicates_removed"] == 2


def test_whitespace_normalization_affects_dedupe_key_when_enabled() -> None:
    layer = ExactDeduplicationLayer(
        config=ExactDeduplicationLayerConfig(normalize_whitespace=True),
    )
    content = "foo   bar\nfoo bar"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "foo   bar\n"
    assert result.metadata["duplicates_removed"] == 1


def test_metadata_reports_duplicates_removed() -> None:
    layer = ExactDeduplicationLayer()
    content = "dup\ndup\ndup\nunique"
    result = layer.optimize(_request(content))

    assert result.metadata["duplicates_removed"] == 2
    assert len(result.metadata["duplicate_groups"]) == 1


def test_metadata_reports_dedupe_saved_chars() -> None:
    layer = ExactDeduplicationLayer()
    content = "repeat\nrepeat\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.metadata["dedupe_saved_chars"] == len("repeat\n")


def test_trailing_newline_preserved_when_duplicate_removed() -> None:
    layer = ExactDeduplicationLayer()
    content = "alpha\nbeta\nalpha\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "alpha\nbeta\n"
    assert result.metadata["dedupe_saved_chars"] == len("alpha\n")


def test_crlf_line_endings_preserved_for_kept_lines() -> None:
    layer = ExactDeduplicationLayer()
    content = "alpha\r\nbeta\r\nalpha\r\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "alpha\r\nbeta\r\n"
    assert result.metadata["dedupe_saved_chars"] == len("alpha\r\n")


def test_mixed_line_endings_preserved_for_kept_lines() -> None:
    layer = ExactDeduplicationLayer()
    content = "alpha\nbeta\r\nalpha\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "alpha\nbeta\r\n"
    assert result.metadata["dedupe_saved_chars"] == len("alpha\n")


def test_dedupe_saved_chars_equals_removed_raw_duplicate_line_length() -> None:
    layer = ExactDeduplicationLayer()
    content = "short\nlonger line\nshort\nlonger line\r\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    expected_saved = len("short\n") + len("longer line\r\n")
    assert result.metadata["dedupe_saved_chars"] == expected_saved
    assert result.metadata["dedupe_saved_chars"] == len(content) - len(result.output_content)


def test_dedupe_key_ignores_line_ending_but_output_keeps_original_ending() -> None:
    layer = ExactDeduplicationLayer()
    content = "alpha\nalpha\r\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "alpha\n"
    assert result.metadata["duplicates_removed"] == 1
    assert result.metadata["dedupe_saved_chars"] == len("alpha\r\n")


def test_duplicate_groups_metadata_contains_no_raw_content() -> None:
    layer = ExactDeduplicationLayer()
    secret_line = "secret-token-abc123"
    content = f"{secret_line}\nother\n{secret_line}\n"
    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    metadata_blob = str(result.metadata["duplicate_groups"])
    assert secret_line not in metadata_blob
    for group in result.metadata["duplicate_groups"]:
        assert set(group.keys()) == {
            "representative_line_index",
            "duplicate_line_indices",
            "dedupe_key_hash",
        }


def test_metadata_contains_base_config_and_effective_config() -> None:
    config = ExactDeduplicationLayerConfig(case_sensitive=False)
    layer = ExactDeduplicationLayer(config=config)
    result = layer.optimize(_request("a\na"))

    assert result.metadata["base_config"]["case_sensitive"] is False
    assert result.metadata["effective_config"]["case_sensitive"] is False
    assert result.metadata["config_overrides"] == {}


def test_normal_dedupe_does_not_override_previous_changes() -> None:
    layer = ExactDeduplicationLayer()
    result = layer.optimize(_request("x\nx", original_content="original\noriginal"))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.previous_changes_overridden is False
    assert result.overridden_layer_ids == ()
    assert result.override_reason is None


def test_unsupported_source_type_returns_bypass() -> None:
    layer = ExactDeduplicationLayer()
    result = layer.optimize(
        _request("dup\ndup", source_type=TokenOptimizationSourceType.TOOL_CATALOG),
    )

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.bypass_reason is TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE
    assert result.output_content == "dup\ndup"


def test_protected_region_validation_failure_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = ExactDeduplicationLayer()
    content = "dup\ndup"

    def _force_failed_validation(
        original_content: str,
        optimized_content: str,
        **kwargs: object,
    ) -> ProtectedRegionValidationResult:
        return ProtectedRegionValidationResult(
            status=ProtectedRegionValidationStatus.FAILED,
            regions_checked=1,
            regions_preserved=0,
            regions_failed=1,
            failures=("protected region missing",),
        )

    monkeypatch.setattr(
        "intergrax.runtime.token_optimization.layers.exact_deduplication.validate_protected_regions",
        _force_failed_validation,
    )

    result = layer.optimize(_request(content))

    assert result.decision is TokenOptimizationLayerDecision.FALLBACK
    assert result.fallback_used is True
    assert result.output_content == content
    assert result.metadata["fallback_reason"] == "protected_region_validation_failed"
    assert result.metadata["dedupe_saved_chars"] == 0
