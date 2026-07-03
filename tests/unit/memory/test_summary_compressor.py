# © Artur Czarnecki. All rights reserved.

"""TOKEN-5A: MemorySummaryCompressor tests."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from intergrax.memory.summary_compressor import (
    DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY,
    MemorySummaryCandidate,
    MemorySummaryCompressionConfig,
    MemorySummaryCompressionStatus,
    MemorySummaryCompressor,
    SemanticValidationResult,
    SemanticValidationStatus,
    compress_memory_summary,
    optimize_memory_summary,
)
from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenSavingsClaimConfidence,
)
from intergrax.runtime.token_optimization.receipts import hash_content

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def fake_token_counter(value: str) -> int:
    return len(value.split())


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        compression_level=CompressionLevel.LIGHT,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=True,
    )


def _compressible_summary() -> str:
    return (
        "  Session   summary   for   user   preferences.\n\n\n"
        "User   prefers   concise   answers.\n\n\n\n"
        "Next   step:   review   docs.  "
    )


def test_compresses_simple_memory_summary_deterministically() -> None:
    original = _compressible_summary()
    outcome = compress_memory_summary(original, token_policy=_enabled_policy())

    assert outcome.changed is True
    assert outcome.status is MemorySummaryCompressionStatus.APPLIED
    assert outcome.optimized_content == (
        "Session summary for user preferences.\n\n"
        "User prefers concise answers.\n\n"
        "Next step: review docs."
    )
    assert outcome.original_content == original


def test_original_input_is_not_mutated() -> None:
    original = _compressible_summary()
    candidate = MemorySummaryCandidate(
        content=original,
        summary_id="sum_1",
        metadata={"tenant": "tenant_acme"},
    )
    snapshot = copy.deepcopy(candidate)

    outcome = optimize_memory_summary(candidate, token_policy=_enabled_policy())

    assert candidate == snapshot
    assert outcome.candidate == snapshot
    assert outcome.candidate.content == original


def test_disabled_policy_bypasses_optimization() -> None:
    original = _compressible_summary()
    disabled = TokenOptimizationPolicy(enabled=False)

    outcome = compress_memory_summary(original, token_policy=disabled)

    assert outcome.optimized_content == original
    assert outcome.changed is False
    assert outcome.status is MemorySummaryCompressionStatus.BYPASSED
    assert outcome.result.decision is TokenOptimizationDecision.BYPASS
    assert outcome.result.bypass_reason is TokenOptimizationBypassReason.DISABLED


def test_disabled_compression_config_still_respects_policy() -> None:
    original = _compressible_summary()
    config = MemorySummaryCompressionConfig(
        compact_whitespace=False,
        trim_blank_lines=False,
        trim_edges=False,
    )

    outcome = optimize_memory_summary(
        original,
        token_policy=_enabled_policy(),
        config=config,
    )

    assert outcome.optimized_content == original
    assert outcome.status is MemorySummaryCompressionStatus.UNCHANGED


def test_protected_dates_are_preserved() -> None:
    original = "  Meeting   on   2026-07-01   was   scheduled.\n\n\nFollow   up   later.  "
    outcome = compress_memory_summary(original, token_policy=_enabled_policy())

    assert "2026-07-01" in outcome.optimized_content
    assert outcome.fallback_status is False


def test_protected_ids_hashes_paths_and_urls_are_preserved() -> None:
    original = (
        "  Record   run_a1b2c3d4e5f67890   at   "
        "https://example.com/docs   path   /var/log/app.log   "
        "hash   deadbeefdeadbeefdeadbeefdeadbeef   "
    )
    outcome = compress_memory_summary(original, token_policy=_enabled_policy())

    assert "run_a1b2c3d4e5f67890" in outcome.optimized_content
    assert "https://example.com/docs" in outcome.optimized_content
    assert "/var/log/app.log" in outcome.optimized_content
    assert "deadbeefdeadbeefdeadbeefdeadbeef" in outcome.optimized_content
    assert outcome.fallback_status is False


def test_protected_policy_terms_are_preserved() -> None:
    original = "  Store   DATABASE_URL   in   vault   only.\n\n\nNever   log   secrets.  "
    outcome = compress_memory_summary(original, token_policy=_enabled_policy())

    assert "DATABASE_URL" in outcome.optimized_content
    assert outcome.fallback_status is False


def test_validation_failure_falls_back_to_original() -> None:
    original = _compressible_summary()
    failed_validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
        failures=("missing date preview='2026-07-01'",),
    )

    with patch(
        "intergrax.memory.summary_compressor.validate_protected_regions",
        return_value=failed_validation,
    ):
        outcome = compress_memory_summary(original, token_policy=_enabled_policy())

    assert outcome.optimized_content == original
    assert outcome.fallback_status is True
    assert outcome.status is MemorySummaryCompressionStatus.FALLBACK
    assert outcome.result.decision is TokenOptimizationDecision.FALLBACK
    assert outcome.result.bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED


def test_semantic_validation_hook_rejection_falls_back_to_original() -> None:
    original = _compressible_summary()

    def reject_hook(orig: str, opt: str, metadata: object) -> bool:
        return False

    outcome = compress_memory_summary(
        original,
        token_policy=_enabled_policy(),
        semantic_validation_hook=reject_hook,
    )

    assert outcome.optimized_content == original
    assert outcome.fallback_status is True
    assert outcome.semantic_validation_status is SemanticValidationStatus.FAILED
    assert outcome.status is MemorySummaryCompressionStatus.FALLBACK


def test_semantic_validation_hook_acceptance_allows_optimized_candidate() -> None:
    original = _compressible_summary()

    def accept_hook(orig: str, opt: str, metadata: object) -> SemanticValidationResult:
        return SemanticValidationResult(status=SemanticValidationStatus.PASSED)

    outcome = compress_memory_summary(
        original,
        token_policy=_enabled_policy(),
        semantic_validation_hook=accept_hook,
    )

    assert outcome.optimized_content != original
    assert outcome.fallback_status is False
    assert outcome.semantic_validation_status is SemanticValidationStatus.PASSED
    assert outcome.status is MemorySummaryCompressionStatus.APPLIED


def test_receipt_contains_hashes_and_savings_fields() -> None:
    original = _compressible_summary()
    outcome = compress_memory_summary(
        original,
        token_policy=_enabled_policy(),
        token_counter=fake_token_counter,
    )

    assert outcome.receipt is not None
    assert outcome.receipt_ref is not None
    assert outcome.receipt.original_hash == hash_content(original)
    assert outcome.receipt.optimized_hash == hash_content(outcome.optimized_content)
    assert outcome.receipt.measurement is not None
    assert outcome.receipt.measurement.saved_tokens >= 0
    assert 0.0 <= outcome.receipt.measurement.saved_ratio <= 1.0
    assert outcome.receipt_ref.original_hash == outcome.receipt.original_hash
    assert outcome.receipt_ref.optimized_hash == outcome.receipt.optimized_hash


def test_rollback_metadata_present_on_apply_and_fallback() -> None:
    original = _compressible_summary()
    applied = compress_memory_summary(original, token_policy=_enabled_policy())
    assert applied.rollback_metadata.rollback_available is True
    assert applied.rollback_metadata.original_hash == applied.original_hash
    assert applied.rollback_metadata.optimized_hash == applied.optimized_hash
    assert applied.rollback_metadata.strategy_id == applied.strategy.strategy_id
    assert applied.rollback_metadata.rollback_source == "memory_summary_compression"

    failed_validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
        failures=("missing identifier",),
    )
    with patch(
        "intergrax.memory.summary_compressor.validate_protected_regions",
        return_value=failed_validation,
    ):
        fallback = compress_memory_summary(original, token_policy=_enabled_policy())

    assert fallback.rollback_metadata.rollback_available is True
    assert fallback.rollback_metadata.original_hash == fallback.original_hash
    assert fallback.rollback_metadata.optimized_hash == fallback.optimized_hash


def test_optional_token_counter_is_used_when_provided() -> None:
    original = _compressible_summary()
    outcome = optimize_memory_summary(
        original,
        token_policy=_enabled_policy(),
        token_counter=fake_token_counter,
    )

    assert outcome.original_tokens == fake_token_counter(original)
    assert outcome.optimized_tokens == fake_token_counter(outcome.optimized_content)
    assert outcome.saved_tokens == outcome.original_tokens - outcome.optimized_tokens
    assert outcome.result.measurement is not None
    assert outcome.result.measurement.confidence is TokenSavingsClaimConfidence.MEASURED


def test_no_token_counter_required() -> None:
    outcome = compress_memory_summary(_compressible_summary(), token_policy=_enabled_policy())

    assert outcome.original_tokens is None
    assert outcome.optimized_tokens is None
    assert outcome.saved_tokens is None
    assert outcome.saved_ratio is None
    assert outcome.result.measurement is None


def test_benchmark_ready_result_fields() -> None:
    original = _compressible_summary()
    outcome = compress_memory_summary(
        original,
        token_policy=_enabled_policy(),
        token_counter=fake_token_counter,
    )

    assert outcome.source_type is TokenOptimizationSourceType.MEMORY
    assert outcome.strategy.strategy_id == "memory_summary.light_structural_compaction"
    assert outcome.original_hash == hash_content(original)
    assert outcome.optimized_hash == hash_content(outcome.optimized_content)
    assert outcome.validation_status in (
        ProtectedRegionValidationStatus.PASSED,
        ProtectedRegionValidationStatus.NOT_APPLICABLE,
    )
    assert outcome.receipt is not None
    assert outcome.receipt_ref is not None
    assert outcome.rollback_metadata is not None


def test_class_wrapper_delegates_to_helpers() -> None:
    compressor = MemorySummaryCompressor()
    original = _compressible_summary()

    compressed = compressor.compress_memory_summary(original, token_policy=_enabled_policy())
    optimized = compressor.optimize_memory_summary(original, token_policy=_enabled_policy())

    assert compressed.optimized_content == optimized.optimized_content


def test_no_semantic_validation_status_without_hook() -> None:
    outcome = compress_memory_summary(_compressible_summary(), token_policy=_enabled_policy())
    assert outcome.semantic_validation_status is None


def _three_fact_summary() -> str:
    return (
        "User prefers concise answers.\n"
        "User is working on LKW token optimization proof.\n"
        "User does not want runtime memory wiring yet."
    )


def _lossy_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        compression_level=CompressionLevel.LIGHT,
        allow_lossy=True,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=True,
    )


def test_max_summary_chars_does_not_truncate_under_default_policy() -> None:
    original = _three_fact_summary()
    config = MemorySummaryCompressionConfig(max_summary_chars=40)

    outcome = compress_memory_summary(
        original,
        token_policy=_enabled_policy(),
        config=config,
    )

    assert outcome.optimized_content == original
    assert "User does not want runtime memory wiring yet." in outcome.optimized_content
    assert outcome.metadata["lossy_truncation_skipped"] == 1
    assert outcome.metadata["chars_truncated"] == 0


def test_max_summary_chars_does_not_silently_remove_user_fact_under_conservative_policy() -> None:
    original = _three_fact_summary()
    config = MemorySummaryCompressionConfig(max_summary_chars=55)

    outcome = optimize_memory_summary(
        original,
        token_policy=DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY,
        config=config,
    )

    assert "User does not want runtime memory wiring yet." in outcome.optimized_content
    assert outcome.metadata["lossy_truncation_skipped"] == 1
    assert outcome.fallback_status is False


def test_allow_lossy_without_semantic_hook_does_not_truncate() -> None:
    original = _three_fact_summary()
    config = MemorySummaryCompressionConfig(max_summary_chars=40)

    outcome = compress_memory_summary(
        original,
        token_policy=_lossy_policy(),
        config=config,
    )

    assert outcome.optimized_content == original
    assert "User does not want runtime memory wiring yet." in outcome.optimized_content
    assert outcome.metadata["lossy_truncation_skipped"] == 1
    assert outcome.metadata["chars_truncated"] == 0


def test_allow_lossy_with_hook_rejection_falls_back_to_original() -> None:
    original = _three_fact_summary()
    config = MemorySummaryCompressionConfig(
        compact_whitespace=False,
        trim_blank_lines=False,
        trim_edges=False,
        max_summary_chars=40,
    )

    def reject_hook(orig: str, opt: str, metadata: object) -> bool:
        return False

    outcome = compress_memory_summary(
        original,
        token_policy=_lossy_policy(),
        config=config,
        semantic_validation_hook=reject_hook,
    )

    assert outcome.optimized_content == original
    assert outcome.fallback_status is True
    assert outcome.semantic_validation_status is SemanticValidationStatus.FAILED
    assert outcome.status is MemorySummaryCompressionStatus.FALLBACK


def test_allow_lossy_with_hook_acceptance_allows_truncation_when_protected_regions_pass() -> None:
    original = _three_fact_summary()
    config = MemorySummaryCompressionConfig(
        compact_whitespace=False,
        trim_blank_lines=False,
        trim_edges=False,
        max_summary_chars=40,
    )

    def accept_hook(orig: str, opt: str, metadata: object) -> bool:
        return True

    outcome = compress_memory_summary(
        original,
        token_policy=_lossy_policy(),
        config=config,
        semantic_validation_hook=accept_hook,
    )

    assert outcome.optimized_content != original
    assert len(outcome.optimized_content) <= 40
    assert outcome.metadata["chars_truncated"] == 1
    assert outcome.fallback_status is False
    assert outcome.semantic_validation_status is SemanticValidationStatus.PASSED
    assert outcome.status is MemorySummaryCompressionStatus.APPLIED


def test_protected_dates_force_fallback_when_truncation_would_remove_them() -> None:
    original = (
        "User prefers concise answers.\n"
        "Follow-up scheduled on 2026-07-01.\n"
        "User does not want runtime memory wiring yet."
    )
    config = MemorySummaryCompressionConfig(
        compact_whitespace=False,
        trim_blank_lines=False,
        trim_edges=False,
        max_summary_chars=55,
    )

    def accept_hook(orig: str, opt: str, metadata: object) -> bool:
        return True

    outcome = compress_memory_summary(
        original,
        token_policy=_lossy_policy(),
        config=config,
        semantic_validation_hook=accept_hook,
    )

    assert outcome.optimized_content == original
    assert "2026-07-01" in outcome.optimized_content
    assert outcome.fallback_status is True
    assert outcome.status is MemorySummaryCompressionStatus.FALLBACK


def test_helper_only_no_memory_store_imports_required() -> None:
    """Compressor operates in-memory; no store write APIs are invoked."""
    original = _compressible_summary()
    outcome = compress_memory_summary(original, token_policy=_enabled_policy())
    assert outcome.changed is True
    assert "stores" not in outcome.metadata
