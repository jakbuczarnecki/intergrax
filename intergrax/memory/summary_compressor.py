# © Artur Czarnecki. All rights reserved.

"""Memory summary compressor (Phase TOKEN-5A)."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    CompressionReceiptRef,
    OutputPolicy,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
    TokenSavingsClaimConfidence,
    TokenSavingsMeasurement,
)
from intergrax.runtime.token_optimization.output_policy import (
    OutputPolicyResolutionContext,
    ResolvedOutputPolicy,
    resolve_output_policy,
)
from intergrax.runtime.token_optimization.protected_regions import validate_protected_regions
from intergrax.runtime.token_optimization.receipts import (
    CompressionReceipt,
    build_compression_receipt,
    hash_content,
    make_compression_receipt_ref,
)

_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")

DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY = TokenOptimizationPolicy(
    enabled=True,
    profile=TokenOptimizationProfile.CONSERVATIVE,
    compression_level=CompressionLevel.LIGHT,
    allow_lossy=False,
    require_validation=True,
    fallback_on_validation_failure=True,
    emit_receipts=True,
    emit_observability=False,
)

DEFAULT_MEMORY_SUMMARY_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id="memory_summary.light_structural_compaction",
    mechanism=TokenOptimizationMechanism.MEMORY_CONTEXT_PRUNING,
    kind=TokenOptimizationStrategyKind.LOSSLESS_STRUCTURAL_COMPRESSION,
    safety_class=StrategySafetyClass.LOSSLESS,
    plugin_id="builtin.memory_summary_compressor",
)


class MemorySummaryCompressionMode(StrEnum):
    """Compaction intensity for persistent memory summaries."""

    LIGHT = "light"


class MemorySummaryCompressionStatus(StrEnum):
    """High-level compressor outcome."""

    APPLIED = "applied"
    BYPASSED = "bypassed"
    FALLBACK = "fallback"
    UNCHANGED = "unchanged"


class SemanticValidationStatus(StrEnum):
    """Outcome of optional semantic validation hook."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class SemanticValidationResult:
    """Structured semantic validation hook result."""

    status: SemanticValidationStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)


SemanticValidationHook = Callable[
    [str, str, Mapping[str, Any]],
    bool | SemanticValidationResult,
]


@dataclass(frozen=True, slots=True)
class MemorySummaryCandidate:
    """Staged memory-summary candidate without live-store mutation."""

    content: str
    summary_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySummaryRollbackMetadata:
    """Rollback metadata for future persistent integration."""

    original_hash: str
    optimized_hash: str
    rollback_available: bool
    rollback_source: str
    strategy_id: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySummaryCompressionConfig:
    """Safe defaults for deterministic memory summary compaction."""

    mode: MemorySummaryCompressionMode = MemorySummaryCompressionMode.LIGHT
    compact_whitespace: bool = True
    trim_blank_lines: bool = True
    trim_edges: bool = True
    max_summary_chars: int | None = None
    include_receipt: bool = True


@dataclass(frozen=True, slots=True)
class MemorySummaryCompressionOutcome:
    """Full compressor outcome without runtime or store wiring."""

    original_content: str
    optimized_content: str
    candidate: MemorySummaryCandidate
    request: TokenOptimizationRequest
    result: TokenOptimizationResult
    receipt: CompressionReceipt | None
    receipt_ref: CompressionReceiptRef | None
    protected_region_validation: ProtectedRegionValidationResult
    resolved_output_policy: ResolvedOutputPolicy
    rollback_metadata: MemorySummaryRollbackMetadata
    changed: bool
    status: MemorySummaryCompressionStatus
    source_type: TokenOptimizationSourceType
    strategy: TokenOptimizationStrategyRef
    original_hash: str
    optimized_hash: str
    original_tokens: int | None
    optimized_tokens: int | None
    saved_tokens: int | None
    saved_ratio: float | None
    validation_status: ProtectedRegionValidationStatus
    fallback_status: bool
    semantic_validation_status: SemanticValidationStatus | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MemorySummaryCompressor:
    """Class wrapper for deterministic memory summary compaction."""

    def __init__(self, config: MemorySummaryCompressionConfig | None = None) -> None:
        self._config = config or MemorySummaryCompressionConfig()

    def compress_memory_summary(
        self,
        summary: str | MemorySummaryCandidate | Mapping[str, Any],
        *,
        token_policy: TokenOptimizationPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        attribution: TokenOptimizationAttribution | None = None,
        config: MemorySummaryCompressionConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
        semantic_validation_hook: SemanticValidationHook | None = None,
    ) -> MemorySummaryCompressionOutcome:
        return compress_memory_summary(
            summary,
            token_policy=token_policy,
            output_policy=output_policy,
            attribution=attribution,
            config=config or self._config,
            token_counter=token_counter,
            semantic_validation_hook=semantic_validation_hook,
        )

    def optimize_memory_summary(
        self,
        summary: str | MemorySummaryCandidate | Mapping[str, Any],
        *,
        token_policy: TokenOptimizationPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        attribution: TokenOptimizationAttribution | None = None,
        config: MemorySummaryCompressionConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
        semantic_validation_hook: SemanticValidationHook | None = None,
    ) -> MemorySummaryCompressionOutcome:
        return optimize_memory_summary(
            summary,
            token_policy=token_policy,
            output_policy=output_policy,
            attribution=attribution,
            config=config or self._config,
            token_counter=token_counter,
            semantic_validation_hook=semantic_validation_hook,
        )


def compress_memory_summary(
    summary: str | MemorySummaryCandidate | Mapping[str, Any],
    *,
    token_policy: TokenOptimizationPolicy | None = None,
    output_policy: OutputPolicy | None = None,
    attribution: TokenOptimizationAttribution | None = None,
    config: MemorySummaryCompressionConfig | None = None,
    token_counter: Callable[[str], int] | None = None,
    semantic_validation_hook: SemanticValidationHook | None = None,
) -> MemorySummaryCompressionOutcome:
    """Compress a memory-summary candidate without mutating live memory stores."""
    return optimize_memory_summary(
        summary,
        token_policy=token_policy,
        output_policy=output_policy,
        attribution=attribution,
        config=config,
        token_counter=token_counter,
        semantic_validation_hook=semantic_validation_hook,
    )


def optimize_memory_summary(
    summary: str | MemorySummaryCandidate | Mapping[str, Any],
    *,
    token_policy: TokenOptimizationPolicy | None = None,
    output_policy: OutputPolicy | None = None,
    attribution: TokenOptimizationAttribution | None = None,
    config: MemorySummaryCompressionConfig | None = None,
    token_counter: Callable[[str], int] | None = None,
    semantic_validation_hook: SemanticValidationHook | None = None,
) -> MemorySummaryCompressionOutcome:
    """Produce a compact memory-summary view without mutating inputs or stores."""
    cfg = config or MemorySummaryCompressionConfig()
    effective_policy = (
        token_policy if token_policy is not None else DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY
    )
    resolved_policy = resolve_output_policy(
        token_policy=effective_policy,
        output_policy=output_policy,
        context=OutputPolicyResolutionContext(
            source_type=TokenOptimizationSourceType.MEMORY,
            token_category=TokenCategory.MEMORY,
        ),
    )

    candidate, input_kind = _parse_candidate(summary)
    original_content = candidate.content

    if not effective_policy.enabled or not resolved_policy.enabled:
        return _build_bypass_outcome(
            original_content=original_content,
            optimized_content=original_content,
            candidate=candidate,
            effective_policy=effective_policy,
            resolved_policy=resolved_policy,
            attribution=attribution,
            config=cfg,
            token_counter=token_counter,
            bypass_reason=TokenOptimizationBypassReason.DISABLED,
            status=MemorySummaryCompressionStatus.BYPASSED,
            metadata=_base_metadata(
                cfg=cfg,
                input_kind=input_kind,
                changed=False,
            ),
            semantic_validation_hook=semantic_validation_hook,
        )

    optimized_content, compaction_stats = _compact_summary_content(original_content, cfg=cfg)
    optimized_content, compaction_stats = _apply_lossy_truncation_guard(
        original_content=original_content,
        optimized_content=optimized_content,
        compaction_stats=compaction_stats,
        cfg=cfg,
        effective_policy=effective_policy,
        semantic_validation_hook=semantic_validation_hook,
    )

    validation = validate_protected_regions(original_content, optimized_content)
    fallback_used = False
    decision = TokenOptimizationDecision.APPLY
    bypass_reason: TokenOptimizationBypassReason | None = None
    status = MemorySummaryCompressionStatus.APPLIED
    semantic_status: SemanticValidationStatus | None = None

    if (
        validation.status is ProtectedRegionValidationStatus.FAILED
        and effective_policy.fallback_on_validation_failure
    ):
        optimized_content = original_content
        fallback_used = True
        decision = TokenOptimizationDecision.FALLBACK
        bypass_reason = TokenOptimizationBypassReason.VALIDATION_FAILED
        status = MemorySummaryCompressionStatus.FALLBACK
    elif semantic_validation_hook is not None and optimized_content != original_content:
        semantic_status = _run_semantic_validation_hook(
            semantic_validation_hook,
            original_content=original_content,
            optimized_content=optimized_content,
            metadata={
                "mode": cfg.mode.value,
                "input_kind": input_kind,
                **compaction_stats,
            },
        )
        if semantic_status is SemanticValidationStatus.FAILED:
            optimized_content = original_content
            fallback_used = True
            decision = TokenOptimizationDecision.FALLBACK
            bypass_reason = TokenOptimizationBypassReason.QUALITY_RISK
            status = MemorySummaryCompressionStatus.FALLBACK
    elif optimized_content == original_content:
        status = MemorySummaryCompressionStatus.UNCHANGED
        if decision is TokenOptimizationDecision.APPLY and not fallback_used:
            decision = TokenOptimizationDecision.BYPASS
            bypass_reason = TokenOptimizationBypassReason.NO_SAVINGS

    changed = optimized_content != original_content
    metadata = _base_metadata(
        cfg=cfg,
        input_kind=input_kind,
        changed=changed,
        compaction_stats=compaction_stats,
        fallback_used=fallback_used,
    )
    measurement = _build_measurement(
        original_content=original_content,
        optimized_content=optimized_content,
        token_counter=token_counter,
        attribution=attribution,
    )

    request = _build_request(
        content=original_content,
        policy=effective_policy,
        attribution=attribution,
        metadata={
            "mode": cfg.mode.value,
            "input_kind": input_kind,
            "summary_id": candidate.summary_id,
        },
    )
    result = TokenOptimizationResult(
        content=optimized_content,
        decision=decision,
        measurement=measurement,
        validation=validation,
        strategy=DEFAULT_MEMORY_SUMMARY_STRATEGY,
        fallback_used=fallback_used,
        bypass_reason=bypass_reason,
        metadata=dict(metadata),
    )

    receipt: CompressionReceipt | None = None
    receipt_ref: CompressionReceiptRef | None = None
    if cfg.include_receipt:
        receipt = build_compression_receipt(
            original_content=original_content,
            optimized_content=optimized_content,
            request=request,
            result=result,
        )
        receipt_ref = make_compression_receipt_ref(receipt)

    original_hash = hash_content(original_content)
    optimized_hash = hash_content(optimized_content)
    rollback_metadata = _build_rollback_metadata(
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        fallback_used=fallback_used,
        receipt=receipt,
    )

    return MemorySummaryCompressionOutcome(
        original_content=original_content,
        optimized_content=optimized_content,
        candidate=candidate,
        request=request,
        result=result,
        receipt=receipt,
        receipt_ref=receipt_ref,
        protected_region_validation=validation,
        resolved_output_policy=resolved_policy,
        rollback_metadata=rollback_metadata,
        changed=changed,
        status=status,
        source_type=TokenOptimizationSourceType.MEMORY,
        strategy=DEFAULT_MEMORY_SUMMARY_STRATEGY,
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        original_tokens=measurement.baseline_tokens if measurement else None,
        optimized_tokens=measurement.optimized_tokens if measurement else None,
        saved_tokens=measurement.saved_tokens if measurement else None,
        saved_ratio=measurement.saved_ratio if measurement else None,
        validation_status=validation.status,
        fallback_status=fallback_used,
        semantic_validation_status=semantic_status,
        metadata=dict(metadata),
    )


def _parse_candidate(
    summary: str | MemorySummaryCandidate | Mapping[str, Any],
) -> tuple[MemorySummaryCandidate, str]:
    if isinstance(summary, MemorySummaryCandidate):
        return summary, "candidate"
    if isinstance(summary, str):
        return MemorySummaryCandidate(content=summary), "string"
    if isinstance(summary, Mapping):
        content = str(summary.get("content") if "content" in summary else summary.get("text", ""))
        summary_id_raw = summary.get("summary_id", summary.get("id"))
        summary_id = str(summary_id_raw) if summary_id_raw is not None else None
        raw_metadata = summary.get("metadata", {})
        metadata: Mapping[str, Any]
        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        else:
            metadata = {}
        return (
            MemorySummaryCandidate(
                content=content,
                summary_id=summary_id,
                metadata=metadata,
            ),
            "mapping",
        )
    return MemorySummaryCandidate(content=str(summary)), "unknown"


def _compact_summary_content(
    content: str,
    *,
    cfg: MemorySummaryCompressionConfig,
) -> tuple[str, dict[str, int]]:
    stats = {
        "lines_trimmed": 0,
        "whitespace_compacted": 0,
        "chars_truncated": 0,
        "lossy_truncation_skipped": 0,
    }
    working = content

    if cfg.trim_edges:
        stripped = working.strip()
        if stripped != working:
            stats["whitespace_compacted"] += 1
        working = stripped

    if cfg.trim_blank_lines:
        collapsed = _BLANK_LINES_RE.sub("\n\n", working)
        if collapsed != working:
            stats["lines_trimmed"] += 1
        working = collapsed

    if cfg.compact_whitespace:
        compacted = _compact_horizontal_whitespace(working)
        if compacted != working:
            stats["whitespace_compacted"] += 1
        working = compacted

    return working, stats


def _apply_lossy_truncation_guard(
    *,
    original_content: str,
    optimized_content: str,
    compaction_stats: dict[str, int],
    cfg: MemorySummaryCompressionConfig,
    effective_policy: TokenOptimizationPolicy,
    semantic_validation_hook: SemanticValidationHook | None,
) -> tuple[str, dict[str, int]]:
    """Apply max_summary_chars only under explicit lossy policy with semantic hook."""
    stats = dict(compaction_stats)
    if cfg.max_summary_chars is None or len(optimized_content) <= cfg.max_summary_chars:
        return optimized_content, stats

    lossy_allowed = (
        effective_policy.allow_lossy and semantic_validation_hook is not None
    )
    if not lossy_allowed:
        stats["lossy_truncation_skipped"] = 1
        return optimized_content, stats

    truncated = optimized_content[: cfg.max_summary_chars].rstrip()
    stats["chars_truncated"] = 1
    return truncated, stats


def _compact_horizontal_whitespace(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        compacted = _HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip()
        lines.append(compacted)
    return "\n".join(lines)


def _run_semantic_validation_hook(
    hook: SemanticValidationHook,
    *,
    original_content: str,
    optimized_content: str,
    metadata: Mapping[str, Any],
) -> SemanticValidationStatus:
    outcome = hook(original_content, optimized_content, metadata)
    if isinstance(outcome, SemanticValidationResult):
        return outcome.status
    if outcome:
        return SemanticValidationStatus.PASSED
    return SemanticValidationStatus.FAILED


def _build_rollback_metadata(
    *,
    original_hash: str,
    optimized_hash: str,
    fallback_used: bool,
    receipt: CompressionReceipt | None,
) -> MemorySummaryRollbackMetadata:
    created_at = receipt.created_at if receipt is not None else datetime.now(UTC).isoformat()
    return MemorySummaryRollbackMetadata(
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        rollback_available=True,
        rollback_source="memory_summary_compression",
        strategy_id=DEFAULT_MEMORY_SUMMARY_STRATEGY.strategy_id,
        created_at=created_at,
        metadata={
            "fallback_used": fallback_used,
            "receipt_id": receipt.receipt_id if receipt is not None else None,
        },
    )


def _base_metadata(
    *,
    cfg: MemorySummaryCompressionConfig,
    input_kind: str,
    changed: bool,
    compaction_stats: dict[str, int] | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "mode": cfg.mode.value,
        "changed": changed,
        "input_kind": input_kind,
        "lines_trimmed": 0,
        "whitespace_compacted": 0,
        "chars_truncated": 0,
        "lossy_truncation_skipped": 0,
    }
    if compaction_stats is not None:
        metadata.update(compaction_stats)
    if fallback_used:
        metadata["fallback_reason"] = "protected_region_or_semantic_validation_failed"
    return metadata


def _build_measurement(
    *,
    original_content: str,
    optimized_content: str,
    token_counter: Callable[[str], int] | None,
    attribution: TokenOptimizationAttribution | None,
) -> TokenSavingsMeasurement | None:
    if token_counter is None:
        return None
    baseline_tokens = token_counter(original_content)
    optimized_tokens = token_counter(optimized_content)
    saved_tokens = baseline_tokens - optimized_tokens
    saved_ratio = saved_tokens / baseline_tokens if baseline_tokens > 0 else 0.0
    return TokenSavingsMeasurement(
        baseline_tokens=baseline_tokens,
        optimized_tokens=optimized_tokens,
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        confidence=TokenSavingsClaimConfidence.MEASURED,
        category=TokenCategory.MEMORY,
        source_type=TokenOptimizationSourceType.MEMORY,
        strategy=DEFAULT_MEMORY_SUMMARY_STRATEGY,
        attribution=attribution,
    )


def _build_request(
    *,
    content: str,
    policy: TokenOptimizationPolicy,
    attribution: TokenOptimizationAttribution | None,
    metadata: Mapping[str, Any],
) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=content,
        source_type=TokenOptimizationSourceType.MEMORY,
        policy=policy,
        attribution=attribution,
        strategy=DEFAULT_MEMORY_SUMMARY_STRATEGY,
        metadata=metadata,
    )


def _build_bypass_outcome(
    *,
    original_content: str,
    optimized_content: str,
    candidate: MemorySummaryCandidate,
    effective_policy: TokenOptimizationPolicy,
    resolved_policy: ResolvedOutputPolicy,
    attribution: TokenOptimizationAttribution | None,
    config: MemorySummaryCompressionConfig,
    token_counter: Callable[[str], int] | None,
    bypass_reason: TokenOptimizationBypassReason,
    status: MemorySummaryCompressionStatus,
    metadata: Mapping[str, Any],
    semantic_validation_hook: SemanticValidationHook | None,
) -> MemorySummaryCompressionOutcome:
    validation = validate_protected_regions(original_content, optimized_content)
    measurement = _build_measurement(
        original_content=original_content,
        optimized_content=optimized_content,
        token_counter=token_counter,
        attribution=attribution,
    )
    request = _build_request(
        content=original_content,
        policy=effective_policy,
        attribution=attribution,
        metadata=dict(metadata),
    )
    result = TokenOptimizationResult(
        content=optimized_content,
        decision=TokenOptimizationDecision.BYPASS,
        measurement=measurement,
        validation=validation,
        strategy=DEFAULT_MEMORY_SUMMARY_STRATEGY,
        fallback_used=False,
        bypass_reason=bypass_reason,
        metadata=dict(metadata),
    )
    receipt: CompressionReceipt | None = None
    receipt_ref: CompressionReceiptRef | None = None
    if config.include_receipt:
        receipt = build_compression_receipt(
            original_content=original_content,
            optimized_content=optimized_content,
            request=request,
            result=result,
        )
        receipt_ref = make_compression_receipt_ref(receipt)

    original_hash = hash_content(original_content)
    optimized_hash = hash_content(optimized_content)
    rollback_metadata = _build_rollback_metadata(
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        fallback_used=False,
        receipt=receipt,
    )

    return MemorySummaryCompressionOutcome(
        original_content=original_content,
        optimized_content=optimized_content,
        candidate=candidate,
        request=request,
        result=result,
        receipt=receipt,
        receipt_ref=receipt_ref,
        protected_region_validation=validation,
        resolved_output_policy=resolved_policy,
        rollback_metadata=rollback_metadata,
        changed=False,
        status=status,
        source_type=TokenOptimizationSourceType.MEMORY,
        strategy=DEFAULT_MEMORY_SUMMARY_STRATEGY,
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        original_tokens=measurement.baseline_tokens if measurement else None,
        optimized_tokens=measurement.optimized_tokens if measurement else None,
        saved_tokens=measurement.saved_tokens if measurement else None,
        saved_ratio=measurement.saved_ratio if measurement else None,
        validation_status=validation.status,
        fallback_status=False,
        semantic_validation_status=(
            SemanticValidationStatus.SKIPPED if semantic_validation_hook is not None else None
        ),
        metadata=dict(metadata),
    )
