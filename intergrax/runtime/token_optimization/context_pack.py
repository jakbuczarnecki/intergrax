# © Artur Czarnecki. All rights reserved.

"""Context pack optimizer (Phase TOKEN-4)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from intergrax.runtime.token_optimization.protected_regions import (
    detect_protected_regions,
    validate_protected_regions,
)
from intergrax.runtime.token_optimization.receipts import (
    CompressionReceipt,
    build_compression_receipt,
    make_compression_receipt_ref,
)

_TRUNCATION_SUFFIX = "…"
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_CONTEXT_PACK_TOKEN_POLICY = TokenOptimizationPolicy(
    enabled=True,
    profile=TokenOptimizationProfile.CONSERVATIVE,
    compression_level=CompressionLevel.LIGHT,
    allow_lossy=False,
    require_validation=True,
    fallback_on_validation_failure=True,
    emit_receipts=True,
    emit_observability=False,
)

DEFAULT_CONTEXT_PACK_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id="context_pack.light_structural_compaction",
    mechanism=TokenOptimizationMechanism.RAG_CONTEXT_PACK_COMPRESSION,
    kind=TokenOptimizationStrategyKind.LOSSLESS_STRUCTURAL_COMPRESSION,
    safety_class=StrategySafetyClass.LOSSLESS,
    plugin_id="builtin.context_pack_optimizer",
)


class ContextPackOptimizationMode(StrEnum):
    """Compaction intensity for context pack fragments."""

    LIGHT = "light"


class ContextPackOptimizationStatus(StrEnum):
    """High-level optimizer outcome."""

    APPLIED = "applied"
    BYPASSED = "bypassed"
    FALLBACK = "fallback"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class ContextFragment:
    """Single context fragment in a pack."""

    fragment_id: str
    content: str
    source_type: TokenOptimizationSourceType = TokenOptimizationSourceType.RAG_CONTEXT_PACK
    required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextPackOptimizationConfig:
    """Safe defaults for deterministic context pack compaction."""

    mode: ContextPackOptimizationMode = ContextPackOptimizationMode.LIGHT
    max_fragment_chars: int = 1200
    compact_whitespace: bool = True
    trim_fragments: bool = True
    preserve_required_fragments: bool = True
    include_receipt: bool = True


@dataclass(frozen=True, slots=True)
class ContextPackOptimizationOutcome:
    """Full optimizer outcome without runtime wiring."""

    original_content: str
    optimized_content: str
    original_fragments: tuple[ContextFragment, ...]
    optimized_fragments: tuple[ContextFragment, ...]
    request: TokenOptimizationRequest
    result: TokenOptimizationResult
    receipt: CompressionReceipt | None
    receipt_ref: CompressionReceiptRef | None
    protected_region_validation: ProtectedRegionValidationResult
    resolved_output_policy: ResolvedOutputPolicy
    changed: bool
    status: ContextPackOptimizationStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ContextPackOptimizer:
    """Class wrapper for deterministic context pack compaction."""

    def __init__(self, config: ContextPackOptimizationConfig | None = None) -> None:
        self._config = config or ContextPackOptimizationConfig()

    def optimize_pack(
        self,
        fragments: Sequence[ContextFragment | Mapping[str, Any] | str],
        *,
        token_policy: TokenOptimizationPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        attribution: TokenOptimizationAttribution | None = None,
        config: ContextPackOptimizationConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> ContextPackOptimizationOutcome:
        return optimize_context_pack(
            fragments,
            token_policy=token_policy,
            output_policy=output_policy,
            attribution=attribution,
            config=config or self._config,
            token_counter=token_counter,
        )


def optimize_context_pack(
    fragments: Sequence[ContextFragment | Mapping[str, Any] | str],
    *,
    token_policy: TokenOptimizationPolicy | None = None,
    output_policy: OutputPolicy | None = None,
    attribution: TokenOptimizationAttribution | None = None,
    config: ContextPackOptimizationConfig | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> ContextPackOptimizationOutcome:
    """Produce a compact context pack view without mutating input fragments."""
    cfg = config or ContextPackOptimizationConfig()
    effective_policy = (
        token_policy if token_policy is not None else DEFAULT_CONTEXT_PACK_TOKEN_POLICY
    )
    resolved_policy = resolve_output_policy(
        token_policy=effective_policy,
        output_policy=output_policy,
        context=OutputPolicyResolutionContext(
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            token_category=TokenCategory.RAG_CONTEXT_PACK,
        ),
    )

    original_fragments, input_kind = _parse_fragments(fragments)
    original_content = _serialize_context_pack(original_fragments)

    if not effective_policy.enabled or not resolved_policy.enabled:
        return _build_bypass_outcome(
            original_content=original_content,
            optimized_content=original_content,
            original_fragments=original_fragments,
            optimized_fragments=original_fragments,
            effective_policy=effective_policy,
            resolved_policy=resolved_policy,
            attribution=attribution,
            config=cfg,
            token_counter=token_counter,
            bypass_reason=TokenOptimizationBypassReason.DISABLED,
            status=ContextPackOptimizationStatus.BYPASSED,
            metadata=_base_metadata(
                cfg=cfg,
                input_kind=input_kind,
                fragment_count=len(original_fragments),
                changed=False,
            ),
        )

    optimized_fragments, compaction_stats = _compact_fragments(original_fragments, cfg=cfg)
    optimized_content = _serialize_context_pack(optimized_fragments)

    validation = validate_protected_regions(original_content, optimized_content)
    fallback_used = False
    decision = TokenOptimizationDecision.APPLY
    bypass_reason: TokenOptimizationBypassReason | None = None
    status = ContextPackOptimizationStatus.APPLIED

    if (
        validation.status is ProtectedRegionValidationStatus.FAILED
        and effective_policy.fallback_on_validation_failure
    ):
        optimized_content = original_content
        optimized_fragments = original_fragments
        fallback_used = True
        decision = TokenOptimizationDecision.FALLBACK
        bypass_reason = TokenOptimizationBypassReason.VALIDATION_FAILED
        status = ContextPackOptimizationStatus.FALLBACK
    elif optimized_content == original_content:
        status = ContextPackOptimizationStatus.UNCHANGED
        if decision is TokenOptimizationDecision.APPLY and not fallback_used:
            decision = TokenOptimizationDecision.BYPASS
            bypass_reason = TokenOptimizationBypassReason.NO_SAVINGS

    changed = optimized_content != original_content
    metadata = _base_metadata(
        cfg=cfg,
        input_kind=input_kind,
        fragment_count=len(original_fragments),
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
        },
    )
    result = TokenOptimizationResult(
        content=optimized_content,
        decision=decision,
        measurement=measurement,
        validation=validation,
        strategy=DEFAULT_CONTEXT_PACK_STRATEGY,
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

    return ContextPackOptimizationOutcome(
        original_content=original_content,
        optimized_content=optimized_content,
        original_fragments=original_fragments,
        optimized_fragments=optimized_fragments,
        request=request,
        result=result,
        receipt=receipt,
        receipt_ref=receipt_ref,
        protected_region_validation=validation,
        resolved_output_policy=resolved_policy,
        changed=changed,
        status=status,
        metadata=dict(metadata),
    )


def _parse_fragments(
    fragments: Sequence[ContextFragment | Mapping[str, Any] | str],
) -> tuple[tuple[ContextFragment, ...], str]:
    parsed: list[ContextFragment] = []
    kinds: set[str] = set()
    for index, item in enumerate(fragments):
        if isinstance(item, ContextFragment):
            kinds.add("fragment")
            parsed.append(item)
        elif isinstance(item, str):
            kinds.add("string")
            parsed.append(
                ContextFragment(
                    fragment_id=f"fragment_{index}",
                    content=item,
                )
            )
        elif isinstance(item, Mapping):
            kinds.add("mapping")
            parsed.append(_parse_mapping_fragment(item, index))
        else:
            kinds.add("unknown")
            parsed.append(
                ContextFragment(
                    fragment_id=f"fragment_{index}",
                    content=str(item),
                )
            )

    if len(kinds) == 1:
        input_kind = next(iter(kinds))
    elif not kinds:
        input_kind = "empty"
    else:
        input_kind = "mixed"

    return tuple(parsed), input_kind


def _parse_mapping_fragment(item: Mapping[str, Any], index: int) -> ContextFragment:
    fragment_id = str(item.get("fragment_id") or item.get("id") or f"fragment_{index}")
    content = str(item.get("content") if "content" in item else item.get("text", ""))
    source_type_raw = item.get("source_type", TokenOptimizationSourceType.RAG_CONTEXT_PACK)
    if isinstance(source_type_raw, TokenOptimizationSourceType):
        source_type = source_type_raw
    elif isinstance(source_type_raw, str):
        try:
            source_type = TokenOptimizationSourceType(source_type_raw)
        except ValueError:
            source_type = TokenOptimizationSourceType.RAG_CONTEXT_PACK
    else:
        source_type = TokenOptimizationSourceType.RAG_CONTEXT_PACK

    required = bool(item.get("required", False))
    raw_metadata = item.get("metadata", {})
    metadata: Mapping[str, Any]
    if isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        metadata = {}

    return ContextFragment(
        fragment_id=fragment_id,
        content=content,
        source_type=source_type,
        required=required,
        metadata=metadata,
    )


def _serialize_context_pack(fragments: Sequence[ContextFragment]) -> str:
    payload = {"fragments": [_fragment_to_dict(fragment) for fragment in fragments]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _fragment_to_dict(fragment: ContextFragment) -> dict[str, Any]:
    return {
        "fragment_id": fragment.fragment_id,
        "content": fragment.content,
        "source_type": fragment.source_type.value,
        "required": fragment.required,
        "metadata": dict(fragment.metadata),
    }


def _compact_fragments(
    fragments: Sequence[ContextFragment],
    *,
    cfg: ContextPackOptimizationConfig,
) -> tuple[tuple[ContextFragment, ...], dict[str, int]]:
    stats = {
        "fragments_compacted": 0,
        "fragments_preserved_required": 0,
        "fragments_preserved_due_to_protected_regions": 0,
        "fragments_truncated": 0,
    }
    optimized: list[ContextFragment] = []

    for fragment in fragments:
        if fragment.required and cfg.preserve_required_fragments:
            stats["fragments_preserved_required"] += 1
            optimized.append(fragment)
            continue

        compacted_content, fragment_stats = _compact_fragment_content(
            fragment.content,
            cfg=cfg,
        )
        stats["fragments_compacted"] += fragment_stats["compacted"]
        stats["fragments_preserved_due_to_protected_regions"] += fragment_stats[
            "preserved_due_to_protected"
        ]
        stats["fragments_truncated"] += fragment_stats["truncated"]

        optimized.append(
            ContextFragment(
                fragment_id=fragment.fragment_id,
                content=compacted_content,
                source_type=fragment.source_type,
                required=fragment.required,
                metadata=fragment.metadata,
            )
        )

    return tuple(optimized), stats


def _compact_fragment_content(
    content: str,
    *,
    cfg: ContextPackOptimizationConfig,
) -> tuple[str, dict[str, int]]:
    stats = {"compacted": 0, "preserved_due_to_protected": 0, "truncated": 0}
    original = content
    working = content

    if cfg.trim_fragments:
        working = working.strip()

    if cfg.compact_whitespace:
        working = _normalize_whitespace(working)

    if len(working) > cfg.max_fragment_chars:
        candidate = _truncate_at_word_boundary(working, cfg.max_fragment_chars)
        regions = detect_protected_regions(original)
        if regions:
            field_validation = validate_protected_regions(
                original,
                candidate,
                regions=regions,
            )
            if field_validation.status is ProtectedRegionValidationStatus.FAILED:
                stats["preserved_due_to_protected"] += 1
                return original, stats
        stats["truncated"] += 1
        working = candidate
    elif working != original:
        regions = detect_protected_regions(original)
        if regions:
            field_validation = validate_protected_regions(
                original,
                working,
                regions=regions,
            )
            if field_validation.status is ProtectedRegionValidationStatus.FAILED:
                stats["preserved_due_to_protected"] += 1
                return original, stats

    if working != original:
        stats["compacted"] += 1

    return working, stats


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return text[:max_chars]
    budget = max_chars - len(_TRUNCATION_SUFFIX)
    candidate = text[:budget]
    last_space = candidate.rfind(" ")
    if last_space > 0:
        candidate = candidate[:last_space]
    return candidate.rstrip() + _TRUNCATION_SUFFIX


def _base_metadata(
    *,
    cfg: ContextPackOptimizationConfig,
    input_kind: str,
    fragment_count: int,
    changed: bool,
    compaction_stats: dict[str, int] | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "mode": cfg.mode.value,
        "changed": changed,
        "fragment_count": fragment_count,
        "input_kind": input_kind,
        "fragments_compacted": 0,
        "fragments_preserved_required": 0,
        "fragments_preserved_due_to_protected_regions": 0,
        "fragments_truncated": 0,
    }
    if compaction_stats is not None:
        metadata.update(compaction_stats)
    if fallback_used:
        metadata["fallback_reason"] = "protected_region_validation_failed"
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
        category=TokenCategory.RAG_CONTEXT_PACK,
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        strategy=DEFAULT_CONTEXT_PACK_STRATEGY,
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
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        policy=policy,
        attribution=attribution,
        strategy=DEFAULT_CONTEXT_PACK_STRATEGY,
        metadata=metadata,
    )


def _build_bypass_outcome(
    *,
    original_content: str,
    optimized_content: str,
    original_fragments: tuple[ContextFragment, ...],
    optimized_fragments: tuple[ContextFragment, ...],
    effective_policy: TokenOptimizationPolicy,
    resolved_policy: ResolvedOutputPolicy,
    attribution: TokenOptimizationAttribution | None,
    config: ContextPackOptimizationConfig,
    token_counter: Callable[[str], int] | None,
    bypass_reason: TokenOptimizationBypassReason,
    status: ContextPackOptimizationStatus,
    metadata: Mapping[str, Any],
) -> ContextPackOptimizationOutcome:
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
        strategy=DEFAULT_CONTEXT_PACK_STRATEGY,
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

    return ContextPackOptimizationOutcome(
        original_content=original_content,
        optimized_content=optimized_content,
        original_fragments=original_fragments,
        optimized_fragments=optimized_fragments,
        request=request,
        result=result,
        receipt=receipt,
        receipt_ref=receipt_ref,
        protected_region_validation=validation,
        resolved_output_policy=resolved_policy,
        changed=False,
        status=status,
        metadata=dict(metadata),
    )
