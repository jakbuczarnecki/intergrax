# © Artur Czarnecki. All rights reserved.

"""Tool catalog schema optimizer (Phase TOKEN-3)."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    CompressionReceiptRef,
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
    CompressionLevel,
    OutputPolicy,
)
from intergrax.runtime.token_optimization.receipts import (
    CompressionReceipt,
    build_compression_receipt,
    make_compression_receipt_ref,
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

_TRUNCATION_SUFFIX = "…"
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_TOOL_CATALOG_TOKEN_POLICY = TokenOptimizationPolicy(
    enabled=True,
    profile=TokenOptimizationProfile.CONSERVATIVE,
    compression_level=CompressionLevel.LIGHT,
    allow_lossy=False,
    require_validation=True,
    fallback_on_validation_failure=True,
    emit_receipts=True,
    emit_observability=False,
)

DEFAULT_TOOL_CATALOG_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id="tool_catalog.light_compaction",
    mechanism=TokenOptimizationMechanism.TOOL_CATALOG_COMPACTION,
    kind=TokenOptimizationStrategyKind.SCHEMA_MINIMIZATION,
    safety_class=StrategySafetyClass.LOSSLESS,
    plugin_id="builtin.tool_schema_optimizer",
)


class ToolSchemaOptimizationMode(StrEnum):
    """Compaction intensity for LLM-facing tool catalog views."""

    LIGHT = "light"


class ToolSchemaOptimizationStatus(StrEnum):
    """High-level optimizer outcome."""

    APPLIED = "applied"
    BYPASSED = "bypassed"
    FALLBACK = "fallback"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class ToolSchemaOptimizationConfig:
    """Safe defaults for deterministic tool catalog compaction."""

    mode: ToolSchemaOptimizationMode = ToolSchemaOptimizationMode.LIGHT
    max_description_chars: int = 240
    compact_json: bool = True
    preserve_property_order: bool = True
    allow_example_removal: bool = False
    include_receipt: bool = True


@dataclass(frozen=True, slots=True)
class ToolSchemaOptimizationOutcome:
    """Full optimizer outcome without runtime wiring."""

    original_content: str
    optimized_content: str
    request: TokenOptimizationRequest
    result: TokenOptimizationResult
    receipt: CompressionReceipt | None
    receipt_ref: CompressionReceiptRef | None
    protected_region_validation: ProtectedRegionValidationResult
    resolved_output_policy: ResolvedOutputPolicy
    changed: bool
    status: ToolSchemaOptimizationStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ToolSchemaOptimizer:
    """Class wrapper for deterministic tool catalog compaction."""

    def __init__(self, config: ToolSchemaOptimizationConfig | None = None) -> None:
        self._config = config or ToolSchemaOptimizationConfig()

    def optimize_catalog(
        self,
        tool_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str,
        *,
        token_policy: TokenOptimizationPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        attribution: TokenOptimizationAttribution | None = None,
        config: ToolSchemaOptimizationConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> ToolSchemaOptimizationOutcome:
        return optimize_tool_schema_catalog(
            tool_catalog,
            token_policy=token_policy,
            output_policy=output_policy,
            attribution=attribution,
            config=config or self._config,
            token_counter=token_counter,
        )


def optimize_tool_schema_catalog(
    tool_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str,
    *,
    token_policy: TokenOptimizationPolicy | None = None,
    output_policy: OutputPolicy | None = None,
    attribution: TokenOptimizationAttribution | None = None,
    config: ToolSchemaOptimizationConfig | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> ToolSchemaOptimizationOutcome:
    """Produce a compact LLM-facing tool catalog view without mutating inputs."""
    cfg = config or ToolSchemaOptimizationConfig()
    effective_policy = (
        token_policy if token_policy is not None else DEFAULT_TOOL_CATALOG_TOKEN_POLICY
    )
    resolved_policy = resolve_output_policy(
        token_policy=effective_policy,
        output_policy=output_policy,
        context=OutputPolicyResolutionContext(
            source_type=TokenOptimizationSourceType.TOOL_CATALOG,
            token_category=TokenCategory.TOOL_CATALOG,
        ),
    )

    parsed, input_kind, parse_error = _parse_tool_catalog(tool_catalog)
    if parse_error is not None:
        return _build_bypass_outcome(
            original_content=parse_error,
            optimized_content=parse_error,
            effective_policy=effective_policy,
            resolved_policy=resolved_policy,
            attribution=attribution,
            config=cfg,
            token_counter=token_counter,
            bypass_reason=TokenOptimizationBypassReason.NOT_APPLICABLE,
            status=ToolSchemaOptimizationStatus.BYPASSED,
            metadata={
                "mode": cfg.mode.value,
                "changed": False,
                "input_kind": input_kind,
                "fallback_reason": "invalid_json",
            },
        )

    original_content = _serialize_catalog(
        parsed,
        compact=cfg.compact_json,
        preserve_property_order=cfg.preserve_property_order,
    )

    if not effective_policy.enabled or not resolved_policy.enabled:
        return _build_bypass_outcome(
            original_content=original_content,
            optimized_content=original_content,
            effective_policy=effective_policy,
            resolved_policy=resolved_policy,
            attribution=attribution,
            config=cfg,
            token_counter=token_counter,
            bypass_reason=TokenOptimizationBypassReason.DISABLED,
            status=ToolSchemaOptimizationStatus.BYPASSED,
            metadata={
                "mode": cfg.mode.value,
                "changed": False,
                "input_kind": input_kind,
                "description_fields_compacted": 0,
                "description_fields_preserved_due_to_protected_regions": 0,
            },
        )

    working_copy = copy.deepcopy(parsed)
    compaction_stats = _compact_catalog_descriptions(
        working_copy,
        max_description_chars=cfg.max_description_chars,
        allow_example_removal=cfg.allow_example_removal,
    )
    optimized_content = _serialize_catalog(
        working_copy,
        compact=cfg.compact_json,
        preserve_property_order=cfg.preserve_property_order,
    )

    validation = validate_protected_regions(original_content, optimized_content)
    fallback_used = False
    decision = TokenOptimizationDecision.APPLY
    bypass_reason: TokenOptimizationBypassReason | None = None
    status = ToolSchemaOptimizationStatus.APPLIED

    if (
        validation.status is ProtectedRegionValidationStatus.FAILED
        and effective_policy.fallback_on_validation_failure
    ):
        optimized_content = original_content
        fallback_used = True
        decision = TokenOptimizationDecision.FALLBACK
        bypass_reason = TokenOptimizationBypassReason.VALIDATION_FAILED
        status = ToolSchemaOptimizationStatus.FALLBACK
    elif optimized_content == original_content:
        status = ToolSchemaOptimizationStatus.UNCHANGED
        if decision is TokenOptimizationDecision.APPLY and not fallback_used:
            decision = TokenOptimizationDecision.BYPASS
            bypass_reason = TokenOptimizationBypassReason.NO_SAVINGS

    changed = optimized_content != original_content
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
        strategy=DEFAULT_TOOL_CATALOG_STRATEGY,
        fallback_used=fallback_used,
        bypass_reason=bypass_reason,
        metadata={
            "mode": cfg.mode.value,
            "changed": changed,
            "input_kind": input_kind,
            "description_fields_compacted": compaction_stats["compacted"],
            "description_fields_preserved_due_to_protected_regions": compaction_stats[
                "preserved_due_to_protected_regions"
            ],
            **({"fallback_reason": "protected_region_validation_failed"} if fallback_used else {}),
        },
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

    return ToolSchemaOptimizationOutcome(
        original_content=original_content,
        optimized_content=optimized_content,
        request=request,
        result=result,
        receipt=receipt,
        receipt_ref=receipt_ref,
        protected_region_validation=validation,
        resolved_output_policy=resolved_policy,
        changed=changed,
        status=status,
        metadata=dict(result.metadata),
    )


def _parse_tool_catalog(
    tool_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str,
) -> tuple[Any | None, str, str | None]:
    if isinstance(tool_catalog, str):
        try:
            parsed = json.loads(tool_catalog)
        except json.JSONDecodeError:
            return None, "json_string", tool_catalog
        return parsed, "json_string", None

    if isinstance(tool_catalog, Mapping):
        return tool_catalog, "mapping", None

    return list(tool_catalog), "sequence", None


def _serialize_catalog(
    data: Any,
    *,
    compact: bool,
    preserve_property_order: bool,
) -> str:
    kwargs: dict[str, Any] = {"ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    if not preserve_property_order:
        kwargs["sort_keys"] = True
    return json.dumps(data, **kwargs)


def _compact_catalog_descriptions(
    node: Any,
    *,
    max_description_chars: int,
    allow_example_removal: bool,
) -> dict[str, int]:
    stats = {"compacted": 0, "preserved_due_to_protected_regions": 0}

    if isinstance(node, Mapping):
        for key, value in list(node.items()):
            if key == "description" and isinstance(value, str):
                normalized, preserved = _normalize_description_field(
                    value,
                    max_description_chars=max_description_chars,
                )
                if normalized != value:
                    stats["compacted"] += 1
                if preserved:
                    stats["preserved_due_to_protected_regions"] += 1
                node[key] = normalized
            elif key == "examples" and allow_example_removal:
                continue
            else:
                child_stats = _compact_catalog_descriptions(
                    value,
                    max_description_chars=max_description_chars,
                    allow_example_removal=allow_example_removal,
                )
                stats["compacted"] += child_stats["compacted"]
                stats["preserved_due_to_protected_regions"] += child_stats[
                    "preserved_due_to_protected_regions"
                ]
    elif isinstance(node, list):
        for item in node:
            child_stats = _compact_catalog_descriptions(
                item,
                max_description_chars=max_description_chars,
                allow_example_removal=allow_example_removal,
            )
            stats["compacted"] += child_stats["compacted"]
            stats["preserved_due_to_protected_regions"] += child_stats[
                "preserved_due_to_protected_regions"
            ]

    return stats


def _normalize_description_field(
    value: str,
    *,
    max_description_chars: int,
) -> tuple[str, bool]:
    stripped = value.strip()
    normalized = _normalize_whitespace(stripped)
    if len(normalized) <= max_description_chars:
        return normalized, False

    shortened = _truncate_at_word_boundary(normalized, max_description_chars)
    regions = detect_protected_regions(value)
    if regions:
        field_validation = validate_protected_regions(value, shortened, regions=regions)
        if field_validation.status is ProtectedRegionValidationStatus.FAILED:
            return value, True
    return shortened, False


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
        category=TokenCategory.TOOL_CATALOG,
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        strategy=DEFAULT_TOOL_CATALOG_STRATEGY,
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
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        policy=policy,
        attribution=attribution,
        strategy=DEFAULT_TOOL_CATALOG_STRATEGY,
        metadata=metadata,
    )


def _build_bypass_outcome(
    *,
    original_content: str,
    optimized_content: str,
    effective_policy: TokenOptimizationPolicy,
    resolved_policy: ResolvedOutputPolicy,
    attribution: TokenOptimizationAttribution | None,
    config: ToolSchemaOptimizationConfig,
    token_counter: Callable[[str], int] | None,
    bypass_reason: TokenOptimizationBypassReason,
    status: ToolSchemaOptimizationStatus,
    metadata: Mapping[str, Any],
) -> ToolSchemaOptimizationOutcome:
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
        metadata=metadata,
    )
    result = TokenOptimizationResult(
        content=optimized_content,
        decision=TokenOptimizationDecision.BYPASS,
        measurement=measurement,
        validation=validation,
        strategy=DEFAULT_TOOL_CATALOG_STRATEGY,
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

    return ToolSchemaOptimizationOutcome(
        original_content=original_content,
        optimized_content=optimized_content,
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
