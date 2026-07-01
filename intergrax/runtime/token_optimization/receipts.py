# © Artur Czarnecki. All rights reserved.

"""Compression receipt builders and integrity validators (Phase TOKEN-1C)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    CompressionReceiptRef,
    ProtectedRegionValidationStatus,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationRequest,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyRef,
    TokenSavingsMeasurement,
    ProtectedRegionValidationResult,
)

_SUPPORTED_HASH_ALGORITHMS: frozenset[str] = frozenset({"sha256"})
_DEFAULT_HASH_ALGORITHM = "sha256"
_RECEIPT_ID_PREFIX = "receipt_"
_RECEIPT_ID_HASH_LENGTH = 16


class CompressionReceiptValidationStatus(StrEnum):
    """Outcome of deterministic receipt integrity validation."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class CompressionReceiptValidationResult:
    """Receipt integrity validation outcome."""

    status: CompressionReceiptValidationStatus
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompressionReceipt:
    """Full compression receipt for audit and attribution."""

    receipt_id: str
    created_at: str
    source_type: TokenOptimizationSourceType
    decision: TokenOptimizationDecision
    original_hash: str
    optimized_hash: str
    measurement: TokenSavingsMeasurement | None = None
    validation: ProtectedRegionValidationResult | None = None
    strategy: TokenOptimizationStrategyRef | None = None
    attribution: TokenOptimizationAttribution | None = None
    fallback_used: bool = False
    bypass_reason: TokenOptimizationBypassReason | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    content_hash_algorithm: str = _DEFAULT_HASH_ALGORITHM
    plugin_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None


def hash_content(content: str, *, algorithm: str = _DEFAULT_HASH_ALGORITHM) -> str:
    """Return a deterministic lowercase hex digest for UTF-8 encoded content."""
    normalized = algorithm.lower()
    if normalized not in _SUPPORTED_HASH_ALGORITHMS:
        supported = ", ".join(sorted(_SUPPORTED_HASH_ALGORITHMS))
        raise ValueError(f"unsupported hash algorithm: {algorithm!r}; supported: {supported}")
    return hashlib.new(normalized, content.encode("utf-8")).hexdigest()


def _derive_receipt_id(
    *,
    source_type: TokenOptimizationSourceType,
    original_hash: str,
    optimized_hash: str,
    strategy: TokenOptimizationStrategyRef | None,
    decision: TokenOptimizationDecision,
) -> str:
    strategy_key = strategy.strategy_id if strategy is not None else decision.value
    payload = f"{source_type.value}|{original_hash}|{optimized_hash}|{strategy_key}"
    short_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_RECEIPT_ID_HASH_LENGTH]
    return f"{_RECEIPT_ID_PREFIX}{short_digest}"


def _resolve_strategy(
    request: TokenOptimizationRequest,
    result: TokenOptimizationResult,
) -> TokenOptimizationStrategyRef | None:
    if result.strategy is not None:
        return result.strategy
    return request.strategy


def _resolve_attribution(
    request: TokenOptimizationRequest,
    result: TokenOptimizationResult,
) -> TokenOptimizationAttribution | None:
    if request.attribution is not None:
        return request.attribution
    if result.measurement is not None and result.measurement.attribution is not None:
        return result.measurement.attribution
    return None


def _resolve_plugin_id(
    strategy: TokenOptimizationStrategyRef | None,
    attribution: TokenOptimizationAttribution | None,
) -> str | None:
    if strategy is not None and strategy.plugin_id is not None:
        return strategy.plugin_id
    if attribution is not None and attribution.plugin_id is not None:
        return attribution.plugin_id
    return None


def build_compression_receipt(
    *,
    original_content: str,
    optimized_content: str,
    request: TokenOptimizationRequest,
    result: TokenOptimizationResult,
    receipt_id: str | None = None,
    created_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompressionReceipt:
    """Build a compression receipt from request/result contracts without mutating content."""
    original_hash = hash_content(original_content)
    optimized_hash = hash_content(optimized_content)
    strategy = _resolve_strategy(request, result)
    attribution = _resolve_attribution(request, result)
    resolved_receipt_id = receipt_id or _derive_receipt_id(
        source_type=request.source_type,
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        strategy=strategy,
        decision=result.decision,
    )
    resolved_created_at = created_at or datetime.now(UTC).isoformat()
    combined_metadata: dict[str, Any] = {}
    if request.metadata:
        combined_metadata.update(request.metadata)
    if result.metadata:
        combined_metadata.update(result.metadata)
    if metadata:
        combined_metadata.update(metadata)

    return CompressionReceipt(
        receipt_id=resolved_receipt_id,
        created_at=resolved_created_at,
        source_type=request.source_type,
        decision=result.decision,
        original_hash=original_hash,
        optimized_hash=optimized_hash,
        measurement=result.measurement,
        validation=result.validation,
        strategy=strategy,
        attribution=attribution,
        fallback_used=result.fallback_used,
        bypass_reason=result.bypass_reason,
        metadata=combined_metadata,
        plugin_id=_resolve_plugin_id(strategy, attribution),
        run_id=attribution.run_id if attribution is not None else None,
        step_id=attribution.step_id if attribution is not None else None,
    )


def make_compression_receipt_ref(
    receipt: CompressionReceipt,
) -> CompressionReceiptRef:
    """Map a full receipt to the minimal receipt reference contract."""
    strategy_id = receipt.strategy.strategy_id if receipt.strategy is not None else None
    return CompressionReceiptRef(
        receipt_id=receipt.receipt_id,
        run_id=receipt.run_id,
        step_id=receipt.step_id,
        strategy_id=strategy_id,
        original_hash=receipt.original_hash,
        optimized_hash=receipt.optimized_hash,
        metadata=receipt.metadata,
    )


def validate_receipt_integrity(
    receipt: CompressionReceipt,
    *,
    original_content: str | None = None,
    optimized_content: str | None = None,
) -> CompressionReceiptValidationResult:
    """Validate deterministic receipt integrity without exposing raw content."""
    failures: list[str] = []

    if not receipt.receipt_id:
        failures.append("receipt_id must not be empty")
    if not receipt.original_hash:
        failures.append("original_hash must not be empty")
    if not receipt.optimized_hash:
        failures.append("optimized_hash must not be empty")

    if original_content is not None:
        if hash_content(original_content) != receipt.original_hash:
            failures.append("original_content hash mismatch")

    if optimized_content is not None:
        if hash_content(optimized_content) != receipt.optimized_hash:
            failures.append("optimized_content hash mismatch")

    if (
        receipt.validation is not None
        and receipt.validation.status is ProtectedRegionValidationStatus.FAILED
    ):
        failures.append("protected_region_validation_failed")

    if failures:
        return CompressionReceiptValidationResult(
            status=CompressionReceiptValidationStatus.FAILED,
            failures=tuple(failures),
        )

    content_provided = original_content is not None or optimized_content is not None
    status = (
        CompressionReceiptValidationStatus.PASSED
        if content_provided
        else CompressionReceiptValidationStatus.NOT_APPLICABLE
    )
    return CompressionReceiptValidationResult(status=status)
