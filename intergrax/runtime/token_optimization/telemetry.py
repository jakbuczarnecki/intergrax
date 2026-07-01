# © Artur Czarnecki. All rights reserved.

"""Token savings telemetry payload contracts and helpers (Phase TOKEN-6A-lite / TOKEN-6A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.context_pack import ContextPackOptimizationOutcome
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationStatus,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenSavingsClaimConfidence,
)
from intergrax.runtime.token_optimization.output_policy import (
    OutputPolicyResolutionStatus,
    ResolvedOutputPolicy,
)
from intergrax.runtime.token_optimization.receipts import CompressionReceipt
from intergrax.runtime.token_optimization.tool_schema import ToolSchemaOptimizationOutcome

_ATTRIBUTE_PREFIX = "intergrax.token_optimization."


class TokenOptimizationTelemetryEventType(StrEnum):
    """Stable event type strings for token optimization telemetry."""

    TOKEN_SAVINGS = "token_optimization.savings"
    TOKEN_OPTIMIZATION_SUMMARY = "token_optimization.summary"


class TokenOptimizationTelemetrySource(StrEnum):
    """Contributing source kinds for token optimization telemetry summaries."""

    COMPRESSION_RECEIPT = "compression_receipt"
    RESOLVED_OUTPUT_POLICY = "resolved_output_policy"
    TOOL_SCHEMA_OUTCOME = "tool_schema_outcome"
    CONTEXT_PACK_OUTCOME = "context_pack_outcome"


class TokenOptimizationTelemetryValidationStatus(StrEnum):
    """Outcome of deterministic telemetry payload validation."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class TokenOptimizationTelemetrySummaryValidationStatus(StrEnum):
    """Outcome of deterministic telemetry summary validation."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TokenOptimizationTelemetryValidationResult:
    """Telemetry payload validation outcome."""

    status: TokenOptimizationTelemetryValidationStatus
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationTelemetrySummaryValidationResult:
    """Telemetry summary validation outcome."""

    status: TokenOptimizationTelemetrySummaryValidationStatus
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationCounterSnapshot:
    """Safe aggregate counters for token optimization activity."""

    total_receipts: int = 0
    total_resolved_policies: int = 0
    total_tool_schema_outcomes: int = 0
    total_context_pack_outcomes: int = 0
    applied_count: int = 0
    bypassed_count: int = 0
    fallback_count: int = 0
    failed_count: int = 0
    unchanged_count: int = 0
    validation_passed_count: int = 0
    validation_failed_count: int = 0
    validation_not_applicable_count: int = 0
    fallback_used_count: int = 0
    receipts_with_measurement_count: int = 0
    baseline_tokens: int = 0
    optimized_tokens: int = 0
    saved_tokens: int = 0
    saved_ratio: float = 0.0
    tool_schema_changed_count: int = 0
    context_pack_changed_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationTelemetrySummary:
    """Typed summary payload for future HOS/domain-signal emission."""

    event_type: TokenOptimizationTelemetryEventType
    snapshot: TokenOptimizationCounterSnapshot
    source_types: tuple[TokenOptimizationSourceType, ...] = ()
    strategy_ids: tuple[str, ...] = ()
    plugin_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    workflow_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationTelemetryPayload:
    """Typed token-savings telemetry payload shape for future HOS/domain-signal emission."""

    event_type: TokenOptimizationTelemetryEventType
    receipt_id: str
    source_type: TokenOptimizationSourceType
    decision: TokenOptimizationDecision
    run_id: str | None = None
    step_id: str | None = None
    workflow_id: str | None = None
    tenant_id: str | None = None
    agent_id: str | None = None
    model: str | None = None
    provider: str | None = None
    runtime_profile: str | None = None
    optimization_profile: TokenOptimizationProfile | None = None
    token_category: TokenCategory | None = None
    strategy_id: str | None = None
    mechanism: TokenOptimizationMechanism | None = None
    strategy_kind: TokenOptimizationStrategyKind | None = None
    plugin_id: str | None = None
    fallback_used: bool = False
    bypass_reason: TokenOptimizationBypassReason | None = None
    validation_status: ProtectedRegionValidationStatus | None = None
    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    savings_confidence: TokenSavingsClaimConfidence | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _resolve_run_id(
    receipt: CompressionReceipt,
    attribution: TokenOptimizationAttribution | None,
) -> str | None:
    if receipt.run_id is not None:
        return receipt.run_id
    if attribution is not None:
        return attribution.run_id
    return None


def _resolve_step_id(
    receipt: CompressionReceipt,
    attribution: TokenOptimizationAttribution | None,
) -> str | None:
    if receipt.step_id is not None:
        return receipt.step_id
    if attribution is not None:
        return attribution.step_id
    return None


def _merge_metadata(
    receipt: CompressionReceipt,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    if receipt.metadata:
        combined.update(receipt.metadata)
    if metadata:
        combined.update(metadata)
    return combined


def build_token_savings_telemetry_payload(
    *,
    receipt: CompressionReceipt,
    workflow_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TokenOptimizationTelemetryPayload:
    """Build a typed token-savings telemetry payload from a compression receipt."""
    measurement = receipt.measurement
    strategy = receipt.strategy
    attribution = receipt.attribution

    resolved_workflow_id = workflow_id
    if resolved_workflow_id is None and attribution is not None:
        resolved_workflow_id = attribution.workflow_id

    token_category: TokenCategory | None = None
    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    savings_confidence: TokenSavingsClaimConfidence | None = None

    if measurement is not None:
        token_category = measurement.category
        baseline_tokens = measurement.baseline_tokens
        optimized_tokens = measurement.optimized_tokens
        saved_tokens = measurement.saved_tokens
        saved_ratio = measurement.saved_ratio
        savings_confidence = measurement.confidence

    strategy_id: str | None = None
    mechanism: TokenOptimizationMechanism | None = None
    strategy_kind: TokenOptimizationStrategyKind | None = None
    plugin_id = receipt.plugin_id

    if strategy is not None:
        strategy_id = strategy.strategy_id
        mechanism = strategy.mechanism
        strategy_kind = strategy.kind
        if plugin_id is None:
            plugin_id = strategy.plugin_id

    if strategy_id is None and attribution is not None:
        strategy_id = attribution.strategy_id
    if plugin_id is None and attribution is not None:
        plugin_id = attribution.plugin_id
    if token_category is None and attribution is not None:
        token_category = attribution.token_category

    validation_status: ProtectedRegionValidationStatus | None = None
    if receipt.validation is not None:
        validation_status = receipt.validation.status

    return TokenOptimizationTelemetryPayload(
        event_type=TokenOptimizationTelemetryEventType.TOKEN_SAVINGS,
        receipt_id=receipt.receipt_id,
        run_id=_resolve_run_id(receipt, attribution),
        step_id=_resolve_step_id(receipt, attribution),
        workflow_id=resolved_workflow_id,
        tenant_id=attribution.tenant_id if attribution is not None else None,
        agent_id=attribution.agent_id if attribution is not None else None,
        model=attribution.model if attribution is not None else None,
        provider=attribution.provider if attribution is not None else None,
        runtime_profile=attribution.runtime_profile if attribution is not None else None,
        optimization_profile=(
            attribution.optimization_profile if attribution is not None else None
        ),
        source_type=receipt.source_type,
        token_category=token_category,
        strategy_id=strategy_id,
        mechanism=mechanism,
        strategy_kind=strategy_kind,
        plugin_id=plugin_id,
        decision=receipt.decision,
        fallback_used=receipt.fallback_used,
        bypass_reason=receipt.bypass_reason,
        validation_status=validation_status,
        baseline_tokens=baseline_tokens,
        optimized_tokens=optimized_tokens,
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        savings_confidence=savings_confidence,
        metadata=_merge_metadata(receipt, metadata),
    )


_FALLBACK_DECISIONS: frozenset[TokenOptimizationDecision] = frozenset(
    {
        TokenOptimizationDecision.FALLBACK,
        TokenOptimizationDecision.BYPASS,
        TokenOptimizationDecision.FAILED,
    }
)


def validate_token_savings_telemetry_payload(
    payload: TokenOptimizationTelemetryPayload,
) -> TokenOptimizationTelemetryValidationResult:
    """Validate a telemetry payload deterministically without exposing raw content."""
    failures: list[str] = []

    if payload.event_type is not TokenOptimizationTelemetryEventType.TOKEN_SAVINGS:
        failures.append("event_type must be token_optimization.savings")

    if not payload.receipt_id:
        failures.append("receipt_id must not be empty")

    if payload.source_type is None:
        failures.append("source_type must be present")

    if payload.decision is None:
        failures.append("decision must be present")

    has_token_fields = (
        payload.baseline_tokens is not None
        or payload.optimized_tokens is not None
        or payload.saved_tokens is not None
        or payload.saved_ratio is not None
    )

    if has_token_fields:
        if payload.baseline_tokens is None:
            failures.append("baseline_tokens must be present when token fields are set")
        elif payload.baseline_tokens < 0:
            failures.append("baseline_tokens must not be negative")

        if payload.optimized_tokens is None:
            failures.append("optimized_tokens must be present when token fields are set")
        elif payload.optimized_tokens < 0:
            failures.append("optimized_tokens must not be negative")

        if (
            payload.baseline_tokens is not None
            and payload.optimized_tokens is not None
            and payload.saved_tokens is not None
        ):
            expected_saved = payload.baseline_tokens - payload.optimized_tokens
            if payload.saved_tokens != expected_saved:
                failures.append("saved_tokens must equal baseline_tokens - optimized_tokens")

        if (
            payload.baseline_tokens is not None
            and payload.baseline_tokens > 0
            and payload.saved_ratio is not None
            and not 0.0 <= payload.saved_ratio <= 1.0
        ):
            failures.append("saved_ratio must be between 0.0 and 1.0 when baseline_tokens > 0")

    if payload.fallback_used and payload.decision not in _FALLBACK_DECISIONS:
        failures.append("decision must be fallback, bypass, or failed when fallback_used is true")

    if failures:
        return TokenOptimizationTelemetryValidationResult(
            status=TokenOptimizationTelemetryValidationStatus.FAILED,
            failures=tuple(failures),
        )

    if not has_token_fields:
        return TokenOptimizationTelemetryValidationResult(
            status=TokenOptimizationTelemetryValidationStatus.NOT_APPLICABLE,
        )

    return TokenOptimizationTelemetryValidationResult(
        status=TokenOptimizationTelemetryValidationStatus.PASSED,
    )


def _scalar_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    return None


def token_savings_payload_to_attributes(
    payload: TokenOptimizationTelemetryPayload,
) -> Mapping[str, Any]:
    """Prepare a safe namespaced attribute mapping for future HOS/domain-signal emission."""
    field_map: tuple[tuple[str, Any], ...] = (
        ("event_type", payload.event_type),
        ("receipt_id", payload.receipt_id),
        ("run_id", payload.run_id),
        ("step_id", payload.step_id),
        ("workflow_id", payload.workflow_id),
        ("source_type", payload.source_type),
        ("token_category", payload.token_category),
        ("strategy_id", payload.strategy_id),
        ("plugin_id", payload.plugin_id),
        ("decision", payload.decision),
        ("fallback_used", payload.fallback_used),
        ("validation_status", payload.validation_status),
        ("baseline_tokens", payload.baseline_tokens),
        ("optimized_tokens", payload.optimized_tokens),
        ("saved_tokens", payload.saved_tokens),
        ("saved_ratio", payload.saved_ratio),
        ("savings_confidence", payload.savings_confidence),
    )

    attributes: dict[str, Any] = {}
    for key, raw_value in field_map:
        scalar = _scalar_value(raw_value)
        if scalar is not None or raw_value is None:
            attributes[f"{_ATTRIBUTE_PREFIX}{key}"] = scalar
    return attributes


_FORBIDDEN_SUMMARY_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "original_content",
        "optimized_content",
        "raw_context",
        "raw_prompt",
        "raw_document",
        "tool_args",
        "chunks",
    }
)

_RECEIPT_DECISION_COUNTERS: dict[TokenOptimizationDecision, str] = {
    TokenOptimizationDecision.APPLY: "applied_count",
    TokenOptimizationDecision.BYPASS: "bypassed_count",
    TokenOptimizationDecision.FALLBACK: "fallback_count",
    TokenOptimizationDecision.FAILED: "failed_count",
}

_OUTCOME_STATUS_COUNTERS: dict[str, str] = {
    "applied": "applied_count",
    "bypassed": "bypassed_count",
    "fallback": "fallback_count",
    "unchanged": "unchanged_count",
}


def _safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {key: value for key, value in metadata.items() if key not in _FORBIDDEN_SUMMARY_METADATA_KEYS}


def _collect_deduplicated_receipts(
    *,
    receipts: Sequence[CompressionReceipt],
    tool_schema_outcomes: Sequence[ToolSchemaOptimizationOutcome],
    context_pack_outcomes: Sequence[ContextPackOptimizationOutcome],
) -> list[CompressionReceipt]:
    ordered: list[CompressionReceipt] = []
    seen_ids: set[str] = set()
    seen_positions: set[int] = set()

    def _add(receipt: CompressionReceipt) -> None:
        if receipt.receipt_id:
            if receipt.receipt_id in seen_ids:
                return
            seen_ids.add(receipt.receipt_id)
            ordered.append(receipt)
            return
        position = id(receipt)
        if position in seen_positions:
            return
        seen_positions.add(position)
        ordered.append(receipt)

    for receipt in receipts:
        _add(receipt)
    for outcome in tool_schema_outcomes:
        if outcome.receipt is not None:
            _add(outcome.receipt)
    for outcome in context_pack_outcomes:
        if outcome.receipt is not None:
            _add(outcome.receipt)
    return ordered


def _count_receipt_decision(counters: dict[str, int], decision: TokenOptimizationDecision) -> None:
    field_name = _RECEIPT_DECISION_COUNTERS.get(decision)
    if field_name is not None:
        counters[field_name] += 1


def _count_outcome_status(counters: dict[str, int], status: Any) -> None:
    status_value = status.value if isinstance(status, StrEnum) else str(status)
    field_name = _OUTCOME_STATUS_COUNTERS.get(status_value)
    if field_name is not None:
        counters[field_name] += 1


def _count_receipt_validation(counters: dict[str, int], receipt: CompressionReceipt) -> None:
    if receipt.validation is None:
        counters["validation_not_applicable_count"] += 1
        return
    if receipt.validation.status is ProtectedRegionValidationStatus.PASSED:
        counters["validation_passed_count"] += 1
    elif receipt.validation.status is ProtectedRegionValidationStatus.FAILED:
        counters["validation_failed_count"] += 1
    else:
        counters["validation_not_applicable_count"] += 1


def _aggregate_token_measurements(receipts: Sequence[CompressionReceipt]) -> tuple[int, int, int, float]:
    baseline_total = 0
    optimized_total = 0
    measured = False
    for receipt in receipts:
        if receipt.measurement is None:
            continue
        measured = True
        baseline_total += receipt.measurement.baseline_tokens
        optimized_total += receipt.measurement.optimized_tokens
    if not measured:
        return 0, 0, 0, 0.0
    saved_total = baseline_total - optimized_total
    saved_ratio = saved_total / baseline_total if baseline_total > 0 else 0.0
    return baseline_total, optimized_total, saved_total, saved_ratio


def _outcome_contributes_status_count(
    outcome: ToolSchemaOptimizationOutcome | ContextPackOptimizationOutcome,
    seen_receipt_ids: set[str],
) -> bool:
    if outcome.receipt is None:
        return True
    if not outcome.receipt.receipt_id:
        return True
    return outcome.receipt.receipt_id not in seen_receipt_ids


def build_token_optimization_counter_snapshot(
    *,
    receipts: Sequence[CompressionReceipt] = (),
    resolved_policies: Sequence[ResolvedOutputPolicy] = (),
    tool_schema_outcomes: Sequence[ToolSchemaOptimizationOutcome] = (),
    context_pack_outcomes: Sequence[ContextPackOptimizationOutcome] = (),
    metadata: Mapping[str, Any] | None = None,
) -> TokenOptimizationCounterSnapshot:
    """Build safe aggregate counters from token optimization helper results."""
    deduplicated_receipts = _collect_deduplicated_receipts(
        receipts=receipts,
        tool_schema_outcomes=tool_schema_outcomes,
        context_pack_outcomes=context_pack_outcomes,
    )
    seen_receipt_ids = {
        receipt.receipt_id for receipt in deduplicated_receipts if receipt.receipt_id
    }

    counters: dict[str, int] = {
        "applied_count": 0,
        "bypassed_count": 0,
        "fallback_count": 0,
        "failed_count": 0,
        "unchanged_count": 0,
        "validation_passed_count": 0,
        "validation_failed_count": 0,
        "validation_not_applicable_count": 0,
        "fallback_used_count": 0,
        "receipts_with_measurement_count": 0,
        "tool_schema_changed_count": 0,
        "context_pack_changed_count": 0,
    }

    for receipt in deduplicated_receipts:
        _count_receipt_decision(counters, receipt.decision)
        _count_receipt_validation(counters, receipt)
        if receipt.fallback_used:
            counters["fallback_used_count"] += 1
        if receipt.measurement is not None:
            counters["receipts_with_measurement_count"] += 1

    for outcome in tool_schema_outcomes:
        if _outcome_contributes_status_count(outcome, seen_receipt_ids):
            _count_outcome_status(counters, outcome.status)
        if outcome.changed:
            counters["tool_schema_changed_count"] += 1

    for outcome in context_pack_outcomes:
        if _outcome_contributes_status_count(outcome, seen_receipt_ids):
            _count_outcome_status(counters, outcome.status)
        if outcome.changed:
            counters["context_pack_changed_count"] += 1

    baseline_tokens, optimized_tokens, saved_tokens, saved_ratio = _aggregate_token_measurements(
        deduplicated_receipts,
    )

    resolved_enabled_count = sum(1 for policy in resolved_policies if policy.enabled)
    resolved_disabled_count = sum(
        1 for policy in resolved_policies if policy.status is OutputPolicyResolutionStatus.DISABLED
    )
    resolved_defaulted_count = sum(
        1 for policy in resolved_policies if policy.status is OutputPolicyResolutionStatus.DEFAULTED
    )

    snapshot_metadata = _safe_metadata(metadata)
    if resolved_policies:
        snapshot_metadata = {
            **snapshot_metadata,
            "resolved_policies_enabled_count": resolved_enabled_count,
            "resolved_policies_disabled_count": resolved_disabled_count,
            "resolved_policies_defaulted_count": resolved_defaulted_count,
        }

    return TokenOptimizationCounterSnapshot(
        total_receipts=len(deduplicated_receipts),
        total_resolved_policies=len(resolved_policies),
        total_tool_schema_outcomes=len(tool_schema_outcomes),
        total_context_pack_outcomes=len(context_pack_outcomes),
        applied_count=counters["applied_count"],
        bypassed_count=counters["bypassed_count"],
        fallback_count=counters["fallback_count"],
        failed_count=counters["failed_count"],
        unchanged_count=counters["unchanged_count"],
        validation_passed_count=counters["validation_passed_count"],
        validation_failed_count=counters["validation_failed_count"],
        validation_not_applicable_count=counters["validation_not_applicable_count"],
        fallback_used_count=counters["fallback_used_count"],
        receipts_with_measurement_count=counters["receipts_with_measurement_count"],
        baseline_tokens=baseline_tokens,
        optimized_tokens=optimized_tokens,
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        tool_schema_changed_count=counters["tool_schema_changed_count"],
        context_pack_changed_count=counters["context_pack_changed_count"],
        metadata=snapshot_metadata,
    )


def _collect_summary_source_types(
    deduplicated_receipts: Sequence[CompressionReceipt],
    tool_schema_outcomes: Sequence[ToolSchemaOptimizationOutcome],
    context_pack_outcomes: Sequence[ContextPackOptimizationOutcome],
) -> tuple[TokenOptimizationSourceType, ...]:
    source_types: list[TokenOptimizationSourceType] = []
    seen: set[TokenOptimizationSourceType] = set()

    def _add(source_type: TokenOptimizationSourceType | None) -> None:
        if source_type is None or source_type in seen:
            return
        seen.add(source_type)
        source_types.append(source_type)

    for receipt in deduplicated_receipts:
        _add(receipt.source_type)
    for outcome in tool_schema_outcomes:
        _add(outcome.request.source_type)
    for outcome in context_pack_outcomes:
        _add(outcome.request.source_type)
    return tuple(source_types)


def _collect_summary_strategy_ids(receipts: Sequence[CompressionReceipt]) -> tuple[str, ...]:
    strategy_ids: list[str] = []
    seen: set[str] = set()
    for receipt in receipts:
        strategy_id: str | None = None
        if receipt.strategy is not None:
            strategy_id = receipt.strategy.strategy_id
        elif receipt.attribution is not None:
            strategy_id = receipt.attribution.strategy_id
        if strategy_id and strategy_id not in seen:
            seen.add(strategy_id)
            strategy_ids.append(strategy_id)
    return tuple(strategy_ids)


def _collect_summary_plugin_ids(receipts: Sequence[CompressionReceipt]) -> tuple[str, ...]:
    plugin_ids: list[str] = []
    seen: set[str] = set()
    for receipt in receipts:
        plugin_id = receipt.plugin_id
        if plugin_id is None and receipt.strategy is not None:
            plugin_id = receipt.strategy.plugin_id
        if plugin_id is None and receipt.attribution is not None:
            plugin_id = receipt.attribution.plugin_id
        if plugin_id and plugin_id not in seen:
            seen.add(plugin_id)
            plugin_ids.append(plugin_id)
    return tuple(plugin_ids)


def _collect_summary_receipt_ids(receipts: Sequence[CompressionReceipt]) -> tuple[str, ...]:
    receipt_ids: list[str] = []
    seen: set[str] = set()
    for receipt in receipts:
        if receipt.receipt_id and receipt.receipt_id not in seen:
            seen.add(receipt.receipt_id)
            receipt_ids.append(receipt.receipt_id)
    return tuple(receipt_ids)


def _collect_summary_source_kinds(
    *,
    receipts: Sequence[CompressionReceipt],
    resolved_policies: Sequence[ResolvedOutputPolicy],
    tool_schema_outcomes: Sequence[ToolSchemaOptimizationOutcome],
    context_pack_outcomes: Sequence[ContextPackOptimizationOutcome],
) -> tuple[TokenOptimizationTelemetrySource, ...]:
    sources: list[TokenOptimizationTelemetrySource] = []
    if receipts:
        sources.append(TokenOptimizationTelemetrySource.COMPRESSION_RECEIPT)
    if resolved_policies:
        sources.append(TokenOptimizationTelemetrySource.RESOLVED_OUTPUT_POLICY)
    if tool_schema_outcomes:
        sources.append(TokenOptimizationTelemetrySource.TOOL_SCHEMA_OUTCOME)
    if context_pack_outcomes:
        sources.append(TokenOptimizationTelemetrySource.CONTEXT_PACK_OUTCOME)
    return tuple(sources)


def build_token_optimization_telemetry_summary(
    *,
    receipts: Sequence[CompressionReceipt] = (),
    resolved_policies: Sequence[ResolvedOutputPolicy] = (),
    tool_schema_outcomes: Sequence[ToolSchemaOptimizationOutcome] = (),
    context_pack_outcomes: Sequence[ContextPackOptimizationOutcome] = (),
    workflow_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TokenOptimizationTelemetrySummary:
    """Build a typed telemetry summary payload from token optimization helper results."""
    deduplicated_receipts = _collect_deduplicated_receipts(
        receipts=receipts,
        tool_schema_outcomes=tool_schema_outcomes,
        context_pack_outcomes=context_pack_outcomes,
    )
    snapshot = build_token_optimization_counter_snapshot(
        receipts=receipts,
        resolved_policies=resolved_policies,
        tool_schema_outcomes=tool_schema_outcomes,
        context_pack_outcomes=context_pack_outcomes,
        metadata=metadata,
    )
    summary_metadata = _safe_metadata(metadata)
    source_kinds = _collect_summary_source_kinds(
        receipts=deduplicated_receipts,
        resolved_policies=resolved_policies,
        tool_schema_outcomes=tool_schema_outcomes,
        context_pack_outcomes=context_pack_outcomes,
    )
    if source_kinds:
        summary_metadata = {
            **summary_metadata,
            "telemetry_sources": ",".join(source.value for source in source_kinds),
        }

    return TokenOptimizationTelemetrySummary(
        event_type=TokenOptimizationTelemetryEventType.TOKEN_OPTIMIZATION_SUMMARY,
        workflow_id=workflow_id,
        snapshot=snapshot,
        source_types=_collect_summary_source_types(
            deduplicated_receipts,
            tool_schema_outcomes,
            context_pack_outcomes,
        ),
        strategy_ids=_collect_summary_strategy_ids(deduplicated_receipts),
        plugin_ids=_collect_summary_plugin_ids(deduplicated_receipts),
        receipt_ids=_collect_summary_receipt_ids(deduplicated_receipts),
        metadata=summary_metadata,
    )


def validate_token_optimization_telemetry_summary(
    summary: TokenOptimizationTelemetrySummary,
) -> TokenOptimizationTelemetrySummaryValidationResult:
    """Validate a telemetry summary deterministically without exposing raw content."""
    failures: list[str] = []

    if summary.event_type is not TokenOptimizationTelemetryEventType.TOKEN_OPTIMIZATION_SUMMARY:
        failures.append("event_type must be token_optimization.summary")

    snapshot = summary.snapshot
    non_negative_fields = (
        ("total_receipts", snapshot.total_receipts),
        ("total_resolved_policies", snapshot.total_resolved_policies),
        ("total_tool_schema_outcomes", snapshot.total_tool_schema_outcomes),
        ("total_context_pack_outcomes", snapshot.total_context_pack_outcomes),
        ("applied_count", snapshot.applied_count),
        ("bypassed_count", snapshot.bypassed_count),
        ("fallback_count", snapshot.fallback_count),
        ("failed_count", snapshot.failed_count),
        ("unchanged_count", snapshot.unchanged_count),
        ("validation_passed_count", snapshot.validation_passed_count),
        ("validation_failed_count", snapshot.validation_failed_count),
        ("validation_not_applicable_count", snapshot.validation_not_applicable_count),
        ("fallback_used_count", snapshot.fallback_used_count),
        ("receipts_with_measurement_count", snapshot.receipts_with_measurement_count),
        ("baseline_tokens", snapshot.baseline_tokens),
        ("optimized_tokens", snapshot.optimized_tokens),
        ("saved_tokens", snapshot.saved_tokens),
        ("tool_schema_changed_count", snapshot.tool_schema_changed_count),
        ("context_pack_changed_count", snapshot.context_pack_changed_count),
    )
    for field_name, value in non_negative_fields:
        if value < 0:
            failures.append(f"{field_name} must not be negative")

    if snapshot.receipts_with_measurement_count > 0 or snapshot.baseline_tokens > 0:
        expected_saved = snapshot.baseline_tokens - snapshot.optimized_tokens
        if snapshot.saved_tokens != expected_saved:
            failures.append("saved_tokens must equal baseline_tokens - optimized_tokens")
        if snapshot.baseline_tokens > 0 and not 0.0 <= snapshot.saved_ratio <= 1.0:
            failures.append("saved_ratio must be between 0.0 and 1.0 when baseline_tokens > 0")

    if summary.receipt_ids:
        if len(summary.receipt_ids) != len(set(summary.receipt_ids)):
            failures.append("receipt_ids must be unique")

    for metadata_map in (summary.metadata, snapshot.metadata):
        for key in metadata_map:
            if key in _FORBIDDEN_SUMMARY_METADATA_KEYS:
                failures.append(f"metadata must not contain forbidden key: {key}")

    if failures:
        return TokenOptimizationTelemetrySummaryValidationResult(
            status=TokenOptimizationTelemetrySummaryValidationStatus.FAILED,
            failures=tuple(failures),
        )

    return TokenOptimizationTelemetrySummaryValidationResult(
        status=TokenOptimizationTelemetrySummaryValidationStatus.PASSED,
    )


def token_optimization_summary_to_attributes(
    summary: TokenOptimizationTelemetrySummary,
) -> Mapping[str, Any]:
    """Prepare a safe namespaced attribute mapping for a telemetry summary."""
    snapshot = summary.snapshot
    field_map: tuple[tuple[str, Any], ...] = (
        ("event_type", summary.event_type),
        ("workflow_id", summary.workflow_id),
        ("total_receipts", snapshot.total_receipts),
        ("total_resolved_policies", snapshot.total_resolved_policies),
        ("total_tool_schema_outcomes", snapshot.total_tool_schema_outcomes),
        ("total_context_pack_outcomes", snapshot.total_context_pack_outcomes),
        ("applied_count", snapshot.applied_count),
        ("bypassed_count", snapshot.bypassed_count),
        ("fallback_count", snapshot.fallback_count),
        ("failed_count", snapshot.failed_count),
        ("unchanged_count", snapshot.unchanged_count),
        ("validation_passed_count", snapshot.validation_passed_count),
        ("validation_failed_count", snapshot.validation_failed_count),
        ("validation_not_applicable_count", snapshot.validation_not_applicable_count),
        ("fallback_used_count", snapshot.fallback_used_count),
        ("receipts_with_measurement_count", snapshot.receipts_with_measurement_count),
        ("baseline_tokens", snapshot.baseline_tokens),
        ("optimized_tokens", snapshot.optimized_tokens),
        ("saved_tokens", snapshot.saved_tokens),
        ("saved_ratio", snapshot.saved_ratio),
        ("tool_schema_changed_count", snapshot.tool_schema_changed_count),
        ("context_pack_changed_count", snapshot.context_pack_changed_count),
    )

    attributes: dict[str, Any] = {}
    for key, raw_value in field_map:
        scalar = _scalar_value(raw_value)
        if scalar is not None or raw_value is None:
            attributes[f"{_ATTRIBUTE_PREFIX}{key}"] = scalar

    for key, raw_value in summary.metadata.items():
        if key in _FORBIDDEN_SUMMARY_METADATA_KEYS:
            continue
        scalar = _scalar_value(raw_value)
        if scalar is not None:
            attributes[f"{_ATTRIBUTE_PREFIX}metadata.{key}"] = scalar

    for key, raw_value in snapshot.metadata.items():
        if key in _FORBIDDEN_SUMMARY_METADATA_KEYS:
            continue
        scalar = _scalar_value(raw_value)
        if scalar is not None:
            attributes[f"{_ATTRIBUTE_PREFIX}snapshot.metadata.{key}"] = scalar

    return attributes
