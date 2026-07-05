# © Artur Czarnecki. All rights reserved.

"""Built-in char-budget context packing prototype layer (TOKEN-OPT-3D).

This is a **char-budget prototype** packing layer, not provider-aware token-budget
optimization. ``budget_unit`` is always ``"chars"``; ``max_chars`` is an estimated
character budget only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ContextFragmentPriority,
    ContextPackingDecisionKind,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationMechanism,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.protected_regions import validate_protected_regions

_LAYER_ID = "builtin.budget_aware_context_packing"
_BUDGET_UNIT = "chars"
_WHITESPACE_RE = re.compile(r"\s+")

# ranking_pruning is the closest existing strategy kind for tier-based packing drops.
_BUILTIN_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id=_LAYER_ID,
    mechanism=TokenOptimizationMechanism.RAG_CONTEXT_PACK_COMPRESSION,
    kind=TokenOptimizationStrategyKind.RANKING_PRUNING,
    safety_class=StrategySafetyClass.LOSSLESS,
    version="1",
)

_SUPPORTED_SOURCE_TYPES = (
    TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
)


@dataclass(frozen=True, slots=True)
class BudgetAwarePackingFragment:
    """Layer-local fragment model for char-budget prototype packing."""

    fragment_id: str
    content: str
    priority: ContextFragmentPriority
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fragment_id:
            raise ValueError("fragment_id cannot be empty")
        if self.content is None:
            raise ValueError("content must not be None")
        if not isinstance(self.priority, ContextFragmentPriority):
            raise ValueError("priority must be ContextFragmentPriority")


@dataclass(frozen=True, slots=True)
class BudgetAwarePackingInput:
    """Layer-local structured packing payload (prototype adapter via request metadata)."""

    fragments: tuple[BudgetAwarePackingFragment, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for fragment in self.fragments:
            if fragment.fragment_id in seen:
                raise ValueError("fragment ids must be unique")
            seen.add(fragment.fragment_id)


@dataclass(frozen=True, slots=True)
class BudgetAwareContextPackingLayerConfig:
    """Pipeline-level defaults for char-budget prototype context packing."""

    max_chars: int
    compact_compressible_whitespace: bool = True
    include_droppable_when_budget_available: bool = False
    separator: str = "\n"

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if self.separator is None:
            raise ValueError("separator must not be None")


@dataclass(frozen=True, slots=True)
class _FragmentOutcome:
    fragment_id: str
    priority: ContextFragmentPriority
    decision: ContextPackingDecisionKind
    original_chars: int
    output_chars: int
    reason: str
    output_content: str | None = None


class BudgetAwareContextPackingLayer:
    """Deterministic char-budget prototype packing as a standalone optimization layer."""

    def __init__(
        self,
        *,
        config: BudgetAwareContextPackingLayerConfig,
    ) -> None:
        self._config = config

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return TokenOptimizationLayerDescriptor(
            layer_id=_LAYER_ID,
            name="Budget-Aware Context Packing",
            version="1",
            strategy=_BUILTIN_STRATEGY,
            supported_source_types=_SUPPORTED_SOURCE_TYPES,
            safety_class=StrategySafetyClass.LOSSLESS,
            built_in=True,
            requires_validation=True,
        )

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        base_config = self._config
        effective_config = base_config
        config_overrides: dict[str, Any] = {}

        if request.source_type not in _SUPPORTED_SOURCE_TYPES:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE,
            )

        if not _policy_allows_optimization(request.policy):
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=_policy_bypass_reason(request.policy),
            )

        packing_input = _extract_packing_input(request.metadata)
        if packing_input is None:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=TokenOptimizationBypassReason.NOT_APPLICABLE,
            )

        if not packing_input.fragments:
            return self._bypass_result(
                request=request,
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                bypass_reason=TokenOptimizationBypassReason.NOT_APPLICABLE,
            )

        outcomes, fallback_reason = _pack_fragments(
            packing_input.fragments,
            effective_config,
        )
        if fallback_reason == "must_keep_exceeds_char_budget":
            must_keep_chars = _must_keep_chars(packing_input.fragments, effective_config.separator)
            metadata = _build_metadata(
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                fragments=packing_input.fragments,
                outcomes=outcomes,
                final_chars=len(request.current_content),
                char_budget_satisfied=False,
                fallback_reason=fallback_reason,
                must_keep_chars=must_keep_chars,
            )
            return TokenOptimizationLayerResult(
                layer_id=_LAYER_ID,
                output_content=request.current_content,
                decision=TokenOptimizationLayerDecision.FALLBACK,
                receipt_metadata=metadata,
                fallback_used=True,
                strategy=_BUILTIN_STRATEGY,
                metadata=metadata,
            )

        output_content = _assemble_output(
            packing_input.fragments,
            outcomes,
            effective_config.separator,
        )
        final_chars = len(output_content)

        validation = validate_protected_regions(
            request.current_content,
            output_content,
        )
        if validation.status is ProtectedRegionValidationStatus.FAILED:
            metadata = _build_metadata(
                base_config=base_config,
                effective_config=effective_config,
                config_overrides=config_overrides,
                fragments=packing_input.fragments,
                outcomes=outcomes,
                final_chars=len(request.current_content),
                char_budget_satisfied=final_chars <= effective_config.max_chars,
                fallback_reason="protected_region_validation_failed",
                must_keep_chars=_must_keep_chars(
                    packing_input.fragments,
                    effective_config.separator,
                ),
            )
            return TokenOptimizationLayerResult(
                layer_id=_LAYER_ID,
                output_content=request.current_content,
                decision=TokenOptimizationLayerDecision.FALLBACK,
                validation=validation,
                receipt_metadata=metadata,
                fallback_used=True,
                strategy=_BUILTIN_STRATEGY,
                metadata=metadata,
            )

        metadata = _build_metadata(
            base_config=base_config,
            effective_config=effective_config,
            config_overrides=config_overrides,
            fragments=packing_input.fragments,
            outcomes=outcomes,
            final_chars=final_chars,
            char_budget_satisfied=final_chars <= effective_config.max_chars,
            must_keep_chars=_must_keep_chars(
                packing_input.fragments,
                effective_config.separator,
            ),
        )
        return TokenOptimizationLayerResult(
            layer_id=_LAYER_ID,
            output_content=output_content,
            decision=TokenOptimizationLayerDecision.APPLY,
            validation=validation,
            receipt_metadata=metadata,
            strategy=_BUILTIN_STRATEGY,
            metadata=metadata,
        )

    def _bypass_result(
        self,
        *,
        request: TokenOptimizationLayerRequest,
        base_config: BudgetAwareContextPackingLayerConfig,
        effective_config: BudgetAwareContextPackingLayerConfig,
        config_overrides: dict[str, Any],
        bypass_reason: TokenOptimizationBypassReason,
    ) -> TokenOptimizationLayerResult:
        metadata = _build_metadata(
            base_config=base_config,
            effective_config=effective_config,
            config_overrides=config_overrides,
            fragments=(),
            outcomes=(),
            final_chars=len(request.current_content),
            char_budget_satisfied=True,
            must_keep_chars=0,
        )
        return TokenOptimizationLayerResult(
            layer_id=_LAYER_ID,
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.BYPASS,
            bypass_reason=bypass_reason,
            strategy=_BUILTIN_STRATEGY,
            receipt_metadata=metadata,
            metadata=metadata,
        )


def _extract_packing_input(
    metadata: Mapping[str, Any],
) -> BudgetAwarePackingInput | None:
    raw = metadata.get("packing_input")
    if raw is None:
        return None
    if not isinstance(raw, BudgetAwarePackingInput):
        return None
    return raw


def _compact_whitespace(content: str) -> str:
    return _WHITESPACE_RE.sub(" ", content.strip())


def _assembled_length(parts: tuple[str, ...], separator: str) -> int:
    if not parts:
        return 0
    total = sum(len(part) for part in parts)
    if len(parts) > 1:
        total += len(separator) * (len(parts) - 1)
    return total


def _must_keep_chars(
    fragments: tuple[BudgetAwarePackingFragment, ...],
    separator: str,
) -> int:
    must_keep_contents = tuple(
        fragment.content
        for fragment in fragments
        if fragment.priority is ContextFragmentPriority.MUST_KEEP
    )
    return _assembled_length(must_keep_contents, separator)


def _selected_contents(
    fragments: tuple[BudgetAwarePackingFragment, ...],
    selected_ids: set[str],
    outcomes: dict[str, _FragmentOutcome],
) -> tuple[str, ...]:
    return tuple(
        outcomes[fragment.fragment_id].output_content or ""
        for fragment in fragments
        if fragment.fragment_id in selected_ids
    )


def _would_fit(
    fragments: tuple[BudgetAwarePackingFragment, ...],
    selected_ids: set[str],
    outcomes: dict[str, _FragmentOutcome],
    new_content: str,
    *,
    max_chars: int,
    separator: str,
) -> bool:
    current = _selected_contents(fragments, selected_ids, outcomes)
    trial = current + (new_content,)
    return _assembled_length(trial, separator) <= max_chars


def _pack_fragments(
    fragments: tuple[BudgetAwarePackingFragment, ...],
    config: BudgetAwareContextPackingLayerConfig,
) -> tuple[tuple[_FragmentOutcome, ...], str | None]:
    outcomes: dict[str, _FragmentOutcome] = {}
    selected_ids: set[str] = set()

    must_keep_chars = _must_keep_chars(fragments, config.separator)
    if must_keep_chars > config.max_chars:
        for fragment in fragments:
            if fragment.priority is ContextFragmentPriority.MUST_KEEP:
                outcomes[fragment.fragment_id] = _FragmentOutcome(
                    fragment_id=fragment.fragment_id,
                    priority=fragment.priority,
                    decision=ContextPackingDecisionKind.FALLBACK,
                    original_chars=len(fragment.content),
                    output_chars=len(fragment.content),
                    reason="must_keep_exceeds_char_budget",
                    output_content=fragment.content,
                )
                selected_ids.add(fragment.fragment_id)
            else:
                outcomes[fragment.fragment_id] = _FragmentOutcome(
                    fragment_id=fragment.fragment_id,
                    priority=fragment.priority,
                    decision=ContextPackingDecisionKind.BYPASS,
                    original_chars=len(fragment.content),
                    output_chars=0,
                    reason="must_keep_exceeds_char_budget",
                )
        return tuple(outcomes[fragment.fragment_id] for fragment in fragments), (
            "must_keep_exceeds_char_budget"
        )

    for fragment in fragments:
        if fragment.priority is not ContextFragmentPriority.MUST_KEEP:
            continue
        outcomes[fragment.fragment_id] = _FragmentOutcome(
            fragment_id=fragment.fragment_id,
            priority=fragment.priority,
            decision=ContextPackingDecisionKind.KEEP,
            original_chars=len(fragment.content),
            output_chars=len(fragment.content),
            reason="must_keep_required",
            output_content=fragment.content,
        )
        selected_ids.add(fragment.fragment_id)

    for fragment in fragments:
        if fragment.priority is not ContextFragmentPriority.HIGH_PRIORITY:
            continue
        if _would_fit(
            fragments,
            selected_ids,
            outcomes,
            fragment.content,
            max_chars=config.max_chars,
            separator=config.separator,
        ):
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.KEEP,
                original_chars=len(fragment.content),
                output_chars=len(fragment.content),
                reason="high_priority_included",
                output_content=fragment.content,
            )
            selected_ids.add(fragment.fragment_id)
        else:
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.DROP,
                original_chars=len(fragment.content),
                output_chars=0,
                reason="char_budget_exceeded",
            )

    for fragment in fragments:
        if fragment.priority is not ContextFragmentPriority.COMPRESSIBLE:
            continue
        if _would_fit(
            fragments,
            selected_ids,
            outcomes,
            fragment.content,
            max_chars=config.max_chars,
            separator=config.separator,
        ):
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.KEEP,
                original_chars=len(fragment.content),
                output_chars=len(fragment.content),
                reason="compressible_included",
                output_content=fragment.content,
            )
            selected_ids.add(fragment.fragment_id)
            continue

        compacted = (
            _compact_whitespace(fragment.content)
            if config.compact_compressible_whitespace
            else None
        )
        if (
            compacted is not None
            and compacted != fragment.content
            and _would_fit(
                fragments,
                selected_ids,
                outcomes,
                compacted,
                max_chars=config.max_chars,
                separator=config.separator,
            )
        ):
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.COMPACT,
                original_chars=len(fragment.content),
                output_chars=len(compacted),
                reason="compressible_whitespace_compacted",
                output_content=compacted,
            )
            selected_ids.add(fragment.fragment_id)
            continue

        outcomes[fragment.fragment_id] = _FragmentOutcome(
            fragment_id=fragment.fragment_id,
            priority=fragment.priority,
            decision=ContextPackingDecisionKind.DROP,
            original_chars=len(fragment.content),
            output_chars=0,
            reason="char_budget_exceeded",
        )

    for fragment in fragments:
        if fragment.priority is not ContextFragmentPriority.DROPPABLE:
            continue
        if not config.include_droppable_when_budget_available:
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.DROP,
                original_chars=len(fragment.content),
                output_chars=0,
                reason="droppable_excluded_by_default",
            )
            continue
        if _would_fit(
            fragments,
            selected_ids,
            outcomes,
            fragment.content,
            max_chars=config.max_chars,
            separator=config.separator,
        ):
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.KEEP,
                original_chars=len(fragment.content),
                output_chars=len(fragment.content),
                reason="droppable_included",
                output_content=fragment.content,
            )
            selected_ids.add(fragment.fragment_id)
        else:
            outcomes[fragment.fragment_id] = _FragmentOutcome(
                fragment_id=fragment.fragment_id,
                priority=fragment.priority,
                decision=ContextPackingDecisionKind.DROP,
                original_chars=len(fragment.content),
                output_chars=0,
                reason="char_budget_exceeded",
            )

    return tuple(outcomes[fragment.fragment_id] for fragment in fragments), None


def _assemble_output(
    fragments: tuple[BudgetAwarePackingFragment, ...],
    outcomes: tuple[_FragmentOutcome, ...],
    separator: str,
) -> str:
    outcome_by_id = {outcome.fragment_id: outcome for outcome in outcomes}
    parts = [
        outcome_by_id[fragment.fragment_id].output_content or ""
        for fragment in fragments
        if outcome_by_id[fragment.fragment_id].decision
        in (
            ContextPackingDecisionKind.KEEP,
            ContextPackingDecisionKind.COMPACT,
        )
    ]
    return separator.join(parts)


def _policy_allows_optimization(policy: TokenOptimizationPolicy) -> bool:
    if not policy.enabled:
        return False
    return policy.profile not in (
        TokenOptimizationProfile.OFF,
        TokenOptimizationProfile.MEASURE_ONLY,
    )


def _policy_bypass_reason(
    policy: TokenOptimizationPolicy,
) -> TokenOptimizationBypassReason:
    if not policy.enabled:
        return TokenOptimizationBypassReason.DISABLED
    return TokenOptimizationBypassReason.POLICY_DISALLOWED


def _config_mapping(config: BudgetAwareContextPackingLayerConfig) -> dict[str, Any]:
    return dict(asdict(config))


def _outcome_to_decision_dict(outcome: _FragmentOutcome) -> dict[str, Any]:
    return {
        "fragment_id": outcome.fragment_id,
        "priority": outcome.priority.value,
        "decision": outcome.decision.value,
        "original_chars": outcome.original_chars,
        "output_chars": outcome.output_chars,
        "reason": outcome.reason,
    }


def _build_metadata(
    *,
    base_config: BudgetAwareContextPackingLayerConfig,
    effective_config: BudgetAwareContextPackingLayerConfig,
    config_overrides: dict[str, Any],
    fragments: tuple[BudgetAwarePackingFragment, ...],
    outcomes: tuple[_FragmentOutcome, ...],
    final_chars: int,
    char_budget_satisfied: bool,
    must_keep_chars: int,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    selected = [
        outcome
        for outcome in outcomes
        if outcome.decision
        in (ContextPackingDecisionKind.KEEP, ContextPackingDecisionKind.COMPACT)
    ]
    dropped = [
        outcome
        for outcome in outcomes
        if outcome.decision is ContextPackingDecisionKind.DROP
    ]
    compacted = [
        outcome
        for outcome in outcomes
        if outcome.decision is ContextPackingDecisionKind.COMPACT
    ]

    dropped_chars = sum(outcome.original_chars for outcome in dropped)
    compacted_chars = sum(
        outcome.original_chars - outcome.output_chars for outcome in compacted
    )
    total_input_chars = _assembled_length(
        tuple(fragment.content for fragment in fragments),
        effective_config.separator,
    )
    saved_chars = max(0, total_input_chars - final_chars)

    packing_decisions = [_outcome_to_decision_dict(outcome) for outcome in outcomes]

    metadata: dict[str, Any] = {
        "base_config": _config_mapping(base_config),
        "effective_config": _config_mapping(effective_config),
        "config_overrides": dict(config_overrides),
        "budget_unit": _BUDGET_UNIT,
        "max_chars": effective_config.max_chars,
        "input_fragment_count": len(fragments),
        "selected_fragment_count": len(selected),
        "dropped_fragment_count": len(dropped),
        "compacted_fragment_count": len(compacted),
        "must_keep_chars": must_keep_chars,
        "final_chars": final_chars,
        "char_budget_satisfied": char_budget_satisfied,
        "packing_decisions": packing_decisions,
        "saved_chars": saved_chars,
        "dropped_chars": dropped_chars,
        "compacted_chars": compacted_chars,
    }
    if fallback_reason is not None:
        metadata["fallback_reason"] = fallback_reason
    return metadata
