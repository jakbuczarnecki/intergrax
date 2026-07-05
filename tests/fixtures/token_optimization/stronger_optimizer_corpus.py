# © Artur Czarnecki. All rights reserved.

"""Internal synthetic corpus and evaluation helpers for stronger optimizer layers (TOKEN-OBS-3E-F)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ContextFragmentPriority,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerRequest,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers.budget_aware_packing import (
    BudgetAwareContextPackingLayer,
    BudgetAwareContextPackingLayerConfig,
    BudgetAwarePackingFragment,
    BudgetAwarePackingInput,
)
from intergrax.runtime.token_optimization.layers.exact_deduplication import (
    ExactDeduplicationLayer,
)

SYNTHETIC_CORPUS_MARKER = "SYNTHETIC_EVAL_CORPUS_V1"

ALLOWED_SYNTHETIC_MARKERS: tuple[str, ...] = (
    "PROJECT-ALPHA-001",
    "INVOICE-2026-SYNTH-001",
    "TENANT-SYNTH-A",
    "API_KEY_PLACEHOLDER",
    "CUSTOMER_EXAMPLE",
    "TRACE-SYNTH-001",
)

STRATEGY_DEDUPLICATION = "deduplication"
STRATEGY_BUDGET_AWARE_PACKING = "budget_aware_packing"
STRATEGY_FALLBACK = "fallback"
STRATEGY_NO_OP = "no_op"

_FORBIDDEN_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"Bearer\s+eyJ[a-zA-Z0-9_-]+"),
    re.compile(r"password\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
)

_TOKEN_NAMED_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "saved_tokens",
        "total_saved_tokens",
        "optimized_tokens",
        "original_tokens",
        "baseline_tokens",
        "total_original_tokens",
        "total_optimized_tokens",
    }
)


@dataclass(frozen=True, slots=True)
class StrongerOptimizerCorpusFragment:
    fragment_id: str
    content: str
    priority: ContextFragmentPriority


@dataclass(frozen=True, slots=True)
class StrongerOptimizerExpectedBehavior:
    dedupe_applicable: bool
    packing_applicable: bool
    fallback_expected: bool
    no_op_expected: bool
    protected_region_required: bool
    expected_primary_strategy: str


@dataclass(frozen=True, slots=True)
class StrongerOptimizerCorpusCase:
    case_id: str
    title: str
    source_type: TokenOptimizationSourceType
    current_content: str
    fragments: tuple[StrongerOptimizerCorpusFragment, ...]
    max_chars: int | None
    expected: StrongerOptimizerExpectedBehavior
    safety_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrongerOptimizerEvaluationResult:
    case_id: str
    baseline_chars: int
    stronger_chars: int
    total_saved_chars: int
    strategy_savings: Mapping[str, int]
    decisions: tuple[Mapping[str, object], ...]
    fallback_used: bool
    raw_content_in_report: bool


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
    )


def _assemble_fragments(
    fragments: Sequence[StrongerOptimizerCorpusFragment],
    *,
    separator: str = "\n",
) -> str:
    if not fragments:
        return ""
    return separator.join(fragment.content for fragment in fragments)


def _to_packing_fragments(
    fragments: Sequence[StrongerOptimizerCorpusFragment],
) -> tuple[BudgetAwarePackingFragment, ...]:
    return tuple(
        BudgetAwarePackingFragment(
            fragment_id=fragment.fragment_id,
            content=fragment.content,
            priority=fragment.priority,
        )
        for fragment in fragments
    )


def _empty_strategy_savings() -> dict[str, int]:
    return {
        STRATEGY_DEDUPLICATION: 0,
        STRATEGY_BUDGET_AWARE_PACKING: 0,
        STRATEGY_FALLBACK: 0,
        STRATEGY_NO_OP: 0,
    }


def _safe_packing_decisions(
    packing_decisions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    safe: list[dict[str, object]] = []
    for entry in packing_decisions:
        safe.append(
            {
                "fragment_id": entry["fragment_id"],
                "priority": entry["priority"],
                "decision": entry["decision"],
                "original_chars": entry["original_chars"],
                "output_chars": entry["output_chars"],
                "reason": entry["reason"],
            }
        )
    return tuple(safe)


def _evaluate_case_with_local_test_composition(
    case: StrongerOptimizerCorpusCase,
) -> StrongerOptimizerEvaluationResult:
    """Evaluation-only sequential composition: dedupe then packing when both apply."""

    baseline_chars = len(case.current_content)
    working_content = case.current_content
    strategy_savings = _empty_strategy_savings()
    decisions: list[dict[str, object]] = []
    fallback_used = False

    if case.expected.dedupe_applicable:
        dedupe_request = TokenOptimizationLayerRequest(
            original_content=case.current_content,
            current_content=working_content,
            source_type=case.source_type,
            policy=_enabled_policy(),
        )
        dedupe_result = ExactDeduplicationLayer().optimize(dedupe_request)
        dedupe_saved_chars = int(dedupe_result.metadata.get("dedupe_saved_chars", 0))

        decision_entry: dict[str, object] = {
            "layer": STRATEGY_DEDUPLICATION,
            "decision": dedupe_result.decision.value,
            "saved_chars": dedupe_saved_chars,
        }
        if dedupe_result.fallback_used:
            decision_entry["fallback_used"] = True
            fallback_used = True
            strategy_savings[STRATEGY_FALLBACK] = 0
        elif dedupe_result.decision is TokenOptimizationLayerDecision.APPLY:
            strategy_savings[STRATEGY_DEDUPLICATION] = dedupe_saved_chars
            working_content = dedupe_result.output_content
        elif dedupe_result.decision is TokenOptimizationLayerDecision.BYPASS:
            if dedupe_saved_chars == 0 and not case.expected.packing_applicable:
                strategy_savings[STRATEGY_NO_OP] = 0
        decisions.append(decision_entry)

    if case.expected.packing_applicable:
        if case.max_chars is None:
            raise ValueError(f"packing case {case.case_id!r} requires max_chars")

        packing_input = BudgetAwarePackingInput(
            fragments=_to_packing_fragments(case.fragments),
        )
        packing_request = TokenOptimizationLayerRequest(
            original_content=case.current_content,
            current_content=working_content,
            source_type=case.source_type,
            policy=_enabled_policy(),
            metadata={"packing_input": packing_input},
        )
        packing_result = BudgetAwareContextPackingLayer(
            config=BudgetAwareContextPackingLayerConfig(max_chars=case.max_chars),
        ).optimize(packing_request)
        packing_saved_chars = int(packing_result.metadata.get("saved_chars", 0))

        packing_decision_entry: dict[str, object] = {
            "layer": STRATEGY_BUDGET_AWARE_PACKING,
            "decision": packing_result.decision.value,
            "saved_chars": packing_saved_chars,
            "budget_unit": packing_result.metadata.get("budget_unit"),
            "max_chars": packing_result.metadata.get("max_chars"),
        }
        if "packing_decisions" in packing_result.metadata:
            packing_decision_entry["packing_decisions"] = _safe_packing_decisions(
                packing_result.metadata["packing_decisions"],
            )

        if packing_result.fallback_used:
            packing_decision_entry["fallback_used"] = True
            fallback_used = True
            strategy_savings[STRATEGY_FALLBACK] = 0
            working_content = packing_result.output_content
        elif packing_result.decision is TokenOptimizationLayerDecision.APPLY:
            strategy_savings[STRATEGY_BUDGET_AWARE_PACKING] = packing_saved_chars
            working_content = packing_result.output_content
        elif packing_result.decision is TokenOptimizationLayerDecision.BYPASS:
            if not case.expected.dedupe_applicable:
                strategy_savings[STRATEGY_NO_OP] = 0
        decisions.append(packing_decision_entry)

    stronger_chars = len(working_content)
    total_saved_chars = max(0, baseline_chars - stronger_chars)

    if (
        case.expected.no_op_expected
        and total_saved_chars == 0
        and not fallback_used
    ):
        strategy_savings[STRATEGY_NO_OP] = 0

    result = StrongerOptimizerEvaluationResult(
        case_id=case.case_id,
        baseline_chars=baseline_chars,
        stronger_chars=stronger_chars,
        total_saved_chars=total_saved_chars,
        strategy_savings=strategy_savings,
        decisions=tuple(decisions),
        fallback_used=fallback_used,
        raw_content_in_report=False,
    )
    report = build_safe_evaluation_report(result, case)
    raw_content_in_report = _report_contains_raw_case_content(report, case)
    return StrongerOptimizerEvaluationResult(
        case_id=result.case_id,
        baseline_chars=result.baseline_chars,
        stronger_chars=result.stronger_chars,
        total_saved_chars=result.total_saved_chars,
        strategy_savings=result.strategy_savings,
        decisions=result.decisions,
        fallback_used=result.fallback_used,
        raw_content_in_report=raw_content_in_report,
    )


def evaluate_case(case: StrongerOptimizerCorpusCase) -> StrongerOptimizerEvaluationResult:
    """Evaluate one corpus case: baseline content vs direct stronger-layer application."""

    return _evaluate_case_with_local_test_composition(case)


def build_safe_evaluation_report(
    result: StrongerOptimizerEvaluationResult,
    case: StrongerOptimizerCorpusCase,
) -> dict[str, object]:
    """Build a raw-content-safe evaluation report for internal assertions."""

    return {
        "case_id": result.case_id,
        "title": case.title,
        "source_type": case.source_type.value,
        "baseline_chars": result.baseline_chars,
        "stronger_chars": result.stronger_chars,
        "total_saved_chars": result.total_saved_chars,
        "strategy_savings_chars": dict(result.strategy_savings),
        "decisions": [dict(entry) for entry in result.decisions],
        "fallback_used": result.fallback_used,
        "expected_primary_strategy": case.expected.expected_primary_strategy,
        "fragment_ids": [fragment.fragment_id for fragment in case.fragments],
        "priorities": [fragment.priority.value for fragment in case.fragments],
        "max_chars": case.max_chars,
        "protected_region_required": case.expected.protected_region_required,
        "safety_notes": list(case.safety_notes),
        "synthetic_marker": SYNTHETIC_CORPUS_MARKER,
    }


_PRIORITY_VALUES: frozenset[str] = frozenset(p.value for p in ContextFragmentPriority)


def _collect_string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        collected: list[str] = []
        for nested in value.values():
            collected.extend(_collect_string_values(nested))
        return tuple(collected)
    if isinstance(value, list):
        collected = []
        for nested in value:
            collected.extend(_collect_string_values(nested))
        return tuple(collected)
    return ()


def _report_contains_raw_case_content(
    report: Mapping[str, object],
    case: StrongerOptimizerCorpusCase,
) -> bool:
    restricted_report = {
        key: value for key, value in report.items() if key not in {"safety_notes", "title"}
    }
    report_values = _collect_string_values(restricted_report)
    forbidden_values: set[str] = set()
    if len(case.current_content) > 12:
        forbidden_values.add(case.current_content)
    for fragment in case.fragments:
        if len(fragment.content) <= 12:
            continue
        if fragment.content in _PRIORITY_VALUES:
            continue
        forbidden_values.add(fragment.content)
    return any(value in forbidden_values for value in report_values)


def corpus_contains_forbidden_secret_patterns(
    cases: Sequence[StrongerOptimizerCorpusCase],
) -> list[str]:
    violations: list[str] = []
    for case in cases:
        blob = case.current_content + "\n".join(fragment.content for fragment in case.fragments)
        for pattern in _FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(blob):
                violations.append(f"{case.case_id}: forbidden pattern {pattern.pattern!r}")
    return violations


def collect_metric_field_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            names.add(str(key))
            names.update(collect_metric_field_names(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            names.update(collect_metric_field_names(nested))
    return names


def _case(
    *,
    case_id: str,
    title: str,
    source_type: TokenOptimizationSourceType,
    current_content: str,
    fragments: tuple[StrongerOptimizerCorpusFragment, ...] = (),
    max_chars: int | None = None,
    expected: StrongerOptimizerExpectedBehavior,
    safety_notes: tuple[str, ...] = (),
) -> StrongerOptimizerCorpusCase:
    notes = (SYNTHETIC_CORPUS_MARKER, *safety_notes)
    return StrongerOptimizerCorpusCase(
        case_id=case_id,
        title=title,
        source_type=source_type,
        current_content=current_content,
        fragments=fragments,
        max_chars=max_chars,
        expected=expected,
        safety_notes=notes,
    )


def _fragment(
    fragment_id: str,
    content: str,
    priority: ContextFragmentPriority,
) -> StrongerOptimizerCorpusFragment:
    return StrongerOptimizerCorpusFragment(
        fragment_id=fragment_id,
        content=content,
        priority=priority,
    )


STRONGER_OPTIMIZER_CORPUS: tuple[StrongerOptimizerCorpusCase, ...] = (
    _case(
        case_id="stronger_opt.dedupe_rag_duplicate_lines",
        title="Exact duplicate RAG retrieval lines",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content=(
            "Retrieved chunk for PROJECT-ALPHA-001: deployment checklist step 1\n"
            "Retrieved chunk for PROJECT-ALPHA-001: deployment checklist step 2\n"
            "Retrieved chunk for PROJECT-ALPHA-001: deployment checklist step 1\n"
            "Retrieved chunk for PROJECT-ALPHA-001: deployment checklist step 3\n"
        ),
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=True,
            packing_applicable=False,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_DEDUPLICATION,
        ),
        safety_notes=("uses PROJECT-ALPHA-001 synthetic marker",),
    ),
    _case(
        case_id="stronger_opt.dedupe_evidence_boilerplate",
        title="Repeated boilerplate in retrieved evidence",
        source_type=TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
        current_content=(
            "INVOICE-2026-SYNTH-001 evidence header\n"
            "Line item A: synthetic procurement record\n"
            "Source reliability: verified synthetic corpus\n"
            "Line item B: synthetic procurement record\n"
            "Source reliability: verified synthetic corpus\n"
        ),
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=True,
            packing_applicable=False,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_DEDUPLICATION,
        ),
        safety_notes=("uses INVOICE-2026-SYNTH-001 synthetic marker",),
    ),
    _case(
        case_id="stronger_opt.dedupe_conversation_disclaimer",
        title="Conversation history with repeated assistant disclaimer",
        source_type=TokenOptimizationSourceType.CONVERSATION_HISTORY,
        current_content=(
            "User: Summarize TENANT-SYNTH-A onboarding for CUSTOMER_EXAMPLE.\n"
            "Assistant: Here is a concise synthetic summary.\n"
            "Assistant: I cannot provide legal advice. Synthetic disclaimer.\n"
            "User: Any risks?\n"
            "Assistant: I cannot provide legal advice. Synthetic disclaimer.\n"
        ),
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=True,
            packing_applicable=False,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_DEDUPLICATION,
        ),
        safety_notes=("uses TENANT-SYNTH-A and CUSTOMER_EXAMPLE synthetic markers",),
    ),
    _case(
        case_id="stronger_opt.packing_priority_tiers",
        title="Packing preserves MUST_KEEP and prefers HIGH_PRIORITY over DROPPABLE",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content="",
        fragments=(
            _fragment("mk", "mk", ContextFragmentPriority.MUST_KEEP),
            _fragment("high", "TRACE-SYNTH-001", ContextFragmentPriority.HIGH_PRIORITY),
            _fragment("comp", "compress", ContextFragmentPriority.COMPRESSIBLE),
            _fragment("drop", "optional appendix", ContextFragmentPriority.DROPPABLE),
        ),
        max_chars=18,
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=False,
            packing_applicable=True,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_BUDGET_AWARE_PACKING,
        ),
        safety_notes=("uses TRACE-SYNTH-001 synthetic marker",),
    ),
    _case(
        case_id="stronger_opt.packing_compressible_whitespace",
        title="Compressible fragment uses whitespace compaction only",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content="",
        fragments=(
            _fragment("mk", "mk", ContextFragmentPriority.MUST_KEEP),
            _fragment("comp", "  hello   world  ", ContextFragmentPriority.COMPRESSIBLE),
        ),
        max_chars=14,
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=False,
            packing_applicable=True,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_BUDGET_AWARE_PACKING,
        ),
        safety_notes=("compressible whitespace-only synthetic fragment",),
    ),
    _case(
        case_id="stronger_opt.packing_must_keep_over_budget",
        title="MUST_KEEP exceeding char budget triggers fallback",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content="fallback-baseline-content",
        fragments=(
            _fragment("mk1", "must_keep", ContextFragmentPriority.MUST_KEEP),
            _fragment("high", "hi", ContextFragmentPriority.HIGH_PRIORITY),
        ),
        max_chars=5,
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=False,
            packing_applicable=True,
            fallback_expected=True,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_FALLBACK,
        ),
        safety_notes=("must_keep exceeds max_chars synthetic budget case",),
    ),
    _case(
        case_id="stronger_opt.packing_protected_region",
        title="Protected synthetic URL marker survives packing",
        source_type=TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
        current_content="",
        fragments=(
            _fragment(
                "mk",
                "Config ref https://synth.example.com/PROJECT-ALPHA-001",
                ContextFragmentPriority.MUST_KEEP,
            ),
            _fragment("drop", "droppable fluff", ContextFragmentPriority.DROPPABLE),
        ),
        max_chars=120,
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=False,
            packing_applicable=True,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=True,
            expected_primary_strategy=STRATEGY_BUDGET_AWARE_PACKING,
        ),
        safety_notes=("protected synthetic URL marker only",),
    ),
    _case(
        case_id="stronger_opt.noop_clean_context",
        title="Clean unique context produces no savings",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content=(
            "Unique synthetic line alpha for PROJECT-ALPHA-001\n"
            "Unique synthetic line beta for PROJECT-ALPHA-001\n"
            "Unique synthetic line gamma for PROJECT-ALPHA-001\n"
        ),
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=True,
            packing_applicable=False,
            fallback_expected=False,
            no_op_expected=True,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_NO_OP,
        ),
        safety_notes=("no duplicate lines; no-op expected",),
    ),
    _case(
        case_id="stronger_opt.mixed_dedupe_and_packing",
        title="Mixed dedupe and packing with separated attribution",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content=(
            "Summary: TENANT-SYNTH-A overview\n"
            "Summary: TENANT-SYNTH-A overview\n"
            "Detail line for packing context\n"
        ),
        fragments=(
            _fragment("mk", "TENANT-SYNTH-A mk", ContextFragmentPriority.MUST_KEEP),
            _fragment("high", "hi", ContextFragmentPriority.HIGH_PRIORITY),
            _fragment(
                "drop",
                "long droppable optional context block",
                ContextFragmentPriority.DROPPABLE,
            ),
        ),
        max_chars=18,
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=True,
            packing_applicable=True,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy="mixed",
        ),
        safety_notes=("mixed dedupe then packing synthetic case",),
    ),
    _case(
        case_id="stronger_opt.packing_droppable_excluded_default",
        title="Droppable optional context excluded by default",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        current_content="",
        fragments=(
            _fragment("mk", "keep", ContextFragmentPriority.MUST_KEEP),
            _fragment("drop", "optional", ContextFragmentPriority.DROPPABLE),
        ),
        max_chars=100,
        expected=StrongerOptimizerExpectedBehavior(
            dedupe_applicable=False,
            packing_applicable=True,
            fallback_expected=False,
            no_op_expected=False,
            protected_region_required=False,
            expected_primary_strategy=STRATEGY_BUDGET_AWARE_PACKING,
        ),
        safety_notes=("droppable excluded even with budget headroom",),
    ),
)

# Populate assembled current_content for packing-only cases when left empty.
STRONGER_OPTIMIZER_CORPUS = tuple(
    StrongerOptimizerCorpusCase(
        case_id=case.case_id,
        title=case.title,
        source_type=case.source_type,
        current_content=(
            case.current_content
            if case.current_content
            else _assemble_fragments(case.fragments)
        ),
        fragments=case.fragments,
        max_chars=case.max_chars,
        expected=case.expected,
        safety_notes=case.safety_notes,
    )
    for case in STRONGER_OPTIMIZER_CORPUS
)

__all__ = [
    "ALLOWED_SYNTHETIC_MARKERS",
    "STRONGER_OPTIMIZER_CORPUS",
    "SYNTHETIC_CORPUS_MARKER",
    "STRATEGY_BUDGET_AWARE_PACKING",
    "STRATEGY_DEDUPLICATION",
    "STRATEGY_FALLBACK",
    "STRATEGY_NO_OP",
    "StrongerOptimizerCorpusCase",
    "StrongerOptimizerCorpusFragment",
    "StrongerOptimizerEvaluationResult",
    "StrongerOptimizerExpectedBehavior",
    "_TOKEN_NAMED_METRIC_FIELDS",
    "build_safe_evaluation_report",
    "collect_metric_field_names",
    "corpus_contains_forbidden_secret_patterns",
    "evaluate_case",
]
