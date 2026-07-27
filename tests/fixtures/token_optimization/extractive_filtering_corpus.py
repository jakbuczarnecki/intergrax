# © Artur Czarnecki. All rights reserved.

"""Internal synthetic corpus and evaluation helpers for ExtractiveFilteringLayer (TOKEN-OPT-4B)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerRequest,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers.extractive_filtering import (
    ExtractiveFilteringLayer,
    ExtractiveFilteringLayerConfig,
)

EXTRACTIVE_FILTERING_SYNTHETIC_CORPUS_MARKER = "SYNTHETIC_EXTRACTIVE_FILTERING_CORPUS_V1"

STRATEGY_EXTRACTIVE_FILTERING = "extractive_filtering"
STRATEGY_FALLBACK = "fallback"
STRATEGY_NO_OP = "no_op"

_ALLOWED_STRATEGIES = frozenset(
    {
        STRATEGY_EXTRACTIVE_FILTERING,
        STRATEGY_FALLBACK,
        STRATEGY_NO_OP,
    }
)

_ALLOWED_SOURCE_TYPES = frozenset(
    {
        TokenOptimizationSourceType.TOOL_OUTPUT,
        TokenOptimizationSourceType.TERMINAL_OUTPUT,
        TokenOptimizationSourceType.LOG_OUTPUT,
    }
)

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
        "token_savings",
        "TokenSavingsMeasurement",
    }
)

_PROTECTED_BODY_VALUE = "PROTECTED-SYNTH-REGION-VALUE-7788"


@dataclass(frozen=True, slots=True)
class ExtractiveFilteringExpectedBehavior:
    expected_primary_strategy: str
    expected_decision: str
    should_save_chars: bool
    should_fallback: bool
    should_preserve_failure_evidence: bool = False
    should_preserve_traceback: bool = False
    should_report_repeated_line_groups: bool = False
    should_bypass: bool = False
    expected_source_type: TokenOptimizationSourceType | None = None


@dataclass(frozen=True, slots=True)
class ExtractiveFilteringCorpusCase:
    case_id: str
    title: str
    source_type: TokenOptimizationSourceType
    content: str
    expected: ExtractiveFilteringExpectedBehavior
    synthetic_marker: str = EXTRACTIVE_FILTERING_SYNTHETIC_CORPUS_MARKER
    protected_regions: tuple[ProtectedRegion, ...] = ()
    preservation_markers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractiveFilteringEvaluationResult:
    case_id: str
    source_type: str
    decision: str
    fallback_used: bool
    strategy: str
    budget_unit: str
    baseline_chars: int
    output_chars: int
    saved_chars: int
    omitted_line_count: int
    repeated_line_group_count: int
    important_line_count: int
    traceback_block_count: int
    protected_region_count: int
    char_budget_satisfied: bool
    synthetic_marker: str
    failure_evidence_preserved: bool
    traceback_preserved: bool
    warning_signal_preserved: bool
    important_markers_preserved: bool
    raw_content_in_report: bool


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.BALANCED,
    )


def _deterministic_config() -> ExtractiveFilteringLayerConfig:
    return ExtractiveFilteringLayerConfig(
        enabled=True,
        max_output_chars=4000,
        head_lines=3,
        tail_lines=3,
        min_lines_before_filtering=10,
        preserve_error_lines=True,
        preserve_warning_lines=True,
        preserve_traceback_blocks=True,
        collapse_repeated_lines=True,
        repeated_line_threshold=3,
        case_sensitive_repeated_lines=True,
    )


def _progress_noise_lines(count: int, *, prefix: str = "INFO") -> list[str]:
    return [f"{prefix}: synthetic progress step {index}" for index in range(count)]


def _strategy_for_decision(
    decision: TokenOptimizationLayerDecision,
    *,
    fallback_used: bool,
) -> str:
    if fallback_used or decision is TokenOptimizationLayerDecision.FALLBACK:
        return STRATEGY_FALLBACK
    if decision is TokenOptimizationLayerDecision.BYPASS:
        return STRATEGY_NO_OP
    return STRATEGY_EXTRACTIVE_FILTERING


def _markers_preserved(output_content: str, markers: Sequence[str]) -> bool:
    if not markers:
        return True
    return all(marker in output_content for marker in markers)


def evaluate_case(case: ExtractiveFilteringCorpusCase) -> ExtractiveFilteringEvaluationResult:
    """Evaluate one corpus case using the real ExtractiveFilteringLayer."""

    metadata: dict[str, Any] = {}
    if case.protected_regions:
        metadata["protected_regions"] = case.protected_regions

    request = TokenOptimizationLayerRequest(
        original_content=case.content,
        current_content=case.content,
        source_type=case.source_type,
        policy=_enabled_policy(),
        metadata=metadata,
    )
    layer_result = ExtractiveFilteringLayer(config=_deterministic_config()).optimize(request)
    layer_meta = layer_result.metadata
    fallback_used = bool(layer_result.fallback_used)
    strategy = _strategy_for_decision(layer_result.decision, fallback_used=fallback_used)
    output_content = layer_result.output_content

    failure_markers = ("FAILED", "AssertionError", "ERROR", "exit code")
    failure_evidence_preserved = any(marker in output_content for marker in failure_markers)
    traceback_preserved = (
        "Traceback (most recent call last):" in output_content
        and "ValueError:" in output_content
    )
    warning_signal_preserved = "WARNING:" in output_content or "warning" in output_content.lower()
    important_markers_preserved = _markers_preserved(output_content, case.preservation_markers)

    result = ExtractiveFilteringEvaluationResult(
        case_id=case.case_id,
        source_type=case.source_type.value,
        decision=layer_result.decision.value,
        fallback_used=fallback_used,
        strategy=strategy,
        budget_unit=str(layer_meta.get("budget_unit", "chars")),
        baseline_chars=int(layer_meta.get("original_chars", len(case.content))),
        output_chars=int(layer_meta.get("output_chars", len(output_content))),
        saved_chars=int(layer_meta.get("saved_chars", 0)),
        omitted_line_count=int(layer_meta.get("omitted_line_count", 0)),
        repeated_line_group_count=len(layer_meta.get("repeated_line_groups", ()) or ()),
        important_line_count=int(layer_meta.get("important_line_count", 0)),
        traceback_block_count=int(layer_meta.get("traceback_block_count", 0)),
        protected_region_count=int(layer_meta.get("protected_region_count", 0)),
        char_budget_satisfied=bool(layer_meta.get("char_budget_satisfied", True)),
        synthetic_marker=case.synthetic_marker,
        failure_evidence_preserved=failure_evidence_preserved,
        traceback_preserved=traceback_preserved,
        warning_signal_preserved=warning_signal_preserved,
        important_markers_preserved=important_markers_preserved,
        raw_content_in_report=False,
    )
    report = build_safe_evaluation_report(result)
    raw_leak = _report_contains_raw_case_content(report, case)
    if not raw_leak:
        return result
    return ExtractiveFilteringEvaluationResult(
        case_id=result.case_id,
        source_type=result.source_type,
        decision=result.decision,
        fallback_used=result.fallback_used,
        strategy=result.strategy,
        budget_unit=result.budget_unit,
        baseline_chars=result.baseline_chars,
        output_chars=result.output_chars,
        saved_chars=result.saved_chars,
        omitted_line_count=result.omitted_line_count,
        repeated_line_group_count=result.repeated_line_group_count,
        important_line_count=result.important_line_count,
        traceback_block_count=result.traceback_block_count,
        protected_region_count=result.protected_region_count,
        char_budget_satisfied=result.char_budget_satisfied,
        synthetic_marker=result.synthetic_marker,
        failure_evidence_preserved=result.failure_evidence_preserved,
        traceback_preserved=result.traceback_preserved,
        warning_signal_preserved=result.warning_signal_preserved,
        important_markers_preserved=result.important_markers_preserved,
        raw_content_in_report=True,
    )


def build_safe_evaluation_report(
    result: ExtractiveFilteringEvaluationResult,
) -> dict[str, object]:
    """Build a raw-content-safe evaluation report (char-level metrics only)."""

    return {
        "case_id": result.case_id,
        "source_type": result.source_type,
        "decision": result.decision,
        "fallback_used": result.fallback_used,
        "strategy": result.strategy,
        "budget_unit": result.budget_unit,
        "baseline_chars": result.baseline_chars,
        "output_chars": result.output_chars,
        "saved_chars": result.saved_chars,
        "omitted_line_count": result.omitted_line_count,
        "repeated_line_group_count": result.repeated_line_group_count,
        "important_line_count": result.important_line_count,
        "traceback_block_count": result.traceback_block_count,
        "protected_region_count": result.protected_region_count,
        "char_budget_satisfied": result.char_budget_satisfied,
        "synthetic_marker": result.synthetic_marker,
        "failure_evidence_preserved": result.failure_evidence_preserved,
        "traceback_preserved": result.traceback_preserved,
        "warning_signal_preserved": result.warning_signal_preserved,
        "important_markers_preserved": result.important_markers_preserved,
    }


def _collect_string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        collected: list[str] = []
        for nested in value.values():
            collected.extend(_collect_string_values(nested))
        return tuple(collected)
    if isinstance(value, (list, tuple)):
        collected = []
        for nested in value:
            collected.extend(_collect_string_values(nested))
        return tuple(collected)
    return ()


def _report_contains_raw_case_content(
    report: Mapping[str, object],
    case: ExtractiveFilteringCorpusCase,
) -> bool:
    report_values = _collect_string_values(report)
    if len(case.content) > 12 and case.content in report_values:
        return True
    return False


def collect_metric_field_names(payload: object) -> set[str]:
    names: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            names.add(str(key))
            names.update(collect_metric_field_names(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            names.update(collect_metric_field_names(item))
    return names


def corpus_contains_forbidden_secret_patterns(
    corpus: Sequence[ExtractiveFilteringCorpusCase],
) -> list[str]:
    violations: list[str] = []
    for case in corpus:
        for pattern in _FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(case.content):
                violations.append(f"{case.case_id}: {pattern.pattern}")
    return violations


def _case_verbose_progress_noise() -> ExtractiveFilteringCorpusCase:
    content = "\n".join(_progress_noise_lines(160)) + "\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.terminal_verbose_progress_noise",
        title="Verbose terminal progress noise",
        source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_EXTRACTIVE_FILTERING,
            expected_decision=TokenOptimizationLayerDecision.APPLY.value,
            should_save_chars=True,
            should_fallback=False,
            expected_source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        ),
    )


def _case_pytest_failure_with_noise() -> ExtractiveFilteringCorpusCase:
    head = _progress_noise_lines(40)
    body = _progress_noise_lines(60)
    failure = [
        "============================= test session starts ==============================",
        "collected 12 items",
        "tests/unit/synth/test_alpha.py::test_alpha FAILED",
        "E       AssertionError: expected PROJECT-ALPHA-001",
        "ERROR collecting leftover noise",
        "===== 1 failed, 11 passed in 0.42s =====",
        "exit code: 1",
    ]
    tail = _progress_noise_lines(40, prefix="DEBUG")
    content = "\n".join(head + body + failure + tail) + "\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.terminal_pytest_failure_with_noise",
        title="Pytest failure evidence amid terminal noise",
        source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_EXTRACTIVE_FILTERING,
            expected_decision=TokenOptimizationLayerDecision.APPLY.value,
            should_save_chars=True,
            should_fallback=False,
            should_preserve_failure_evidence=True,
            expected_source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        ),
        preservation_markers=("FAILED", "AssertionError", "ERROR", "exit code"),
    )


def _case_traceback_inside_long_output() -> ExtractiveFilteringCorpusCase:
    head = _progress_noise_lines(50)
    traceback_lines = [
        "Traceback (most recent call last):",
        '  File "synth_app.py", line 12, in run_main',
        "    raise ValueError('synthetic boom')",
        "ValueError: synthetic boom",
        "",
    ]
    tail = _progress_noise_lines(50)
    content = "\n".join(head + traceback_lines + tail) + "\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.terminal_traceback_inside_long_output",
        title="Traceback preserved inside long terminal output",
        source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_EXTRACTIVE_FILTERING,
            expected_decision=TokenOptimizationLayerDecision.APPLY.value,
            should_save_chars=True,
            should_fallback=False,
            should_preserve_traceback=True,
            expected_source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        ),
        preservation_markers=(
            "Traceback (most recent call last):",
            "ValueError: synthetic boom",
        ),
    )


def _case_repeated_warnings() -> ExtractiveFilteringCorpusCase:
    repeated = ["WARNING: synthetic dependency already satisfied"] * 10
    tail = _progress_noise_lines(20, prefix="INFO")
    content = "\n".join(repeated + tail) + "\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.terminal_repeated_warnings",
        title="Repeated terminal warnings collapsed",
        source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_EXTRACTIVE_FILTERING,
            expected_decision=TokenOptimizationLayerDecision.APPLY.value,
            should_save_chars=True,
            should_fallback=False,
            should_report_repeated_line_groups=True,
            expected_source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        ),
        preservation_markers=("WARNING: synthetic dependency already satisfied",),
    )


def _case_protected_value_in_body() -> ExtractiveFilteringCorpusCase:
    lines = _progress_noise_lines(40)
    lines[20] = f"marker before {_PROTECTED_BODY_VALUE}"
    content = "\n".join(lines) + "\n"
    protected = ProtectedRegion(
        kind=ProtectedRegionKind.IDENTIFIER,
        value=_PROTECTED_BODY_VALUE,
    )
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.terminal_protected_value_in_body",
        title="Protected region in filterable body triggers fallback",
        source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_FALLBACK,
            expected_decision=TokenOptimizationLayerDecision.FALLBACK.value,
            should_save_chars=False,
            should_fallback=True,
            expected_source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        ),
        protected_regions=(protected,),
    )


def _case_short_clean_output() -> ExtractiveFilteringCorpusCase:
    content = "ok\nready\ndone\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.terminal_short_clean_output",
        title="Short clean terminal output bypasses filtering",
        source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_NO_OP,
            expected_decision=TokenOptimizationLayerDecision.BYPASS.value,
            should_save_chars=False,
            should_fallback=False,
            should_bypass=True,
            expected_source_type=TokenOptimizationSourceType.TERMINAL_OUTPUT,
        ),
    )


def _case_tool_output_json_like_noise() -> ExtractiveFilteringCorpusCase:
    lines = [
        f'{{"status": "ok", "item": {index}, "payload": "SYNTH-JSON-NOISE"}}'
        for index in range(140)
    ]
    content = "\n".join(lines) + "\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.tool_output_large_json_like_noise",
        title="Large JSON-like tool output noise",
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_EXTRACTIVE_FILTERING,
            expected_decision=TokenOptimizationLayerDecision.APPLY.value,
            should_save_chars=True,
            should_fallback=False,
            expected_source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        ),
    )


def _case_log_output_warning_error_mix() -> ExtractiveFilteringCorpusCase:
    head = _progress_noise_lines(40, prefix="INFO")
    important = [
        "WARNING: cache miss for TENANT-SYNTH-A",
        "ERROR: worker failed to flush buffer",
        "exit code: 2",
    ]
    body = _progress_noise_lines(50, prefix="DEBUG")
    tail = _progress_noise_lines(30, prefix="INFO")
    content = "\n".join(head + important + body + tail) + "\n"
    return ExtractiveFilteringCorpusCase(
        case_id="extractive_filtering.log_output_warning_error_mix",
        title="Log warning and error mix preserves important lines",
        source_type=TokenOptimizationSourceType.LOG_OUTPUT,
        content=content,
        expected=ExtractiveFilteringExpectedBehavior(
            expected_primary_strategy=STRATEGY_EXTRACTIVE_FILTERING,
            expected_decision=TokenOptimizationLayerDecision.APPLY.value,
            should_save_chars=True,
            should_fallback=False,
            expected_source_type=TokenOptimizationSourceType.LOG_OUTPUT,
        ),
        preservation_markers=(
            "WARNING: cache miss for TENANT-SYNTH-A",
            "ERROR: worker failed to flush buffer",
            "exit code: 2",
        ),
    )


EXTRACTIVE_FILTERING_CORPUS: tuple[ExtractiveFilteringCorpusCase, ...] = (
    _case_verbose_progress_noise(),
    _case_pytest_failure_with_noise(),
    _case_traceback_inside_long_output(),
    _case_repeated_warnings(),
    _case_protected_value_in_body(),
    _case_short_clean_output(),
    _case_tool_output_json_like_noise(),
    _case_log_output_warning_error_mix(),
)
