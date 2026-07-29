# © Artur Czarnecki. All rights reserved.

"""Synthetic corpus and evaluation helpers for pipeline configuration evals (TOKEN-8C)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from intergrax.runtime.token_optimization.builtin_catalog import (
    BuiltInTokenOptimizationLayerSelection,
    create_builtin_token_optimization_layer_catalog,
)
from intergrax.runtime.token_optimization.contracts import (
    ContextFragmentPriority,
    ProtectedRegion,
    ProtectedRegionKind,
    TokenOptimizationLayerRef,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers import (
    BudgetAwareContextPackingLayerConfig,
    BudgetAwarePackingFragment,
    BudgetAwarePackingInput,
    ExtractiveFilteringLayerConfig,
)
from intergrax.runtime.token_optimization.pipeline import TokenOptimizationPipelineRunner

PIPELINE_CONFIGURATION_SYNTHETIC_CORPUS_MARKER = (
    "synthetic_token_optimization_pipeline_configuration_corpus_v1"
)

_EXACT_DEDUP_ID = "builtin.exact_deduplication"
_EXTRACTIVE_ID = "builtin.extractive_filtering"
_BUDGET_PACKING_ID = "builtin.budget_aware_context_packing"

_PACKING_MAX_CHARS = 80
_MIXED_PACKING_MAX_CHARS = 50

_PROTECTED_SYNTH_VALUE = "PROTECTED-SYNTH-PIPELINE-EVAL-7788"

_FORBIDDEN_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"Bearer\s+eyJ[a-zA-Z0-9_-]+"),
    re.compile(r"password\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
)

_FORBIDDEN_REPORT_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "content",
        "original_content",
        "final_content",
        "current_content",
        "output_content",
        "fragment_content",
        "protected_value",
        "request_metadata",
        "result_metadata",
        "receipt_metadata",
        "exception",
        "traceback",
    }
)

_RECOMMENDATION_SUBSTRINGS: tuple[str, ...] = (
    "winner",
    "best_configuration",
    "recommended_configuration",
    "recommendation",
    "production_ready",
    "quality_score",
)


@dataclass(frozen=True, slots=True)
class PipelineConfigurationEvaluationCase:
    case_id: str
    title: str
    source_type: TokenOptimizationSourceType
    content: str
    protected_regions: tuple[ProtectedRegion, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    synthetic_marker: str = PIPELINE_CONFIGURATION_SYNTHETIC_CORPUS_MARKER

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if self.source_type is TokenOptimizationSourceType.UNKNOWN:
            raise ValueError("source_type must not be UNKNOWN")
        if self.synthetic_marker != PIPELINE_CONFIGURATION_SYNTHETIC_CORPUS_MARKER:
            raise ValueError("case must use the canonical synthetic corpus marker")


@dataclass(frozen=True, slots=True)
class PipelineConfigurationEvaluationConfiguration:
    configuration_id: str
    selections: tuple[BuiltInTokenOptimizationLayerSelection, ...]
    layer_refs: tuple[TokenOptimizationLayerRef, ...]
    policy: TokenOptimizationPolicy
    pipeline_mode: TokenOptimizationPipelineMode = TokenOptimizationPipelineMode.REPLACE

    def __post_init__(self) -> None:
        if not self.configuration_id.strip():
            raise ValueError("configuration_id must be non-empty")
        if self.pipeline_mode is not TokenOptimizationPipelineMode.REPLACE:
            raise ValueError("pipeline_mode must be REPLACE")
        if len(self.selections) != len(self.layer_refs):
            raise ValueError("selections and layer_refs must have the same length")
        selection_ids = [selection.layer_id for selection in self.selections]
        ref_ids = [layer_ref.layer_id for layer_ref in self.layer_refs]
        if selection_ids != ref_ids:
            raise ValueError("selection order must match layer-ref order exactly")
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("duplicate selected layer IDs are not allowed")
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("duplicate layer-ref IDs are not allowed")


@dataclass(frozen=True, slots=True)
class PipelineLayerEvaluationOutcome:
    layer_id: str
    decision: str
    bypass_reason: str | None
    validation_status: str | None


@dataclass(frozen=True, slots=True)
class PipelineConfigurationEvaluationResult:
    case_id: str
    configuration_id: str
    source_type: str
    budget_unit: str
    original_chars: int
    final_chars: int
    char_delta: int
    reduction_ratio: float
    applied_layer_ids: tuple[str, ...]
    bypassed_layer_ids: tuple[str, ...]
    failed_layer_ids: tuple[str, ...]
    executed_layer_ids: tuple[str, ...]
    fallback_used: bool
    completed: bool
    required_failure_layer_id: str | None
    layer_outcomes: tuple[PipelineLayerEvaluationOutcome, ...]


@dataclass(frozen=True, slots=True)
class PipelineConfigurationEvaluationExecution:
    case_count: int
    configuration_count: int
    execution_count: int
    results: tuple[PipelineConfigurationEvaluationResult, ...]


def _layer_ref(layer_id: str) -> TokenOptimizationLayerRef:
    return TokenOptimizationLayerRef(layer_id=layer_id)


def _extractive_config() -> ExtractiveFilteringLayerConfig:
    return ExtractiveFilteringLayerConfig(
        min_lines_before_filtering=10,
        head_lines=3,
        tail_lines=3,
        max_output_chars=4000,
    )


def _packing_config(*, max_chars: int) -> BudgetAwareContextPackingLayerConfig:
    return BudgetAwareContextPackingLayerConfig(max_chars=max_chars)


def _packing_fragment(
    fragment_id: str,
    content: str,
    priority: ContextFragmentPriority,
) -> BudgetAwarePackingFragment:
    return BudgetAwarePackingFragment(
        fragment_id=fragment_id,
        content=content,
        priority=priority,
    )


def _priority_packing_input() -> BudgetAwarePackingInput:
    return BudgetAwarePackingInput(
        fragments=(
            _packing_fragment("mk1", "SYNTH-MUST-KEEP-FRAG", ContextFragmentPriority.MUST_KEEP),
            _packing_fragment(
                "hp1",
                "SYNTH-HIGH-PRIORITY-FRAG",
                ContextFragmentPriority.HIGH_PRIORITY,
            ),
            _packing_fragment(
                "cp1",
                "SYNTH   compressible   expandable   filler",
                ContextFragmentPriority.COMPRESSIBLE,
            ),
            _packing_fragment("dp1", "D" * 200, ContextFragmentPriority.DROPPABLE),
        ),
    )


def _mixed_packing_input() -> BudgetAwarePackingInput:
    return BudgetAwarePackingInput(
        fragments=(
            _packing_fragment("mk1", "SYNTH-MIXED-MUST-KEEP", ContextFragmentPriority.MUST_KEEP),
            _packing_fragment("dp1", "Z" * 100, ContextFragmentPriority.DROPPABLE),
        ),
    )


def _assembled_fragment_content(packing_input: BudgetAwarePackingInput) -> str:
    separator = "\n"
    return separator.join(fragment.content for fragment in packing_input.fragments)


def _progress_noise_lines(count: int, *, prefix: str = "INFO") -> list[str]:
    return [f"{prefix}: synthetic progress step {index}" for index in range(count)]


def _noisy_tool_output() -> str:
    lines = _progress_noise_lines(150)
    lines[75] = "ERROR: synthetic module compile failed"
    lines.append("INFO: synthetic final cleanup")
    return "\n".join(lines) + "\n"


def _protected_tool_output() -> str:
    lines = _progress_noise_lines(40)
    lines[20] = f"marker before {_PROTECTED_SYNTH_VALUE}"
    return "\n".join(lines) + "\n"


def _case_rag_duplicate_lines() -> PipelineConfigurationEvaluationCase:
    content = "\n".join(
        [
            "SYNTH-EVIDENCE-ALPHA",
            "SYNTH-EVIDENCE-ALPHA",
            "SYNTH-EVIDENCE-BETA",
            "SYNTH-EVIDENCE-GAMMA",
        ]
    )
    return PipelineConfigurationEvaluationCase(
        case_id="pipeline_eval.rag_duplicate_lines",
        title="RAG context with exact duplicate evidence lines",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        content=content,
    )


def _case_rag_priority_packing() -> PipelineConfigurationEvaluationCase:
    packing_input = _priority_packing_input()
    return PipelineConfigurationEvaluationCase(
        case_id="pipeline_eval.rag_priority_packing",
        title="RAG context with priority-tier packing input",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        content=_assembled_fragment_content(packing_input),
        metadata={"packing_input": packing_input},
    )


def _case_rag_mixed_dedupe_packing() -> PipelineConfigurationEvaluationCase:
    content = "\n".join(
        [
            "SYNTH-MIXED-LINE-ONE",
            "SYNTH-MIXED-LINE-ONE",
            "SYNTH-MIXED-LINE-TWO",
        ]
    )
    return PipelineConfigurationEvaluationCase(
        case_id="pipeline_eval.rag_mixed_dedupe_packing",
        title="RAG context with duplicate lines and packing input",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        content=content,
        metadata={"packing_input": _mixed_packing_input()},
    )


def _case_tool_noisy_repeated_output() -> PipelineConfigurationEvaluationCase:
    return PipelineConfigurationEvaluationCase(
        case_id="pipeline_eval.tool_noisy_repeated_output",
        title="Noisy repeated synthetic tool output",
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        content=_noisy_tool_output(),
    )


def _case_tool_protected_value() -> PipelineConfigurationEvaluationCase:
    protected = ProtectedRegion(
        kind=ProtectedRegionKind.IDENTIFIER,
        value=_PROTECTED_SYNTH_VALUE,
    )
    return PipelineConfigurationEvaluationCase(
        case_id="pipeline_eval.tool_protected_value",
        title="Protected synthetic value amid noisy tool output",
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        content=_protected_tool_output(),
        protected_regions=(protected,),
    )


def _case_clean_noop() -> PipelineConfigurationEvaluationCase:
    return PipelineConfigurationEvaluationCase(
        case_id="pipeline_eval.clean_noop",
        title="Short clean tool output with no optimization opportunity",
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        content="ok\nready\ndone\n",
    )


PIPELINE_CONFIGURATION_CORPUS: tuple[PipelineConfigurationEvaluationCase, ...] = (
    _case_rag_duplicate_lines(),
    _case_rag_priority_packing(),
    _case_rag_mixed_dedupe_packing(),
    _case_tool_noisy_repeated_output(),
    _case_tool_protected_value(),
    _case_clean_noop(),
)


def _config_disabled() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_BUDGET_PACKING_ID,
            config=_packing_config(max_chars=_PACKING_MAX_CHARS),
        ),
    )
    layer_refs = (
        _layer_ref(_EXACT_DEDUP_ID),
        _layer_ref(_EXTRACTIVE_ID),
        _layer_ref(_BUDGET_PACKING_ID),
    )
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="disabled",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=False,
            profile=TokenOptimizationProfile.OFF,
            allow_lossy=False,
        ),
    )


def _config_measure_only() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_BUDGET_PACKING_ID,
            config=_packing_config(max_chars=_PACKING_MAX_CHARS),
        ),
    )
    layer_refs = (
        _layer_ref(_EXACT_DEDUP_ID),
        _layer_ref(_EXTRACTIVE_ID),
        _layer_ref(_BUDGET_PACKING_ID),
    )
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="measure_only",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.MEASURE_ONLY,
            allow_lossy=False,
        ),
    )


def _config_exact_only() -> PipelineConfigurationEvaluationConfiguration:
    selections = (BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),)
    layer_refs = (_layer_ref(_EXACT_DEDUP_ID),)
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="exact_only",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
            allow_lossy=False,
        ),
    )


def _config_extractive_allowed() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
    )
    layer_refs = (_layer_ref(_EXTRACTIVE_ID),)
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="extractive_allowed",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.BALANCED,
            allow_lossy=True,
        ),
    )


def _config_extractive_blocked() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
    )
    layer_refs = (_layer_ref(_EXTRACTIVE_ID),)
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="extractive_blocked",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
            allow_lossy=False,
        ),
    )


def _config_packing_only() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_BUDGET_PACKING_ID,
            config=_packing_config(max_chars=_PACKING_MAX_CHARS),
        ),
    )
    layer_refs = (_layer_ref(_BUDGET_PACKING_ID),)
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="packing_only",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
            allow_lossy=False,
        ),
    )


def _config_exact_then_packing() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_BUDGET_PACKING_ID,
            config=_packing_config(max_chars=_MIXED_PACKING_MAX_CHARS),
        ),
    )
    layer_refs = (
        _layer_ref(_EXACT_DEDUP_ID),
        _layer_ref(_BUDGET_PACKING_ID),
    )
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="exact_then_packing",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
            allow_lossy=False,
        ),
    )


def _config_exact_then_extractive() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
    )
    layer_refs = (
        _layer_ref(_EXACT_DEDUP_ID),
        _layer_ref(_EXTRACTIVE_ID),
    )
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="exact_then_extractive",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.BALANCED,
            allow_lossy=True,
        ),
    )


def _config_extractive_then_exact() -> PipelineConfigurationEvaluationConfiguration:
    selections = (
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
    )
    layer_refs = (
        _layer_ref(_EXTRACTIVE_ID),
        _layer_ref(_EXACT_DEDUP_ID),
    )
    return PipelineConfigurationEvaluationConfiguration(
        configuration_id="extractive_then_exact",
        selections=selections,
        layer_refs=layer_refs,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.BALANCED,
            allow_lossy=True,
        ),
    )


PIPELINE_CONFIGURATION_MATRIX: tuple[PipelineConfigurationEvaluationConfiguration, ...] = (
    _config_disabled(),
    _config_measure_only(),
    _config_exact_only(),
    _config_extractive_allowed(),
    _config_extractive_blocked(),
    _config_packing_only(),
    _config_exact_then_packing(),
    _config_exact_then_extractive(),
    _config_extractive_then_exact(),
)


def _pipeline_id(case_id: str, configuration_id: str) -> str:
    safe_case = case_id.replace(".", "-")
    return f"pipeline-eval-{safe_case}-{configuration_id}"


def _normalize_executed_layer_ids(receipt_metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw = receipt_metadata["executed_layer_ids"]
    if not isinstance(raw, list):
        raise TypeError("executed_layer_ids must be a list")
    normalized: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError("executed_layer_ids entries must be strings")
        normalized.append(item)
    return tuple(normalized)


def _normalize_completed(receipt_metadata: Mapping[str, object]) -> bool:
    raw = receipt_metadata["completed"]
    if raw is True:
        return True
    if raw is False:
        return False
    raise TypeError("completed must be a strict boolean")


def _normalize_required_failure_layer_id(
    receipt_metadata: Mapping[str, object],
) -> str | None:
    if "required_failure_layer_id" not in receipt_metadata:
        return None
    raw = receipt_metadata["required_failure_layer_id"]
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("required_failure_layer_id must be str or None")
    return raw


def _normalize_layer_outcomes(
    layer_results: tuple[object, ...],
) -> tuple[PipelineLayerEvaluationOutcome, ...]:
    outcomes: list[PipelineLayerEvaluationOutcome] = []
    for layer_result in layer_results:
        layer_id = layer_result.layer_id
        decision = layer_result.decision.value
        bypass_reason = (
            layer_result.bypass_reason.value
            if layer_result.bypass_reason is not None
            else None
        )
        validation_status = None
        if layer_result.validation is not None:
            validation_status = layer_result.validation.status.value
        outcomes.append(
            PipelineLayerEvaluationOutcome(
                layer_id=layer_id,
                decision=decision,
                bypass_reason=bypass_reason,
                validation_status=validation_status,
            )
        )
    return tuple(outcomes)


def _build_request(case: PipelineConfigurationEvaluationCase) -> TokenOptimizationRequest:
    metadata: dict[str, object] = dict(case.metadata)
    return TokenOptimizationRequest(
        content=case.content,
        source_type=case.source_type,
        protected_regions=case.protected_regions,
        metadata=metadata,
    )


def evaluate_pipeline_configuration(
    case: PipelineConfigurationEvaluationCase,
    configuration: PipelineConfigurationEvaluationConfiguration,
) -> PipelineConfigurationEvaluationResult:
    """Evaluate one case/configuration pair through the standard TOKEN-8A runner path."""

    catalog = create_builtin_token_optimization_layer_catalog()
    registry = catalog.create_registry(configuration.selections)
    runner = TokenOptimizationPipelineRunner(registry=registry)
    pipeline_config = TokenOptimizationPipelineConfig(
        pipeline_id=_pipeline_id(case.case_id, configuration.configuration_id),
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=configuration.layer_refs,
    )
    request = _build_request(case)
    request_with_policy = TokenOptimizationRequest(
        content=request.content,
        source_type=request.source_type,
        policy=configuration.policy,
        protected_regions=request.protected_regions,
        metadata=request.metadata,
    )
    result = runner.run(request=request_with_policy, config=pipeline_config)

    original_chars = len(case.content)
    final_chars = len(result.final_content)
    char_delta = original_chars - final_chars
    reduction_ratio = 0.0 if original_chars == 0 else char_delta / original_chars

    receipt_metadata = result.receipt_metadata
    return PipelineConfigurationEvaluationResult(
        case_id=case.case_id,
        configuration_id=configuration.configuration_id,
        source_type=case.source_type.value,
        budget_unit="chars",
        original_chars=original_chars,
        final_chars=final_chars,
        char_delta=char_delta,
        reduction_ratio=reduction_ratio,
        applied_layer_ids=result.applied_layer_ids,
        bypassed_layer_ids=result.bypassed_layer_ids,
        failed_layer_ids=result.failed_layer_ids,
        executed_layer_ids=_normalize_executed_layer_ids(receipt_metadata),
        fallback_used=result.fallback_used,
        completed=_normalize_completed(receipt_metadata),
        required_failure_layer_id=_normalize_required_failure_layer_id(receipt_metadata),
        layer_outcomes=_normalize_layer_outcomes(result.layer_results),
    )


def run_pipeline_configuration_evaluation_matrix(
    cases: tuple[PipelineConfigurationEvaluationCase, ...] = PIPELINE_CONFIGURATION_CORPUS,
    configurations: tuple[
        PipelineConfigurationEvaluationConfiguration,
        ...
    ] = PIPELINE_CONFIGURATION_MATRIX,
) -> PipelineConfigurationEvaluationExecution:
    """Run the canonical case/configuration matrix in deterministic order."""

    results: list[PipelineConfigurationEvaluationResult] = []
    for case in cases:
        for configuration in configurations:
            results.append(evaluate_pipeline_configuration(case, configuration))
    return PipelineConfigurationEvaluationExecution(
        case_count=len(cases),
        configuration_count=len(configurations),
        execution_count=len(results),
        results=tuple(results),
    )


def _result_to_report_dict(
    result: PipelineConfigurationEvaluationResult,
) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "configuration_id": result.configuration_id,
        "source_type": result.source_type,
        "budget_unit": result.budget_unit,
        "original_chars": result.original_chars,
        "final_chars": result.final_chars,
        "char_delta": result.char_delta,
        "reduction_ratio": result.reduction_ratio,
        "applied_layer_ids": list(result.applied_layer_ids),
        "bypassed_layer_ids": list(result.bypassed_layer_ids),
        "failed_layer_ids": list(result.failed_layer_ids),
        "executed_layer_ids": list(result.executed_layer_ids),
        "fallback_used": result.fallback_used,
        "completed": result.completed,
        "required_failure_layer_id": result.required_failure_layer_id,
        "layer_outcomes": [
            {
                "layer_id": outcome.layer_id,
                "decision": outcome.decision,
                "bypass_reason": outcome.bypass_reason,
                "validation_status": outcome.validation_status,
            }
            for outcome in result.layer_outcomes
        ],
    }


def build_safe_pipeline_configuration_report(
    execution: PipelineConfigurationEvaluationExecution,
) -> dict[str, object]:
    """Build a raw-content-safe pipeline configuration evaluation report."""

    return {
        "synthetic_marker": PIPELINE_CONFIGURATION_SYNTHETIC_CORPUS_MARKER,
        "budget_unit": "chars",
        "case_count": execution.case_count,
        "configuration_count": execution.configuration_count,
        "execution_count": execution.execution_count,
        "results": [_result_to_report_dict(result) for result in execution.results],
    }


def collect_report_field_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            names.add(str(key))
            names.update(collect_report_field_names(nested))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.update(collect_report_field_names(item))
    return names


def collect_report_string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        collected: list[str] = []
        for nested in value.values():
            collected.extend(collect_report_string_values(nested))
        return tuple(collected)
    if isinstance(value, (list, tuple)):
        collected: list[str] = []
        for nested in value:
            collected.extend(collect_report_string_values(nested))
        return tuple(collected)
    return ()


def corpus_contains_forbidden_secret_patterns(
    cases: tuple[PipelineConfigurationEvaluationCase, ...],
) -> list[str]:
    violations: list[str] = []
    for case in cases:
        for pattern in _FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(case.content):
                violations.append(f"{case.case_id}: {pattern.pattern}")
        for protected in case.protected_regions:
            for protected_pattern in _FORBIDDEN_SECRET_PATTERNS:
                if protected_pattern.search(protected.value):
                    violations.append(
                        f"{case.case_id}: protected {protected_pattern.pattern}"
                    )
    return violations


def protected_synthetic_value() -> str:
    """Return the synthetic protected value for focused tests (not for reports)."""

    return _PROTECTED_SYNTH_VALUE
