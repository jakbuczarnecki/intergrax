# © Artur Czarnecki. All rights reserved.

"""Benchmark orchestration for local model qualification."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

from local_workspace_application.benchmarks.local_model_qualification.config import (
    LocalModelQualificationConfig,
    ModelConfig,
    QualificationConfig,
    configuration_sha256,
    load_config,
)
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    BENCHMARK_ID,
    CaseExecutionStatus,
    CaseResult,
    HostMetadata,
    LatencyStats,
    LocalModelQualificationResult,
    ModelMetadata,
    ModelResult,
    ModelStatus,
    ObservedExecutionMode,
    OllamaEnvironment,
    ProtocolResult,
    ProtocolStatus,
    QualificationSummary,
    SchemaProbeStatus,
    StructuralFailureCategory,
    WarmupStatus,
)
from local_workspace_application.benchmarks.local_model_qualification.corpus import (
    QualificationCase,
    case_by_id,
    qualification_cases,
)
from local_workspace_application.benchmarks.local_model_qualification.environment import (
    collect_host_metadata,
    collect_ollama_environment,
    derive_execution_mode,
    fetch_model_metadata,
    list_installed_models,
    pull_model,
)
from local_workspace_application.benchmarks.local_model_qualification.evaluator import evaluate_semantics
from local_workspace_application.benchmarks.local_model_qualification.protocols import (
    PROTOCOL_SINGLE_PLAN_TOOL,
    PROTOCOL_STRUCTURED_OUTPUT,
    BenchmarkAdapter,
    run_protocol_attempt,
)
from local_workspace_application.benchmarks.local_model_qualification.report import (
    qualification_rank,
    render_markdown,
    serialize_result_json,
)
from local_workspace_application.conversation.interaction_models import ConversationInteractionPlan

_PROBE_CASE_ID = "planner.workspace_list"
_INVALID_DRAFT_CATEGORIES = frozenset(
    {
        StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
        StructuralFailureCategory.INVALID_TOOL_ARGUMENTS.value,
    }
)


def percentile_nearest_rank(values: list[float], p: float) -> float:
    """Deterministic nearest-rank percentile (p in [0, 1])."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(1, math.ceil(p * len(sorted_values)))
    return sorted_values[rank - 1]


def median_deterministic(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        return LatencyStats(minimum=0.0, median=0.0, p95=0.0, maximum=0.0)
    return LatencyStats(
        minimum=min(values),
        median=median_deterministic(values),
        p95=percentile_nearest_rank(values, 0.95),
        maximum=max(values),
    )


def model_slug(model_name: str) -> str:
    return model_name.replace(":", "-")


def build_run_id(model_name: str, protocol: str, case_id: str, repetition: int) -> str:
    return f"lkw-model-qualification:{model_slug(model_name)}:{protocol}:{case_id}:{repetition}"


def enabled_protocols(config: LocalModelQualificationConfig) -> tuple[str, ...]:
    protocols: list[str] = []
    if config.protocols.structured_output:
        protocols.append(PROTOCOL_STRUCTURED_OUTPUT)
    if config.protocols.single_plan_tool:
        protocols.append(PROTOCOL_SINGLE_PLAN_TOOL)
    return tuple(protocols)


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return output.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _create_adapter(config: LocalModelQualificationConfig, model_name: str) -> BenchmarkAdapter:
    return LLMAdapterRegistry.create(
        LLMProvider.OLLAMA,
        model=model_name,
        base_url=config.ollama.host,
        keep_alive=config.ollama.keep_alive,
    )


def _declared_capabilities(adapter: BenchmarkAdapter) -> tuple[str, ...]:
    caps = getattr(adapter, "model_capabilities", None)
    if caps is None:
        return ()
    capabilities = getattr(caps, "capabilities", ())
    return tuple(sorted(capabilities))


def _classify_probe(attempt) -> SchemaProbeStatus:
    if attempt.failure_category == StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value:
        return SchemaProbeStatus.PROTOCOL_UNSUPPORTED
    if attempt.failure_category in {
        StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
        StructuralFailureCategory.INVALID_TOOL_ARGUMENTS.value,
        StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value,
        StructuralFailureCategory.CANONICAL_VALIDATION_FAILED.value,
    }:
        return SchemaProbeStatus.SCHEMA_INCOMPATIBLE
    if attempt.failure_category == StructuralFailureCategory.RESOURCE_LIMIT.value:
        return SchemaProbeStatus.PROVIDER_ERROR
    if attempt.ok:
        return SchemaProbeStatus.PASS
    return SchemaProbeStatus.PROVIDER_ERROR


def _case_status_from_attempt(attempt, semantic_passed: bool) -> CaseExecutionStatus:
    if attempt.failure_category == StructuralFailureCategory.RESOURCE_LIMIT.value:
        return CaseExecutionStatus.RESOURCE_LIMIT
    if not attempt.ok:
        if attempt.failure_category == StructuralFailureCategory.PROVIDER_ERROR.value:
            return CaseExecutionStatus.PROVIDER_ERROR
        return CaseExecutionStatus.FAIL
    if semantic_passed:
        return CaseExecutionStatus.PASS
    return CaseExecutionStatus.FAIL


def _qualification_status(
    *,
    config: QualificationConfig,
    schema_probe: SchemaProbeStatus,
    warmup_status: WarmupStatus,
    samples: int,
    semantic_success_rate: float,
    invalid_draft_count: int,
    provider_failure_count: int,
    unsafe_state_change_count: int,
) -> ProtocolStatus:
    if schema_probe != SchemaProbeStatus.PASS:
        if schema_probe == SchemaProbeStatus.PROTOCOL_UNSUPPORTED:
            return ProtocolStatus.PROTOCOL_UNSUPPORTED
        if schema_probe == SchemaProbeStatus.SCHEMA_INCOMPATIBLE:
            return ProtocolStatus.SCHEMA_INCOMPATIBLE
        return ProtocolStatus.PROVIDER_ERROR
    if warmup_status == WarmupStatus.FAILED:
        return ProtocolStatus.WARMUP_FAILED
    if samples < config.minimum_samples:
        return ProtocolStatus.NOT_QUALIFIED
    if (
        invalid_draft_count > config.maximum_invalid_drafts
        or provider_failure_count > config.maximum_provider_failures
        or unsafe_state_change_count > config.maximum_unsafe_state_changes
    ):
        return ProtocolStatus.NOT_QUALIFIED
    if semantic_success_rate >= config.qualified_semantic_success_rate:
        return ProtocolStatus.QUALIFIED
    if semantic_success_rate >= config.conditional_semantic_success_rate:
        return ProtocolStatus.CONDITIONALLY_QUALIFIED
    return ProtocolStatus.NOT_QUALIFIED


def _compare_protocols(left: ProtocolResult, right: ProtocolResult) -> int:
    for value in (
        qualification_rank(left.qualification_status) - qualification_rank(right.qualification_status),
        left.unsafe_state_change_count - right.unsafe_state_change_count,
        left.invalid_draft_count - right.invalid_draft_count,
        -1 if left.semantic_success_rate > right.semantic_success_rate else (
            1 if left.semantic_success_rate < right.semantic_success_rate else 0
        ),
        left.provider_failure_count - right.provider_failure_count,
        -1 if left.latency_ms.median < right.latency_ms.median else (
            1 if left.latency_ms.median > right.latency_ms.median else 0
        ),
    ):
        if value != 0:
            return value
    if left.protocol < right.protocol:
        return -1
    if left.protocol > right.protocol:
        return 1
    return 0


def _build_summary(models: tuple[ModelResult, ...]) -> QualificationSummary:
    qualified: list[tuple[str, ProtocolResult]] = []
    conditional: list[str] = []
    for model in models:
        best: ProtocolResult | None = None
        for protocol in model.protocols:
            if protocol.qualification_status == ProtocolStatus.CONDITIONALLY_QUALIFIED:
                conditional.append(f"{model.name} / {protocol.protocol}")
            if best is None or _compare_protocols(protocol, best) < 0:
                best = protocol
        if best is not None and best.qualification_status == ProtocolStatus.QUALIFIED:
            qualified.append((model.name, best))

    if not qualified:
        return QualificationSummary(
            recommended_model=None,
            recommended_protocol=None,
            conditional_candidates=tuple(sorted(conditional)),
            message=(
                "No tested model/protocol pair met the full LKW qualification threshold."
            ),
        )

    qualified.sort(key=lambda item: (_compare_protocols(item[1], qualified[0][1]), item[0]))
    model_name, protocol = qualified[0]
    return QualificationSummary(
        recommended_model=model_name,
        recommended_protocol=protocol.protocol,
        conditional_candidates=tuple(sorted(conditional)),
        message=(
            f"Recommended production configuration: {model_name} with "
            f"{protocol.protocol} (fully qualified)."
        ),
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _ordered_cases(config: LocalModelQualificationConfig) -> tuple[QualificationCase, ...]:
    cases = list(qualification_cases())
    if config.benchmark.randomize_case_order:
        raise ValueError("randomize_case_order is not supported")
    return tuple(cases)


def run_protocol_benchmark(
    *,
    config: LocalModelQualificationConfig,
    model: ModelConfig,
    protocol: str,
    adapter: BenchmarkAdapter,
    metadata: ModelMetadata,
    progress: Callable[[str], None] | None = None,
) -> ProtocolResult:
    emit = progress or (lambda _message: None)
    capability_supported = (
        adapter.supports_structured_output()
        if protocol == PROTOCOL_STRUCTURED_OUTPUT
        else adapter.supports_tools()
    )
    probe_case = case_by_id(_PROBE_CASE_ID)
    probe_attempt = run_protocol_attempt(
        adapter=adapter,
        protocol=protocol,
        request=probe_case.request,
        benchmark=config.benchmark,
        run_id=build_run_id(model.name, protocol, _PROBE_CASE_ID, 0),
    )
    schema_probe_status = _classify_probe(probe_attempt)
    emit(
        f"model={model.name} protocol={protocol} phase=probe status={schema_probe_status.value}"
    )
    if schema_probe_status != SchemaProbeStatus.PASS:
        return ProtocolResult(
            protocol=protocol,
            capability_supported=capability_supported,
            schema_probe_status=schema_probe_status,
            warmup_status=WarmupStatus.SKIPPED,
            qualification_status=(
                ProtocolStatus.PROTOCOL_UNSUPPORTED
                if schema_probe_status == SchemaProbeStatus.PROTOCOL_UNSUPPORTED
                else ProtocolStatus.SCHEMA_INCOMPATIBLE
                if schema_probe_status == SchemaProbeStatus.SCHEMA_INCOMPATIBLE
                else ProtocolStatus.PROVIDER_ERROR
            ),
            case_count=0,
            pass_count=0,
            failure_count=0,
            semantic_success_rate=0.0,
            invalid_draft_count=0,
            provider_failure_count=0,
            unsafe_state_change_count=0,
            failure_category_counts={},
            latency_ms=latency_stats([]),
            case_results=(),
        )

    warmup_status = WarmupStatus.PASS
    for warmup_index in range(1, config.benchmark.warmup_runs + 1):
        warmup_attempt = run_protocol_attempt(
            adapter=adapter,
            protocol=protocol,
            request=probe_case.request,
            benchmark=config.benchmark,
            run_id=build_run_id(model.name, protocol, _PROBE_CASE_ID, -warmup_index),
        )
        if not warmup_attempt.ok:
            warmup_status = WarmupStatus.FAILED
            break
    if warmup_status == WarmupStatus.FAILED:
        emit(f"model={model.name} protocol={protocol} qualification={ProtocolStatus.WARMUP_FAILED.value}")
        return ProtocolResult(
            protocol=protocol,
            capability_supported=capability_supported,
            schema_probe_status=schema_probe_status,
            warmup_status=warmup_status,
            qualification_status=ProtocolStatus.WARMUP_FAILED,
            case_count=0,
            pass_count=0,
            failure_count=0,
            semantic_success_rate=0.0,
            invalid_draft_count=0,
            provider_failure_count=0,
            unsafe_state_change_count=0,
            failure_category_counts={},
            latency_ms=latency_stats([]),
            case_results=(),
        )

    case_results: list[CaseResult] = []
    scored_latencies: list[float] = []
    failure_category_counts: Counter[str] = Counter()
    invalid_draft_count = 0
    provider_failure_count = 0
    unsafe_state_change_count = 0
    pass_count = 0

    for case in _ordered_cases(config):
        for repetition in range(1, config.benchmark.repetitions + 1):
            started = time.perf_counter()
            attempt = run_protocol_attempt(
                adapter=adapter,
                protocol=protocol,
                request=case.request,
                benchmark=config.benchmark,
                run_id=build_run_id(model.name, protocol, case.case_id, repetition),
            )
            semantic = None
            if attempt.ok and isinstance(attempt.plan, ConversationInteractionPlan):
                semantic = evaluate_semantics(attempt.plan, case)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            scored_latencies.append(elapsed_ms)

            failure_categories: tuple[str, ...] = ()
            primary_failure: str | None = None
            status = CaseExecutionStatus.FAIL
            action_types: tuple[str, ...] = ()
            object_types: tuple[str, ...] = ()
            workspace_refs = ()
            clarification_count = 0
            unsafe_count = 0

            if attempt.failure_category in _INVALID_DRAFT_CATEGORIES:
                invalid_draft_count += 1
                failure_categories = (attempt.failure_category,)
                primary_failure = attempt.failure_category
                failure_category_counts[attempt.failure_category] += 1
            elif attempt.failure_category == StructuralFailureCategory.PROVIDER_ERROR.value:
                provider_failure_count += 1
                failure_categories = (attempt.failure_category,)
                primary_failure = attempt.failure_category
                failure_category_counts[attempt.failure_category] += 1
            elif attempt.failure_category:
                failure_categories = (attempt.failure_category,)
                primary_failure = attempt.failure_category
                failure_category_counts[attempt.failure_category] += 1

            if semantic is not None:
                action_types = semantic.action_types
                object_types = semantic.object_types
                workspace_refs = semantic.workspace_reference_summaries
                clarification_count = semantic.clarification_count
                unsafe_count = semantic.unsafe_state_change_count
                unsafe_state_change_count += semantic.unsafe_state_change_count
                if not semantic.passed:
                    for category in semantic.failure_categories:
                        failure_category_counts[category] += 1
                    failure_categories = semantic.failure_categories
                    primary_failure = semantic.primary_failure_category

            status = _case_status_from_attempt(attempt, semantic.passed if semantic else False)
            if status == CaseExecutionStatus.PASS:
                pass_count += 1

            emit(
                f"model={model.name} protocol={protocol} case={case.case_id} "
                f"repeat={repetition} status={status.value} latency_ms={elapsed_ms:.1f}"
            )
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    repetition=repetition,
                    status=status,
                    primary_failure_category=primary_failure,
                    failure_categories=failure_categories,
                    latency_ms=elapsed_ms,
                    action_types=action_types,
                    object_types=object_types,
                    workspace_references=workspace_refs,
                    clarification_count=clarification_count,
                    unsafe_state_change_count=unsafe_count,
                    error_type=attempt.error_type,
                )
            )

    case_count = len(case_results)
    failure_count = case_count - pass_count
    semantic_success_rate = pass_count / case_count if case_count else 0.0
    qualification_status = _qualification_status(
        config=config.qualification,
        schema_probe=schema_probe_status,
        warmup_status=warmup_status,
        samples=case_count,
        semantic_success_rate=semantic_success_rate,
        invalid_draft_count=invalid_draft_count,
        provider_failure_count=provider_failure_count,
        unsafe_state_change_count=unsafe_state_change_count,
    )
    emit(f"model={model.name} protocol={protocol} qualification={qualification_status.value}")
    return ProtocolResult(
        protocol=protocol,
        capability_supported=capability_supported,
        schema_probe_status=schema_probe_status,
        warmup_status=warmup_status,
        qualification_status=qualification_status,
        case_count=case_count,
        pass_count=pass_count,
        failure_count=failure_count,
        semantic_success_rate=semantic_success_rate,
        invalid_draft_count=invalid_draft_count,
        provider_failure_count=provider_failure_count,
        unsafe_state_change_count=unsafe_state_change_count,
        failure_category_counts=dict(sorted(failure_category_counts.items())),
        latency_ms=latency_stats(scored_latencies),
        case_results=tuple(case_results),
    )


def run_benchmark(
    config: LocalModelQualificationConfig,
    *,
    progress: Callable[[str], None] | None = None,
    client_factory: Callable | None = None,
    adapter_factory: Callable[[str], BenchmarkAdapter] | None = None,
    generated_at_utc: str | None = None,
    generated_from_commit: str | None = None,
) -> LocalModelQualificationResult:
    emit = progress or (lambda _message: None)
    host = collect_host_metadata()
    ollama = collect_ollama_environment(config.ollama, client_factory=client_factory)
    installed = list_installed_models(config.ollama, client_factory=client_factory)
    model_results: list[ModelResult] = []

    create_adapter = adapter_factory or (lambda model_name: _create_adapter(config, model_name))

    for model_cfg in config.models:
        if not model_cfg.enabled:
            continue
        if model_cfg.name not in installed:
            if config.ollama.pull_missing_models:
                try:
                    pull_model(config.ollama, model_cfg.name, client_factory=client_factory)
                    installed = list_installed_models(config.ollama, client_factory=client_factory)
                except Exception:
                    model_results.append(
                        ModelResult(
                            name=model_cfg.name,
                            role=model_cfg.role,
                            installed=False,
                            metadata=ModelMetadata(),
                            declared_capabilities=(),
                            observed_execution_mode=ObservedExecutionMode.UNKNOWN,
                            status=ModelStatus.PULL_FAILED,
                            protocols=(),
                        )
                    )
                    continue
            else:
                model_results.append(
                    ModelResult(
                        name=model_cfg.name,
                        role=model_cfg.role,
                        installed=False,
                        metadata=ModelMetadata(),
                        declared_capabilities=(),
                        observed_execution_mode=ObservedExecutionMode.UNKNOWN,
                        status=ModelStatus.NOT_INSTALLED,
                        protocols=(),
                    )
                )
                continue

        try:
            metadata = fetch_model_metadata(
                config.ollama,
                model_cfg.name,
                client_factory=client_factory,
            )
            adapter = create_adapter(model_cfg.name)
            declared = _declared_capabilities(adapter)
            execution_mode = derive_execution_mode(metadata)
            protocol_results: list[ProtocolResult] = []
            model_had_failure = False
            for protocol in enabled_protocols(config):
                protocol_result = run_protocol_benchmark(
                    config=config,
                    model=model_cfg,
                    protocol=protocol,
                    adapter=adapter,
                    metadata=metadata,
                    progress=emit,
                )
                protocol_results.append(protocol_result)
                if protocol_result.qualification_status not in {
                    ProtocolStatus.QUALIFIED,
                    ProtocolStatus.CONDITIONALLY_QUALIFIED,
                    ProtocolStatus.NOT_QUALIFIED,
                }:
                    model_had_failure = True
            status = (
                ModelStatus.COMPLETED_WITH_FAILURES
                if model_had_failure or any(
                    result.failure_count > 0 for result in protocol_results
                )
                else ModelStatus.COMPLETED
            )
            model_results.append(
                ModelResult(
                    name=model_cfg.name,
                    role=model_cfg.role,
                    installed=True,
                    metadata=metadata,
                    declared_capabilities=declared,
                    observed_execution_mode=execution_mode,
                    status=status,
                    protocols=tuple(protocol_results),
                )
            )
        except Exception:
            model_results.append(
                ModelResult(
                    name=model_cfg.name,
                    role=model_cfg.role,
                    installed=True,
                    metadata=ModelMetadata(),
                    declared_capabilities=(),
                    observed_execution_mode=ObservedExecutionMode.UNKNOWN,
                    status=ModelStatus.PROVIDER_UNAVAILABLE,
                    protocols=(),
                )
            )
            if not config.ollama.continue_on_model_error:
                raise

    models_tuple = tuple(model_results)
    summary = _build_summary(models_tuple)
    return LocalModelQualificationResult(
        generated_at_utc=generated_at_utc or datetime.now(UTC).isoformat(),
        generated_from_commit=generated_from_commit or _git_commit(),
        configuration_sha256=configuration_sha256(config),
        host=host,
        ollama=ollama,
        models=models_tuple,
        summary=summary,
    )


def write_artifacts(config: LocalModelQualificationConfig, result: LocalModelQualificationResult) -> None:
    json_content = serialize_result_json(result)
    markdown_content = render_markdown(result)
    _atomic_write(config.results_json_path, json_content)
    _atomic_write(config.report_markdown_path, markdown_content)


def _has_partial_model_failure(result: LocalModelQualificationResult) -> bool:
    partial_model_statuses = {
        ModelStatus.NOT_INSTALLED,
        ModelStatus.PULL_FAILED,
        ModelStatus.MODEL_METADATA_UNAVAILABLE,
        ModelStatus.RESOURCE_LIMIT,
        ModelStatus.PROVIDER_UNAVAILABLE,
    }
    partial_protocol_statuses = {
        ProtocolStatus.PROTOCOL_UNSUPPORTED,
        ProtocolStatus.SCHEMA_INCOMPATIBLE,
        ProtocolStatus.WARMUP_FAILED,
        ProtocolStatus.PROVIDER_ERROR,
        ProtocolStatus.RESOURCE_LIMIT,
        ProtocolStatus.NOT_RUN,
    }
    for model in result.models:
        if model.status in partial_model_statuses:
            return True
        for protocol in model.protocols:
            if protocol.qualification_status in partial_protocol_statuses:
                return True
    return False


def run_from_config(config: LocalModelQualificationConfig) -> tuple[LocalModelQualificationResult, int]:
    result = run_benchmark(config, progress=lambda message: print(message, flush=True))
    write_artifacts(config, result)
    exit_code = 2 if _has_partial_model_failure(result) else 0
    return result, exit_code


def main() -> int:
    try:
        config = load_config()
        result, exit_code = run_from_config(config)
        repo_root = config.application_root.parent.parent
        results_rel = os.path.relpath(config.results_json_path, repo_root)
        report_rel = os.path.relpath(config.report_markdown_path, repo_root)
        print(f"benchmark_status=COMPLETED")
        print(f"benchmark_exit_code={exit_code}")
        print(f"results_json={results_rel}")
        print(f"report_markdown={report_rel}")
        return exit_code
    except Exception as exc:
        print("benchmark_status=FAILED")
        print(f"benchmark_error={type(exc).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
