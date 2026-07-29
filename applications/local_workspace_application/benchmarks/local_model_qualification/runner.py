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
from functools import cmp_to_key
from pathlib import Path
from typing import Callable

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

from local_workspace_application.benchmarks.local_model_qualification.config import (
    LocalModelQualificationConfig,
    ModelConfig,
    QualificationConfig,
    configuration_sha256,
    enabled_model_names,
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
    ProvisioningResult,
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
    build_inventory_metadata,
    collect_host_metadata,
    collect_ollama_environment,
    derive_execution_mode,
    fetch_model_inventory,
    fetch_runtime_metadata,
    fetch_show_metadata,
    merge_runtime_metadata,
)
from local_workspace_application.benchmarks.local_model_qualification.evaluator import evaluate_semantics
from local_workspace_application.benchmarks.local_model_qualification.protocols import (
    PROTOCOL_SINGLE_PLAN_TOOL,
    PROTOCOL_STRUCTURED_OUTPUT,
    BenchmarkAdapter,
    ProtocolAttemptSuccess,
    run_protocol_attempt,
)
from local_workspace_application.benchmarks.local_model_qualification.provisioning import (
    ProvisioningError,
    provision_ollama_runtime,
)
from local_workspace_application.benchmarks.local_model_qualification.report import (
    compare_candidates,
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
_SCHEMA_INCOMPATIBLE_CATEGORIES = frozenset(
    {
        StructuralFailureCategory.MISSING_PLAN_TOOL_CALL.value,
        StructuralFailureCategory.MULTIPLE_PLAN_TOOL_CALLS.value,
        StructuralFailureCategory.UNEXPECTED_PLAN_TOOL.value,
        StructuralFailureCategory.INVALID_TOOL_ARGUMENTS.value,
        StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
        StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value,
        StructuralFailureCategory.CANONICAL_VALIDATION_FAILED.value,
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


def expected_scored_call_count(config: LocalModelQualificationConfig) -> int:
    enabled_models = len(enabled_model_names(config))
    return (
        enabled_models
        * len(enabled_protocols(config))
        * len(qualification_cases())
        * config.benchmark.repetitions
    )


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


def _classify_probe(attempt: ProtocolAttemptSuccess) -> SchemaProbeStatus:
    if attempt.failure_category == StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value:
        return SchemaProbeStatus.PROTOCOL_UNSUPPORTED
    if attempt.failure_category in _SCHEMA_INCOMPATIBLE_CATEGORIES:
        return SchemaProbeStatus.SCHEMA_INCOMPATIBLE
    if attempt.failure_category == StructuralFailureCategory.RESOURCE_LIMIT.value:
        return SchemaProbeStatus.RESOURCE_LIMIT
    if attempt.ok:
        return SchemaProbeStatus.PASS
    return SchemaProbeStatus.PROVIDER_ERROR


def _probe_failure_counts(attempt: ProtocolAttemptSuccess, status: SchemaProbeStatus) -> tuple[int, dict[str, int]]:
    if status == SchemaProbeStatus.PASS:
        return 0, {}
    category = attempt.failure_category or StructuralFailureCategory.PROVIDER_ERROR.value
    counts = {category: 1}
    provider_failures = 1 if status == SchemaProbeStatus.PROVIDER_ERROR else 0
    return provider_failures, counts


def _probe_protocol_status(status: SchemaProbeStatus) -> ProtocolStatus:
    if status == SchemaProbeStatus.PROTOCOL_UNSUPPORTED:
        return ProtocolStatus.PROTOCOL_UNSUPPORTED
    if status == SchemaProbeStatus.SCHEMA_INCOMPATIBLE:
        return ProtocolStatus.SCHEMA_INCOMPATIBLE
    if status == SchemaProbeStatus.RESOURCE_LIMIT:
        return ProtocolStatus.RESOURCE_LIMIT
    return ProtocolStatus.PROVIDER_ERROR


def _case_status_from_attempt(attempt: ProtocolAttemptSuccess, semantic_passed: bool) -> CaseExecutionStatus:
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
        return _probe_protocol_status(schema_probe)
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


def _build_summary(
    config: LocalModelQualificationConfig,
    models: tuple[ModelResult, ...],
    provisioning: ProvisioningResult,
    actual_scored_calls: int,
) -> QualificationSummary:
    qualified: list[tuple[str, ProtocolResult]] = []
    conditional: list[str] = []
    attempted_pairs = 0
    for model in models:
        for protocol in model.protocols:
            attempted_pairs += 1
            if protocol.qualification_status == ProtocolStatus.CONDITIONALLY_QUALIFIED:
                conditional.append(f"{model.name} / {protocol.protocol}")
            if protocol.qualification_status == ProtocolStatus.QUALIFIED:
                qualified.append((model.name, protocol))

    required_model_count = len(provisioning.required_models)
    expected_pairs = required_model_count * len(enabled_protocols(config))
    expected_calls = expected_scored_call_count(config)

    if not qualified:
        return QualificationSummary(
            recommended_model=None,
            recommended_protocol=None,
            conditional_candidates=tuple(sorted(conditional)),
            message="No tested model/protocol pair met the full LKW qualification threshold.",
            required_model_count=required_model_count,
            provisioned_model_count=len(provisioning.models),
            attempted_model_protocol_pairs=attempted_pairs,
            expected_model_protocol_pairs=expected_pairs,
            expected_scored_call_count=expected_calls,
            actual_scored_call_count=actual_scored_calls,
        )

    qualified.sort(key=cmp_to_key(compare_candidates))
    model_name, protocol = qualified[0]
    return QualificationSummary(
        recommended_model=model_name,
        recommended_protocol=protocol.protocol,
        conditional_candidates=tuple(sorted(conditional)),
        message=(
            f"Recommended production configuration: {model_name} with "
            f"{protocol.protocol} (fully qualified)."
        ),
        required_model_count=required_model_count,
        provisioned_model_count=len(provisioning.models),
        attempted_model_protocol_pairs=attempted_pairs,
        expected_model_protocol_pairs=expected_pairs,
        expected_scored_call_count=expected_calls,
        actual_scored_call_count=actual_scored_calls,
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


def _failed_probe_result(
    *,
    protocol: str,
    capability_supported: bool,
    schema_probe_status: SchemaProbeStatus,
    probe_attempt: ProtocolAttemptSuccess,
    probe_latency_ms: float,
) -> ProtocolResult:
    provider_failure_count, failure_category_counts = _probe_failure_counts(
        probe_attempt,
        schema_probe_status,
    )
    return ProtocolResult(
        protocol=protocol,
        capability_supported=capability_supported,
        schema_probe_status=schema_probe_status,
        warmup_status=WarmupStatus.SKIPPED,
        qualification_status=_probe_protocol_status(schema_probe_status),
        case_count=0,
        pass_count=0,
        failure_count=0,
        semantic_success_rate=0.0,
        invalid_draft_count=0,
        provider_failure_count=provider_failure_count,
        unsafe_state_change_count=0,
        failure_category_counts=failure_category_counts,
        latency_ms=latency_stats([]),
        case_results=(),
        probe_failure_category=probe_attempt.failure_category,
        probe_failure_phase=probe_attempt.failure_phase,
        probe_error_type=probe_attempt.error_type,
        probe_safe_error_code=probe_attempt.safe_error_code,
        probe_latency_ms=probe_latency_ms,
    )


def run_protocol_benchmark(
    *,
    config: LocalModelQualificationConfig,
    model: ModelConfig,
    protocol: str,
    adapter: BenchmarkAdapter,
    progress: Callable[[str], None] | None = None,
) -> ProtocolResult:
    emit = progress or (lambda _message: None)
    capability_supported = (
        adapter.supports_structured_output()
        if protocol == PROTOCOL_STRUCTURED_OUTPUT
        else adapter.supports_tools()
    )
    probe_case = case_by_id(_PROBE_CASE_ID)
    probe_started = time.perf_counter()
    probe_attempt = run_protocol_attempt(
        adapter=adapter,
        protocol=protocol,
        request=probe_case.request,
        benchmark=config.benchmark,
        run_id=build_run_id(model.name, protocol, _PROBE_CASE_ID, 0),
    )
    probe_latency_ms = (time.perf_counter() - probe_started) * 1000.0
    schema_probe_status = _classify_probe(probe_attempt)
    emit(
        f"model={model.name} protocol={protocol} phase=probe status={schema_probe_status.value}"
    )
    if schema_probe_status != SchemaProbeStatus.PASS:
        return _failed_probe_result(
            protocol=protocol,
            capability_supported=capability_supported,
            schema_probe_status=schema_probe_status,
            probe_attempt=probe_attempt,
            probe_latency_ms=probe_latency_ms,
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
            probe_failure_category=None,
            probe_failure_phase=None,
            probe_error_type=None,
            probe_safe_error_code=None,
            probe_latency_ms=probe_latency_ms,
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
            elif attempt.failure_category == StructuralFailureCategory.RESOURCE_LIMIT.value:
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
        probe_failure_category=None,
        probe_failure_phase=None,
        probe_error_type=None,
        probe_safe_error_code=None,
        probe_latency_ms=probe_latency_ms,
    )


def run_benchmark(
    config: LocalModelQualificationConfig,
    provisioning: ProvisioningResult,
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
    inventory = fetch_model_inventory(config.ollama, client_factory=client_factory)
    model_results: list[ModelResult] = []
    actual_scored_calls = 0

    create_adapter = adapter_factory or (lambda model_name: _create_adapter(config, model_name))

    for model_cfg in config.models:
        if not model_cfg.enabled:
            continue
        try:
            inventory_record = inventory[model_cfg.name]
            show = fetch_show_metadata(
                config.ollama,
                model_cfg.name,
                client_factory=client_factory,
            )
            metadata = build_inventory_metadata(inventory_record, show)
            adapter = create_adapter(model_cfg.name)
            declared = _declared_capabilities(adapter)
            protocol_results: list[ProtocolResult] = []
            model_had_failure = False
            for protocol in enabled_protocols(config):
                protocol_result = run_protocol_benchmark(
                    config=config,
                    model=model_cfg,
                    protocol=protocol,
                    adapter=adapter,
                    progress=emit,
                )
                protocol_results.append(protocol_result)
                actual_scored_calls += protocol_result.case_count
                if protocol_result.qualification_status not in {
                    ProtocolStatus.QUALIFIED,
                    ProtocolStatus.CONDITIONALLY_QUALIFIED,
                    ProtocolStatus.NOT_QUALIFIED,
                }:
                    model_had_failure = True

            loaded_size, size_vram = fetch_runtime_metadata(
                config.ollama,
                model_cfg.name,
                client_factory=client_factory,
            )
            metadata = merge_runtime_metadata(metadata, loaded_size, size_vram)
            execution_mode = derive_execution_mode(metadata)

            status = (
                ModelStatus.COMPLETED_WITH_FAILURES
                if model_had_failure or any(result.failure_count > 0 for result in protocol_results)
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
    summary = _build_summary(config, models_tuple, provisioning, actual_scored_calls)
    return LocalModelQualificationResult(
        generated_at_utc=generated_at_utc or datetime.now(UTC).isoformat(),
        generated_from_commit=generated_from_commit or _git_commit(),
        configuration_sha256=configuration_sha256(config),
        host=host,
        ollama=ollama,
        provisioning=provisioning,
        models=models_tuple,
        summary=summary,
    )


def write_artifacts(config: LocalModelQualificationConfig, result: LocalModelQualificationResult) -> None:
    json_content = serialize_result_json(result)
    markdown_content = render_markdown(result)
    _atomic_write(config.results_json_path, json_content)
    _atomic_write(config.report_markdown_path, markdown_content)


def compute_exit_code(result: LocalModelQualificationResult) -> int:
    runtime_failure_statuses = {
        ProtocolStatus.PROVIDER_ERROR,
        ProtocolStatus.RESOURCE_LIMIT,
        ProtocolStatus.WARMUP_FAILED,
    }
    partial_model_statuses = {
        ModelStatus.MODEL_METADATA_UNAVAILABLE,
        ModelStatus.RESOURCE_LIMIT,
        ModelStatus.PROVIDER_UNAVAILABLE,
    }
    for model in result.models:
        if model.status in partial_model_statuses:
            return 2
        for protocol in model.protocols:
            if protocol.qualification_status in runtime_failure_statuses:
                return 2
    return 0


def run_from_config(
    config: LocalModelQualificationConfig,
    *,
    provision: Callable[[LocalModelQualificationConfig], ProvisioningResult] | None = None,
) -> tuple[LocalModelQualificationResult, int]:
    provisioner = provision or provision_ollama_runtime
    provisioning = provisioner(config, progress=lambda message: print(message, flush=True))
    result = run_benchmark(config, provisioning, progress=lambda message: print(message, flush=True))
    write_artifacts(config, result)
    return result, compute_exit_code(result)


def main() -> int:
    try:
        config = load_config()
        result, exit_code = run_from_config(config)
        repo_root = config.repository_root
        results_rel = os.path.relpath(config.results_json_path, repo_root)
        report_rel = os.path.relpath(config.report_markdown_path, repo_root)
        print("benchmark_status=COMPLETED")
        print(f"benchmark_exit_code={exit_code}")
        print(f"results_json={results_rel}")
        print(f"report_markdown={report_rel}")
        return exit_code
    except ProvisioningError as exc:
        print("benchmark_status=FAILED")
        print(f"benchmark_error={exc.code}")
        return 1
    except Exception as exc:
        print("benchmark_status=FAILED")
        print(f"benchmark_error={type(exc).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
