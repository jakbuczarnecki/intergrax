# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

from local_workspace_application.benchmarks.local_model_qualification.config import (
    QualificationConfig,
    load_config,
)
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    BENCHMARK_ID,
    CORPUS_VERSION,
    CaseExecutionStatus,
    CaseResult,
    FailurePhase,
    HostMetadata,
    LatencyStats,
    LocalModelQualificationResult,
    ModelMetadata,
    ModelResult,
    ModelStatus,
    ModelProvisioningStatus,
    ObservedExecutionMode,
    OllamaEnvironment,
    ProtocolResult,
    ProtocolStatus,
    ProvisionedModel,
    ProvisioningResult,
    QualificationSummary,
    RESULT_SCHEMA_VERSION,
    SafeErrorCode,
    SchemaProbeStatus,
    StructuralFailureCategory,
    WarmupStatus,
)
from local_workspace_application.benchmarks.local_model_qualification.report import (
    compare_candidates,
    contains_secret_patterns,
    render_markdown,
    serialize_result_json,
    sort_qualified_candidates,
)
from local_workspace_application.benchmarks.local_model_qualification.runner import (
    _qualification_status,
    build_model_failure_protocol_results,
    latency_stats,
    median_deterministic,
    percentile_nearest_rank,
)


def _provisioning() -> ProvisioningResult:
    return ProvisioningResult(
        runtime="docker",
        compose_file="../../../infra/docker/ollama/docker-compose.yml",
        compose_service="ollama",
        container_name="intergrax-ollama",
        persistent_model_volume="intergrax-ollama-models",
        readiness_result="READY",
        required_models=("qwen2.5:14b",),
        models=(
            ProvisionedModel(model="qwen2.5:14b", status=ModelProvisioningStatus.ALREADY_AVAILABLE),
        ),
    )


def _protocol(**overrides) -> ProtocolResult:
    base = dict(
        protocol="structured_output",
        capability_supported=True,
        schema_probe_status=SchemaProbeStatus.PASS,
        warmup_status=WarmupStatus.PASS,
        qualification_status=ProtocolStatus.NOT_QUALIFIED,
        case_count=1,
        pass_count=0,
        failure_count=1,
        semantic_success_rate=0.0,
        invalid_draft_count=0,
        provider_failure_count=0,
        unsafe_state_change_count=0,
        failure_category_counts={"SEMANTIC_MISMATCH": 1},
        latency_ms=LatencyStats(minimum=10.0, median=20.0, p95=30.0, maximum=40.0),
        case_results=(
            CaseResult(
                case_id="planner.workspace_list",
                repetition=1,
                status=CaseExecutionStatus.FAIL,
                latency_ms=20.0,
            ),
        ),
    )
    base.update(overrides)
    return ProtocolResult(**base)


def _sample_result(*, digest: str | None = "sha256:abc", execution_mode=ObservedExecutionMode.UNKNOWN) -> LocalModelQualificationResult:
    protocol = _protocol()
    return LocalModelQualificationResult(
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="abc123",
        configuration_sha256="deadbeef",
        host=HostMetadata(
            operating_system="Windows",
            os_release="11",
            machine_architecture="AMD64",
            python_version="3.12.0",
        ),
        ollama=OllamaEnvironment(version="0.5.0", host="http://localhost:11434"),
        provisioning=_provisioning(),
        models=(
            ModelResult(
                name="qwen2.5:14b",
                role="baseline",
                installed=True,
                metadata=ModelMetadata(digest=digest, artifact_size_bytes=1000),
                declared_capabilities=("tools",),
                observed_execution_mode=execution_mode,
                status=ModelStatus.COMPLETED_WITH_FAILURES,
                protocols=(protocol,),
            ),
        ),
        summary=QualificationSummary(
            recommended_model=None,
            recommended_protocol=None,
            conditional_candidates=(),
            message="No tested model/protocol pair met the full LKW qualification threshold.",
            required_model_count=1,
            provisioned_model_count=1,
            attempted_model_protocol_pairs=1,
            expected_model_protocol_pairs=1,
            expected_scored_call_count=36,
            actual_scored_call_count=1,
        ),
    )


def test_generated_file_warning_present() -> None:
    text = render_markdown(_sample_result())
    assert "GENERATED FILE" in text


def test_docker_provisioning_section_exists() -> None:
    text = render_markdown(_sample_result())
    assert "## Docker Ollama provisioning" in text
    assert "ALREADY_AVAILABLE" in text


def test_digest_claim_conditional_when_missing() -> None:
    text = render_markdown(_sample_result(digest=None))
    assert "One or more model digests were unavailable" in text


def test_digest_claim_when_all_available() -> None:
    text = render_markdown(_sample_result(digest="sha256:abc"))
    assert "Results apply to the exact model tags, digests" in text


def test_probe_diagnostics_rendered() -> None:
    result = _sample_result()
    failed = _protocol(
        schema_probe_status=SchemaProbeStatus.PROVIDER_ERROR,
        probe_failure_category="PROVIDER_ERROR",
        probe_failure_phase="PROVIDER_INVOKE",
        probe_safe_error_code="OLLAMA_PROVIDER_TRANSPORT_FAILED",
        provider_failure_count=1,
        failure_category_counts={"PROVIDER_ERROR": 1},
        qualification_status=ProtocolStatus.PROVIDER_ERROR,
    )
    model = result.models[0].model_copy(update={"protocols": (failed,)})
    text = render_markdown(result.model_copy(update={"models": (model,)}))
    assert "PROVIDER_INVOKE" in text
    assert "OLLAMA_PROVIDER_TRANSPORT_FAILED" in text


def test_expected_and_actual_call_counts_rendered() -> None:
    text = render_markdown(_sample_result())
    assert "Expected scored calls" in text
    assert "Actual scored calls" in text


def test_not_installed_absent_from_successful_report() -> None:
    text = render_markdown(_sample_result())
    assert "NOT_INSTALLED" not in text


def test_full_gpu_claim_requires_measured_mode() -> None:
    text = render_markdown(_sample_result(execution_mode=ObservedExecutionMode.FULL_GPU))
    assert "FULL_GPU (measured)" in text


def test_byte_stable_rendering() -> None:
    result = _sample_result()
    assert render_markdown(result) == render_markdown(result)
    assert serialize_result_json(result) == serialize_result_json(result)


def test_secret_needles_rejected() -> None:
    assert contains_secret_patterns("Authorization: Bearer secret")
    assert not contains_secret_patterns("safe benchmark summary")


def test_true_total_ordering_selects_best_metric_candidate() -> None:
    candidates = [
        (
            "zebra-model",
            _protocol(
                qualification_status=ProtocolStatus.QUALIFIED,
                semantic_success_rate=0.95,
                latency_ms=LatencyStats(minimum=1.0, median=100.0, p95=100.0, maximum=100.0),
            ),
        ),
        (
            "alpha-model",
            _protocol(
                qualification_status=ProtocolStatus.QUALIFIED,
                semantic_success_rate=1.0,
                latency_ms=LatencyStats(minimum=1.0, median=50.0, p95=50.0, maximum=50.0),
            ),
        ),
        (
            "beta-model",
            _protocol(
                qualification_status=ProtocolStatus.QUALIFIED,
                semantic_success_rate=0.99,
                latency_ms=LatencyStats(minimum=1.0, median=10.0, p95=10.0, maximum=10.0),
            ),
        ),
    ]
    best = sort_qualified_candidates(candidates)[0]
    assert best[0] == "alpha-model"
    assert compare_candidates(candidates[1], candidates[0]) < 0


def test_serialize_schema_versions() -> None:
    payload = _sample_result().to_json_dict()
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["benchmark_id"] == BENCHMARK_ID
    assert payload["corpus_version"] == CORPUS_VERSION
    assert "provisioning" in payload


def test_qualification_policy() -> None:
    cfg = QualificationConfig(
        minimum_samples=30,
        qualified_semantic_success_rate=1.0,
        conditional_semantic_success_rate=0.9,
        maximum_invalid_drafts=0,
        maximum_provider_failures=0,
        maximum_unsafe_state_changes=0,
    )
    qualified = _qualification_status(
        config=cfg,
        schema_probe=SchemaProbeStatus.PASS,
        warmup_status=WarmupStatus.PASS,
        samples=30,
        semantic_success_rate=1.0,
        invalid_draft_count=0,
        provider_failure_count=0,
        unsafe_state_change_count=0,
    )
    assert qualified == ProtocolStatus.QUALIFIED


_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "local-model-qualification.toml"
)


def test_model_preparation_failure_rows_rendered() -> None:
    config = load_config(_CONFIG)
    protocols = build_model_failure_protocol_results(
        config=config,
        category=StructuralFailureCategory.PROVIDER_ERROR,
        phase=FailurePhase.ADAPTER_CONSTRUCTION,
        error_type="RuntimeError",
        safe_error_code=SafeErrorCode.OLLAMA_ADAPTER_CONSTRUCTION_FAILED,
    )
    result = _sample_result()
    model = result.models[0].model_copy(update={"protocols": protocols})
    text = render_markdown(result.model_copy(update={"models": (model,)}))
    assert "ADAPTER_CONSTRUCTION" in text
    assert "OLLAMA_ADAPTER_CONSTRUCTION_FAILED" in text
    assert "SKIPPED" in text


def test_warmup_diagnostics_rendered() -> None:
    failed = _protocol(
        warmup_status=WarmupStatus.FAILED,
        qualification_status=ProtocolStatus.WARMUP_FAILED,
        warmup_failure_category="MISSING_PLAN_TOOL_CALL",
        warmup_failure_phase="TOOL_CALL_VALIDATION",
        warmup_safe_error_code="n/a",
        warmup_failure_repetition=1,
        warmup_failure_latency_ms=42.5,
        failure_category_counts={"MISSING_PLAN_TOOL_CALL": 1},
    )
    result = _sample_result()
    model = result.models[0].model_copy(update={"protocols": (failed,)})
    text = render_markdown(result.model_copy(update={"models": (model,)}))
    assert "Warmup failure category: MISSING_PLAN_TOOL_CALL" in text
    assert "Warmup failure phase: TOOL_CALL_VALIDATION" in text
    assert "Warmup failure repetition: 1" in text
    assert "Warmup failure latency: 42.5" in text
    assert "| FAILED |" in text or "Warmup status: FAILED" in text


def test_successful_warmup_renders_na_diagnostics() -> None:
    passed = _protocol(warmup_status=WarmupStatus.PASS)
    result = _sample_result()
    model = result.models[0].model_copy(update={"protocols": (passed,)})
    text = render_markdown(result.model_copy(update={"models": (model,)}))
    assert "Warmup failure category: n/a" in text
    assert "Warmup failure phase: n/a" in text
    assert "Warmup safe error code: n/a" in text


def test_raw_exception_message_absent_from_report() -> None:
    protocols = build_model_failure_protocol_results(
        config=load_config(_CONFIG),
        category=StructuralFailureCategory.PROVIDER_ERROR,
        phase=FailurePhase.ADAPTER_CONSTRUCTION,
        error_type="RuntimeError",
        safe_error_code=SafeErrorCode.OLLAMA_ADAPTER_CONSTRUCTION_FAILED,
    )
    result = _sample_result()
    model = result.models[0].model_copy(update={"protocols": protocols})
    text = render_markdown(result.model_copy(update={"models": (model,)}))
    assert "adapter construction failed" not in text.lower()
    assert "super secret" not in text.lower()


def markdown_cell_count(row: str) -> int:
    stripped = row.strip()
    assert stripped.startswith("|")
    assert stripped.endswith("|")
    return len(stripped.split("|")) - 2


def test_model_protocol_comparison_table_column_counts() -> None:
    structured = _protocol(protocol="structured_output")
    tools = _protocol(protocol="single_plan_tool")
    result = _sample_result()
    model = result.models[0].model_copy(update={"protocols": (structured, tools)})
    text = render_markdown(result.model_copy(update={"models": (model,)}))
    lines = text.splitlines()
    section_index = next(
        index for index, line in enumerate(lines) if line.strip() == "## 8. Model × protocol comparison"
    )
    header = lines[section_index + 2]
    separator = lines[section_index + 3]
    data_rows = [
        line
        for line in lines[section_index + 4 :]
        if line.startswith("| ") and not line.startswith("| ---")
    ]
    assert markdown_cell_count(header) == 18
    assert markdown_cell_count(separator) == 18
    assert data_rows
    for row in data_rows:
        if row.strip() == "":
            break
        assert markdown_cell_count(row) == 18
    assert all(markdown_cell_count(row) == 18 for row in data_rows if row.strip())
