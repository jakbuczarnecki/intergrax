# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    BENCHMARK_ID,
    CORPUS_VERSION,
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
    RESULT_SCHEMA_VERSION,
    SchemaProbeStatus,
    WarmupStatus,
)
from local_workspace_application.benchmarks.local_model_qualification.report import (
    contains_secret_patterns,
    render_markdown,
    serialize_result_json,
)
from local_workspace_application.benchmarks.local_model_qualification.runner import (
    _qualification_status,
    latency_stats,
    median_deterministic,
    percentile_nearest_rank,
)
from local_workspace_application.benchmarks.local_model_qualification.config import QualificationConfig


def _sample_result() -> LocalModelQualificationResult:
    protocol = ProtocolResult(
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
        models=(
            ModelResult(
                name="qwen2.5:14b",
                role="baseline",
                installed=True,
                metadata=ModelMetadata(digest="sha256:abc"),
                declared_capabilities=("tools",),
                observed_execution_mode=ObservedExecutionMode.FULL_GPU,
                status=ModelStatus.COMPLETED_WITH_FAILURES,
                protocols=(protocol,),
            ),
        ),
        summary=QualificationSummary(
            recommended_model=None,
            recommended_protocol=None,
            conditional_candidates=(),
            message="No tested model/protocol pair met the full LKW qualification threshold.",
        ),
    )


def test_generated_file_warning_present() -> None:
    text = render_markdown(_sample_result())
    assert "GENERATED FILE" in text


def test_normal_invocation_present() -> None:
    text = render_markdown(_sample_result())
    assert "run-local-model-qualification.py" in text


def test_required_sections_present() -> None:
    text = render_markdown(_sample_result())
    for section in (
        "## 1. Scope and interpretation",
        "## 2. Executive summary",
        "## 3. Recommended configuration",
        "## 4. Benchmark methodology",
        "## 5. Benchmark host",
        "## 6. Ollama environment",
        "## 7. Tested model inventory",
        "## 8. Model × protocol comparison",
        "## 9. Safety and state-change results",
        "## 10. Failure categories",
        "## 11. Per-model details",
        "## 12. Reproduction",
        "## 13. Limitations",
    ):
        assert section in text


def test_model_protocol_rows_rendered() -> None:
    text = render_markdown(_sample_result())
    assert "structured_output" in text
    assert "qwen2.5:14b" in text


def test_limitations_rendered() -> None:
    text = render_markdown(_sample_result())
    assert "not a universal LLM ranking" in text
    assert "observed on this benchmark host" in text


def test_no_raw_outputs_rendered() -> None:
    text = render_markdown(_sample_result())
    assert "tool_calls" not in text
    assert "arguments_json" not in text


def test_secret_needles_rejected() -> None:
    assert contains_secret_patterns("Authorization: Bearer secret")
    assert contains_secret_patterns("password=abc")
    assert not contains_secret_patterns("safe benchmark summary")


def test_byte_stable_rendering() -> None:
    result = _sample_result()
    assert render_markdown(result) == render_markdown(result)
    assert serialize_result_json(result) == serialize_result_json(result)


def test_aggregation_percentiles() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert median_deterministic(values) == 30.0
    assert percentile_nearest_rank(values, 0.95) == 50.0
    stats = latency_stats(values)
    assert stats.p95 == 50.0


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

    conditional = _qualification_status(
        config=cfg,
        schema_probe=SchemaProbeStatus.PASS,
        warmup_status=WarmupStatus.PASS,
        samples=30,
        semantic_success_rate=0.95,
        invalid_draft_count=0,
        provider_failure_count=0,
        unsafe_state_change_count=0,
    )
    assert conditional == ProtocolStatus.CONDITIONALLY_QUALIFIED

    not_qualified = _qualification_status(
        config=cfg,
        schema_probe=SchemaProbeStatus.PASS,
        warmup_status=WarmupStatus.PASS,
        samples=30,
        semantic_success_rate=0.5,
        invalid_draft_count=0,
        provider_failure_count=0,
        unsafe_state_change_count=0,
    )
    assert not_qualified == ProtocolStatus.NOT_QUALIFIED


def test_serialize_schema_versions() -> None:
    payload = _sample_result().to_json_dict()
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["benchmark_id"] == BENCHMARK_ID
    assert payload["corpus_version"] == CORPUS_VERSION
