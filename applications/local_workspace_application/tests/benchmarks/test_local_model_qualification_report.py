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
    ModelProvisioningStatus,
    ObservedExecutionMode,
    OllamaEnvironment,
    ProtocolResult,
    ProtocolStatus,
    ProvisionedModel,
    ProvisioningResult,
    QualificationSummary,
    RESULT_SCHEMA_VERSION,
    SchemaProbeStatus,
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
    latency_stats,
    median_deterministic,
    percentile_nearest_rank,
)
from local_workspace_application.benchmarks.local_model_qualification.config import QualificationConfig


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
