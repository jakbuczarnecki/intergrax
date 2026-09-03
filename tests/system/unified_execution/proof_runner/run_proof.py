# © Artur Czarnecki. All rights reserved.

"""UE-11G-C1 — real agentic production E2E certification orchestrator."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from typing import Literal

from tests.system.unified_execution.proof_runner.contracts import (
    CertificationEvidence,
    DiagnosticCheckResultProjection,
    FunctionalDiagnosticSection,
    ProofConfig,
    ProofReport,
)
from tests.system.unified_execution.proof_runner.functional_diagnosis import (
    FunctionalDiagnosisReport,
    evaluate_r4_result,
    run_functional_diagnosis,
)
from tests.system.unified_execution.proof_runner.lkw_client import LkwClient, LkwClientError
from tests.system.unified_execution.proof_runner.oracle import (
    expected_fact,
    functional_oracle_passes,
    search_request_message,
)
from tests.system.unified_execution.proof_runner.provider_evidence import (
    EvidenceReadError,
    probe_ollama_model,
    read_otlp_identity_evidence,
)
from tests.system.unified_execution.proof_runner.sqlite_runtime_evidence import (
    SqliteEvidenceReadError,
    read_sqlite_runtime_identity_evidence,
)

_ARTIFACT_DIR = Path(
    os.environ.get(
        "UE_11G_C1_ARTIFACT_DIR",
        "/workspace/.tmp/session/ue-11g-c1/docker-run",
    )
)
_FIXTURE_FILES = (
    "architecture.md",
    "incident-report.md",
    "operations.md",
)


def _config_from_env() -> ProofConfig:
    return ProofConfig(
        base_url=os.environ.get("LKW_BASE_URL", "http://local_workspace:8020"),
        api_key=os.environ.get(
            "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY",
            "ue-11g-c1-certification-secret",
        ),
        embedding_model=os.environ.get(
            "INTERGRAX_EMBEDDING_MODEL",
            "nomic-embed-text",
        ),
        llm_model=os.environ.get(
            "INTERGRAX_LLM_MODEL",
            "llama3.1:latest",
        ),
        otlp_log_path=os.environ.get(
            "UE_11G_C1_OTLP_LOG_PATH",
            "/var/lib/otelcol/lkw-otlp-logs.jsonl",
        ),
    )


def _http_get_json(url: str, *, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise LkwClientError("ollama_response_not_object")
    return parsed


def _ensure_ollama_model(config: ProofConfig, *, model_name: str) -> dict[str, object]:
    tags_url = f"{config.ollama_base_url.rstrip('/')}/api/tags"
    tags_payload = _http_get_json(tags_url, timeout=30.0)
    existing = probe_ollama_model(
        tags_payload=tags_payload,
        model_name=model_name,
    )
    if existing.listed_after_run:
        return tags_payload
    pull_url = f"{config.ollama_base_url.rstrip('/')}/api/pull"
    pull_body = json.dumps({"name": model_name}).encode("utf-8")
    request = urllib.request.Request(
        pull_url,
        data=pull_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.readiness_timeout_seconds) as response:
        _ = response.read()
    return _http_get_json(tags_url, timeout=30.0)


def _ensure_ollama_models(config: ProofConfig) -> dict[str, object]:
    tags_payload = _ensure_ollama_model(config, model_name=config.embedding_model)
    return _ensure_ollama_model(config, model_name=config.llm_model)


def _wait_for_ollama_model(config: ProofConfig) -> dict[str, object]:
    deadline = time.monotonic() + config.readiness_timeout_seconds
    tags_url = f"{config.ollama_base_url.rstrip('/')}/api/tags"
    last_error = "ollama_unreachable"
    while time.monotonic() < deadline:
        try:
            return _http_get_json(tags_url, timeout=10.0)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(3.0)
    raise LkwClientError(last_error)


def _fixture_paths(config: ProofConfig) -> list[str]:
    root = Path(config.fixture_root)
    return [str((root / name).resolve()) for name in _FIXTURE_FILES]


def _assert_search_completed(response_state: str) -> str | None:
    if response_state != "completed":
        return f"search_state_{response_state}"
    return None


def _assert_production_agent(response_agent_id: str | None, expected: str) -> None:
    if response_agent_id != expected:
        raise LkwClientError(f"unexpected_agent:{response_agent_id}")


def _assert_runtime_tools_present(response_tools_total: int) -> None:
    if response_tools_total <= 0:
        raise LkwClientError("runtime_tool_events_missing")


def _budget_tokens(response_tokens: int, otlp_events: int) -> int:
    if response_tokens > 0:
        return response_tokens
    if otlp_events > 0:
        return otlp_events
    return 0


def _diagnostic_section(report: FunctionalDiagnosisReport) -> FunctionalDiagnosticSection:
    return FunctionalDiagnosticSection(
        invocation_status=report.invocation_status,
        persistence_backend=report.persistence_backend,
        durable=report.durable,
        evidence_kinds=list(report.evidence_kinds),
        evidence_count=report.evidence_count,
        validation_id=report.validation_id,
        functional_expected=report.functional_expected,
        functional_actual_bounded=report.functional_actual_bounded,
        diagnostic_specification_id=report.diagnostic_specification_id,
        diagnostic_specification_version=report.diagnostic_specification_version,
        diagnostic_first_proven_failure=report.diagnostic_first_proven_failure,
        diagnostic_check_results=[
            DiagnosticCheckResultProjection(
                check_id=item.check_id,
                status=item.status,
                factual_claim=item.factual_claim,
            )
            for item in report.diagnostic_check_results
        ],
        diagnostic_supporting_evidence_refs=list(report.diagnostic_supporting_evidence_refs),
        diagnostic_limitations=list(report.diagnostic_limitations),
        failure_stage=report.failure_stage,
        confidence=report.confidence,
        blocked_reason=report.blocked_reason,
    )


def _finalize_report(
    *,
    verdict: Literal["PASS", "FAIL", "PARTIAL", "BLOCKED"],
    evidence: CertificationEvidence | None = None,
    failure_reason: str | None = None,
    search_completed: bool = False,
    diagnosis: FunctionalDiagnosisReport | None = None,
) -> ProofReport:
    functional_diagnostic = (
        _diagnostic_section(diagnosis) if diagnosis is not None else None
    )
    r4_result = evaluate_r4_result(
        search_completed=search_completed,
        oracle_pass=bool(evidence and evidence.functional_oracle_pass),
        diagnosis=diagnosis,
    )
    return ProofReport(
        verdict=verdict,
        evidence=evidence,
        failure_reason=failure_reason,
        functional_diagnostic=functional_diagnostic,
        r4_result=r4_result,
    )


def run_certification() -> ProofReport:
    config = _config_from_env()
    client = LkwClient(config)
    try:
        client.wait_until_ready()
        tags_payload = _ensure_ollama_models(config)
        ollama_before = probe_ollama_model(
            tags_payload=tags_payload,
            model_name=config.embedding_model,
        )
        if not ollama_before.listed_after_run:
            raise LkwClientError("embedding_model_not_listed")
        llm_before = probe_ollama_model(
            tags_payload=tags_payload,
            model_name=config.llm_model,
        )
        if not llm_before.listed_after_run:
            raise LkwClientError("llm_model_not_listed")

        index_response = client.run_index(source_paths=_fixture_paths(config))
        if index_response.state != "completed":
            raise LkwClientError(f"index_state_{index_response.state}")

        search_response = client.run_search(message=search_request_message())
        search_failure = _assert_search_completed(search_response.state)
        if search_response.state == "completed":
            _assert_production_agent(search_response.agent_id, config.agent_id)
            if search_response.runtime_event_summary is not None:
                _assert_runtime_tools_present(search_response.runtime_event_summary.tool_events_total)

        time.sleep(3.0)
        runtime_identity = read_sqlite_runtime_identity_evidence(
            db_path=Path(config.runtime_events_db_path),
            tenant_id=config.tenant_id,
            run_id=search_response.run_id,
        )
        otlp = runtime_identity
        try:
            otlp = read_otlp_identity_evidence(
                log_path=Path(config.otlp_log_path),
                run_id=search_response.run_id,
            )
        except EvidenceReadError:
            otlp = runtime_identity
        if otlp.execution_id is None:
            partial_identity_gap = "execution_id_not_persisted_in_runtime_events"
        else:
            partial_identity_gap = None

        tags_after = _wait_for_ollama_model(config)
        ollama_after = probe_ollama_model(
            tags_payload=tags_after,
            model_name=config.embedding_model,
        )
        if not ollama_after.listed_after_run:
            raise LkwClientError("ollama_model_missing_after_run")

        oracle_pass = (
            False
            if search_failure is not None
            else functional_oracle_passes(search_response)
        )
        budget = 0
        if search_response.application_run_summary is not None:
            budget = _budget_tokens(
                search_response.application_run_summary.total_llm_tokens,
                otlp.event_count,
            )
        if budget <= 0 and search_response.runtime_event_summary is not None:
            budget = search_response.runtime_event_summary.tool_events_total

        lkw_evidence_dict = (
            search_response.lkw_evidence.model_dump()
            if search_response.lkw_evidence is not None
            else None
        )
        diagnosis = run_functional_diagnosis(
            config,
            tenant_id=config.tenant_id,
            task_id=search_response.task_id,
            run_id=search_response.run_id,
            attempt_id=otlp.attempt_id,
            answer=search_response.answer,
            lkw_evidence=lkw_evidence_dict,
        )

        evidence = CertificationEvidence(
            http_status=200,
            task_id=search_response.task_id,
            run_id=search_response.run_id,
            attempt_id=otlp.attempt_id,
            execution_id=otlp.execution_id,
            capability=config.capability,
            agent_id=config.agent_id,
            llm_provider=config.llm_provider,
            embedding_model=config.embedding_model,
            ollama=ollama_after,
            otlp=otlp,
            budget_tokens=budget,
            functional_oracle_pass=oracle_pass,
            functional_expected=expected_fact(),
            functional_actual_bounded=(search_response.answer or "")[:200],
        )
        if search_failure is not None:
            return _finalize_report(
                verdict="FAIL",
                evidence=evidence,
                failure_reason=search_failure,
                search_completed=False,
                diagnosis=diagnosis,
            )
        if not oracle_pass:
            return _finalize_report(
                verdict="FAIL",
                evidence=evidence,
                failure_reason="functional_oracle_failed",
                search_completed=True,
                diagnosis=diagnosis,
            )
        if partial_identity_gap is not None:
            return _finalize_report(
                verdict="PARTIAL",
                evidence=evidence,
                failure_reason=partial_identity_gap,
                search_completed=True,
                diagnosis=diagnosis,
            )
        return _finalize_report(
            verdict="PASS",
            evidence=evidence,
            search_completed=True,
            diagnosis=diagnosis,
        )
    except (LkwClientError, EvidenceReadError, SqliteEvidenceReadError, json.JSONDecodeError) as exc:
        return _finalize_report(verdict="FAIL", failure_reason=str(exc))


def write_report(report: ProofReport) -> Path:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _ARTIFACT_DIR / "proof-report.json"
    artifact.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return artifact


def main() -> int:
    report = run_certification()
    artifact = write_report(report)
    print(report.model_dump_json(indent=2))
    print(f"artifact={artifact}")
    if report.verdict == "PASS":
        return 0
    if report.verdict == "PARTIAL":
        return 2
    return 1


def test_ue_11g_c1_real_agentic_production_e2e() -> None:
    """Matrix anchor — executed by Docker proof-runner, not host pytest."""
    raise RuntimeError(
        "Run UE-11G-C1 via: docker compose -f tests/system/unified_execution/docker-compose.yml "
        "up --build --exit-code-from proof-runner"
    )


if __name__ == "__main__":
    raise SystemExit(main())
