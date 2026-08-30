# © Artur Czarnecki. All rights reserved.

"""Shared helpers for DIAG-FINAL external OpenTelemetry E2E proof."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    OtlpExportOperatorConfig,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "diag_final_otel"
_COMPOSE_PROJECT = "diag-final-otel-proof"
_COLLECTOR_OUTPUT_DIR = _FIXTURES_DIR / "collector-output"
_COLLECTOR_OUTPUT_FILE = _COLLECTOR_OUTPUT_DIR / "diag-final-received.jsonl"
_DEFAULT_OTLP_ENDPOINT = "http://127.0.0.1:14318/v1/logs"
_ROUTE_PREFIX = "/v1/governed_contractor"


@dataclass(frozen=True, slots=True)
class DiagFinalHostComposition:
    """Resolved PRODUCT host used by the external OTLP proof."""

    app: FastAPI
    client: TestClient
    runtime: HarnessHostRuntime
    document_store: InMemoryDocumentStore
    observability_export: ObservabilityExportOperatorConfig


@dataclass(frozen=True, slots=True)
class DiagFinalCollectorStack:
    """Running local OpenTelemetry Collector stack for the proof."""

    endpoint: str
    output_host_path: Path
    compose_file: Path


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def docker_daemon_available() -> bool:
    if not docker_cli_available():
        return False
    completed = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0


def _run_compose(*args: str, timeout: float | None = 120) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "compose",
        "-p",
        _COMPOSE_PROJECT,
        "-f",
        str(_FIXTURES_DIR / "docker-compose.yml"),
        *args,
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _collector_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.post(
                _DEFAULT_OTLP_ENDPOINT,
                json={"resourceLogs": []},
                headers={"Content-Type": "application/json"},
            )
        return response.status_code < 500
    except Exception:
        return False


def _collector_container_id() -> str:
    completed = _run_compose("ps", "-q", "otel-collector", timeout=30)
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(
            "diag-final otel collector container not running:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
    return completed.stdout.strip().splitlines()[0]


def start_collector_stack(tmp_path: Path) -> DiagFinalCollectorStack:
    _COLLECTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_host_path = _COLLECTOR_OUTPUT_FILE
    if output_host_path.exists():
        output_host_path.unlink()

    up = _run_compose("up", "-d", timeout=180)
    if up.returncode != 0:
        raise RuntimeError(
            "failed to start diag-final otel collector:\n"
            f"stdout={up.stdout}\nstderr={up.stderr}",
        )

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if _collector_reachable():
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("diag-final otel collector endpoint did not become reachable")

    _collector_container_id()
    if not output_host_path.exists():
        output_host_path.write_text("", encoding="utf-8")

    return DiagFinalCollectorStack(
        endpoint=_DEFAULT_OTLP_ENDPOINT,
        output_host_path=output_host_path,
        compose_file=_FIXTURES_DIR / "docker-compose.yml",
    )


def stop_collector_stack() -> None:
    _run_compose("down", "-v", timeout=120)


def refresh_collector_output(stack: DiagFinalCollectorStack) -> str:
    if not stack.output_host_path.exists():
        return ""
    return stack.output_host_path.read_text(encoding="utf-8")


def stop_collector_process_only() -> None:
    completed = _run_compose("stop", "otel-collector", timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to stop diag-final otel collector:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


def wait_for_collector_run_id(
    stack: DiagFinalCollectorStack,
    run_id: str,
    *,
    timeout_seconds: float = 60.0,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = refresh_collector_output(stack)
        if run_id in text:
            return text
        time.sleep(0.5)
    raise AssertionError(
        f"collector did not receive export for run_id={run_id!r} within {timeout_seconds}s",
    )


def build_observability_export_config(endpoint: str) -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=False,
        backend_id="otlp",
        otlp=OtlpExportOperatorConfig(
            endpoint=endpoint,
            service_name="diag-final-e2e",
            service_version="proof",
            environment="integration",
            timeout_seconds=0.5,
        ),
    )


def attach_retry_violation_injector(
    runtime: HarnessHostRuntime,
    *,
    tenant_id: str,
) -> None:
    runtime_store = runtime.observability.runtime_event_store
    if runtime_store is None:
        raise ValueError("diag-final proof requires RuntimeEvent persistence")

    def _handler(event: RuntimeEvent) -> None:
        if event.event_type is not RuntimeEventType.TASK_COMPLETED:
            return
        if event.tenant_id != tenant_id:
            return
        runtime_store.append(
            sample_runtime_event(
                tenant_id=event.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                attempt_id=event.attempt_id,
            ).model_copy(
                update={
                    "event_type": RuntimeEventType.RETRY_SCHEDULED,
                    "phase": ExecutionPhase.RETRY_HANDLING,
                },
            ),
            tenant_id=event.tenant_id,
        )

    runtime.nexus_loop.event_bus.subscribe(
        _handler,
        event_types={RuntimeEventType.TASK_COMPLETED},
        priority=10,
        subscription_id="diag_final.retry_violation_injector",
    )


def build_diag_final_product_host(
    *,
    tmp_path: Path,
    document_store: InMemoryDocumentStore,
    observability_export: ObservabilityExportOperatorConfig,
    tenant_id: str,
    inject_violation: bool = True,
) -> DiagFinalHostComposition:
    app = create_governed_contractor_backend_app(
        registry_projection=build_governed_contractor_test_registry_projection(),
        settings=GovernedContractorBackendSettings.from_env(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "runtime_events.db",
        checkpoints_db_path=tmp_path / "checkpoints.db",
        document_store=document_store,
        observability_export=observability_export,
    )
    runtime = app.state.harness_runtime
    if inject_violation:
        attach_retry_violation_injector(runtime, tenant_id=tenant_id)
    return DiagFinalHostComposition(
        app=app,
        client=TestClient(app),
        runtime=runtime,
        document_store=document_store,
        observability_export=observability_export,
    )


def execute_host_run(
    composition: DiagFinalHostComposition,
    *,
    tenant_id: str,
    message: str,
) -> dict[str, object]:
    response = composition.client.post(
        f"{_ROUTE_PREFIX}/run",
        json={
            "tenant_id": tenant_id,
            "user_id": "diag-final-proof",
            "message": message,
            "capability": "external_contractor.adapt",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("state") == "completed", payload
    assert payload.get("run_id")
    return payload


def build_read_service(composition: DiagFinalHostComposition):
    deps = resolve_host_diagnostic_read_dependencies(composition.runtime)
    return build_diagnostic_read_service(deps)


def assert_runtime_event_truth(
    composition: DiagFinalHostComposition,
    *,
    tenant_id: str,
    run_id: str,
    task_id: str,
) -> RuntimeEvent:
    store = composition.runtime.observability.runtime_event_store
    assert store is not None
    events = store.list_for_run(run_id, tenant_id=tenant_id)
    terminal = [
        event
        for event in events
        if event.event_type is RuntimeEventType.TASK_COMPLETED
    ]
    assert terminal, "expected terminal TASK_COMPLETED RuntimeEvent"
    assert terminal[0].tenant_id == tenant_id
    assert terminal[0].task_id == task_id
    assert terminal[0].run_id == run_id
    return terminal[0]


def assert_problem_truth(
    composition: DiagFinalHostComposition,
    *,
    tenant_id: str,
    run_id: str,
) -> str:
    read_service = build_read_service(composition)
    listed = read_service.list_problems(tenant_id=tenant_id)
    assert listed.total_count >= 1, "expected central diagnostic Problem"
    for problem in listed.problems:
        detail = read_service.get_problem(
            tenant_id=tenant_id,
            problem_id=problem.problem_id,
        )
        assert detail is not None
        occurrence_run_ids: set[str] = set()
        for occurrence in detail.occurrences:
            execution = occurrence.subject_ref.execution()
            if execution is not None:
                occurrence_run_ids.add(str(execution.run_id))
        if run_id in occurrence_run_ids:
            return str(problem.problem_id)
    raise AssertionError(f"no central Problem occurrence for run_id={run_id!r}")


def write_proof_artifact(
    artifact_dir: Path,
    *,
    run_id: str,
    task_id: str,
    problem_id: str,
    terminal_event_type: str,
    collector_received: bool,
    collector_excerpt: str,
    collector_available: bool,
    restart_verified: bool,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "diag-final-e2e-proof.json"
    artifact_path.write_text(
        json.dumps(
            {
                "proof_id": "DIAG-FINAL-E2E",
                "run_id": run_id,
                "task_id": task_id,
                "problem_id": problem_id,
                "terminal_runtime_event_type": terminal_event_type,
                "collector_received_export": collector_received,
                "collector_available_after_vendor_failure": collector_available,
                "restart_persistence_verified": restart_verified,
                "collector_excerpt": collector_excerpt[:2000],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_path


@pytest.fixture(scope="module")
def diag_final_collector_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DiagFinalCollectorStack]:
    if not docker_daemon_available():
        pytest.skip("docker daemon unavailable for DIAG-FINAL external OTLP proof")
    tmp_path = tmp_path_factory.mktemp("diag-final-collector")
    stack = start_collector_stack(tmp_path)
    try:
        yield stack
    finally:
        stop_collector_stack()
