# © Artur Czarnecki. All rights reserved.

"""Shared helpers for DIAG-FINAL external OpenTelemetry E2E proof."""

from __future__ import annotations

import json
import os
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
from intergrax.applications._shared.plugin_bootstrap import bootstrap_application_plugins
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.export_boundary import FORBIDDEN_EXPORT_CONTENT_FIELDS
from intergrax.runtime.observability.export_health import (
    ObservabilityExporterHealthRegistry,
    ObservabilityExporterHealthSnapshot,
    ObservabilityExporterHealthStatus,
)
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    OtlpExportOperatorConfig,
    build_observability_export_runtime_plugin,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "diag_final_otel"
_COMPOSE_PROJECT = "diag-final-otel-proof"
_COLLECTOR_OUTPUT_DIR = _FIXTURES_DIR / "collector-output"
_COLLECTOR_OUTPUT_FILE = _COLLECTOR_OUTPUT_DIR / "diag-final-received.jsonl"
_DEFAULT_OTLP_ENDPOINT = "http://127.0.0.1:14318/v1/logs"
_ROUTE_PREFIX = "/v1/governed_contractor"
_FORBIDDEN_PRIVACY_SUBSTRINGS = (
    "secret prompt",
    "raw body",
    "diag-final collector available",
    "diag-final collector unavailable",
    "diag-final collector recovered",
    "diag-final restart persistence",
)


def external_otlp_proof_required() -> bool:
    raw = os.environ.get("INTERGRAX_EXTERNAL_OTLP_PROOF_REQUIRED", "").strip().lower()
    return raw in {"1", "true", "yes"}


def require_docker_for_external_otlp_proof() -> None:
    if docker_daemon_available():
        return
    if external_otlp_proof_required():
        pytest.fail("docker daemon unavailable for required external OTLP proof")
    pytest.skip("docker daemon unavailable for DIAG-FINAL external OTLP proof")


@dataclass(frozen=True, slots=True)
class DiagFinalHostComposition:
    """Resolved PRODUCT host used by the external OTLP proof."""

    app: FastAPI
    client: TestClient
    runtime: HarnessHostRuntime
    document_store: InMemoryDocumentStore
    observability_export: ObservabilityExportOperatorConfig
    health_registry: ObservabilityExporterHealthRegistry
    exporter_id: str


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


def parse_collector_otlp_records(collector_text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in collector_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _otlp_log_record_attribute_maps(record: dict[str, object]) -> list[dict[str, object]]:
    maps: list[dict[str, object]] = []
    resource_logs = record.get("resourceLogs")
    if not isinstance(resource_logs, list):
        return maps
    for resource_log in resource_logs:
        if not isinstance(resource_log, dict):
            continue
        scope_logs = resource_log.get("scopeLogs")
        if not isinstance(scope_logs, list):
            continue
        for scope_log in scope_logs:
            if not isinstance(scope_log, dict):
                continue
            log_records = scope_log.get("logRecords")
            if not isinstance(log_records, list):
                continue
            for log_record in log_records:
                if not isinstance(log_record, dict):
                    continue
                attributes = log_record.get("attributes")
                if not isinstance(attributes, list):
                    continue
                mapped: dict[str, object] = {}
                for attribute in attributes:
                    if not isinstance(attribute, dict):
                        continue
                    key = attribute.get("key")
                    value = attribute.get("value")
                    if not isinstance(key, str) or not isinstance(value, dict):
                        continue
                    if "stringValue" in value:
                        mapped[key] = value["stringValue"]
                    elif "intValue" in value:
                        mapped[key] = value["intValue"]
                    elif "boolValue" in value:
                        mapped[key] = value["boolValue"]
                maps.append(mapped)
    return maps


def _otlp_attribute_map(record: dict[str, object]) -> dict[str, object]:
    maps = _otlp_log_record_attribute_maps(record)
    if not maps:
        return {}
    return maps[-1]


def collector_records_for_run(
    collector_text: str,
    *,
    run_id: str,
) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for record in parse_collector_otlp_records(collector_text):
        attrs = _otlp_attribute_map(record)
        if attrs.get("intergrax.run_id") == run_id or run_id in json.dumps(record, ensure_ascii=False):
            matched.append(record)
    return matched


def assert_collector_identity_matches_runtime_event(
    collector_text: str,
    *,
    terminal_event: RuntimeEvent,
) -> dict[str, object]:
    records = collector_records_for_run(collector_text, run_id=str(terminal_event.run_id))
    assert records, f"collector did not receive export for run_id={terminal_event.run_id!r}"

    event_id = str(terminal_event.event_id)
    attribute_maps = [
        attrs
        for record in records
        for attrs in _otlp_log_record_attribute_maps(record)
        if attrs.get("intergrax.run_id") == str(terminal_event.run_id)
    ]
    assert attribute_maps, "collector export missing intergrax.run_id attribute"

    matching = [attrs for attrs in attribute_maps if attrs.get("intergrax.event_id") == event_id]
    assert matching, f"collector export missing event_id={event_id!r}"

    terminal_matches = [
        attrs
        for attrs in matching
        if attrs.get("intergrax.event_type") == terminal_event.event_type.value
    ]
    assert len(terminal_matches) == 1, "expected exactly one canonical HOS export per RuntimeEvent"
    chosen = terminal_matches[0]

    assert chosen.get("intergrax.run_id") == str(terminal_event.run_id)
    assert chosen.get("intergrax.attempt_id") == str(terminal_event.attempt_id)
    assert chosen.get("intergrax.execution_id") == str(terminal_event.execution_id)
    assert chosen.get("intergrax.task_id") == str(terminal_event.task_id)
    assert chosen.get("intergrax.record_kind") == "runtime_event"
    assert chosen.get("intergrax.event_type")
    return chosen


def assert_collector_hos_privacy(
    collector_text: str,
    *,
    extra_forbidden_substrings: tuple[str, ...] = (),
) -> None:
    serialized_values: list[str] = []
    for record in parse_collector_otlp_records(collector_text):
        resource_logs = record.get("resourceLogs")
        if not isinstance(resource_logs, list):
            continue
        for resource_log in resource_logs:
            if not isinstance(resource_log, dict):
                continue
            scope_logs = resource_log.get("scopeLogs")
            if not isinstance(scope_logs, list):
                continue
            for scope_log in scope_logs:
                if not isinstance(scope_log, dict):
                    continue
                log_records = scope_log.get("logRecords")
                if not isinstance(log_records, list):
                    continue
                for log_record in log_records:
                    if not isinstance(log_record, dict):
                        continue
                    body = log_record.get("body")
                    if isinstance(body, dict):
                        body_value = body.get("stringValue")
                        if isinstance(body_value, str):
                            serialized_values.append(body_value)
                    attributes = log_record.get("attributes")
                    if isinstance(attributes, list):
                        for attribute in attributes:
                            if not isinstance(attribute, dict):
                                continue
                            value = attribute.get("value")
                            if isinstance(value, dict) and "stringValue" in value:
                                string_value = value["stringValue"]
                                if isinstance(string_value, str):
                                    serialized_values.append(string_value)

    combined = "\n".join(serialized_values)
    forbidden_attribute_keys = {
        f"intergrax.{field_name}" for field_name in FORBIDDEN_EXPORT_CONTENT_FIELDS
    }
    for record in parse_collector_otlp_records(collector_text):
        for attrs in _otlp_log_record_attribute_maps(record):
            for key in attrs:
                assert key not in forbidden_attribute_keys, f"forbidden export attribute key: {key}"
    for substring in (*_FORBIDDEN_PRIVACY_SUBSTRINGS, *extra_forbidden_substrings):
        assert substring not in combined, f"raw content leaked to collector: {substring!r}"


def stop_collector_process_only() -> None:
    completed = _run_compose("stop", "otel-collector", timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to stop diag-final otel collector:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


def start_collector_process_only() -> None:
    completed = _run_compose("start", "otel-collector", timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to start diag-final otel collector:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if _collector_reachable():
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("diag-final otel collector endpoint did not become reachable after start")

    _collector_container_id()


def assert_collector_unreachable() -> None:
    assert not _collector_reachable(), "collector OTLP endpoint must be unreachable during outage"


def wait_for_collector_event_id(
    stack: DiagFinalCollectorStack,
    event_id: str,
    *,
    timeout_seconds: float = 60.0,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = refresh_collector_output(stack)
        for record in parse_collector_otlp_records(text):
            for attrs in _otlp_log_record_attribute_maps(record):
                if attrs.get("intergrax.event_id") == event_id:
                    return text
        if event_id in text:
            return text
        time.sleep(0.5)
    raise AssertionError(
        f"collector did not receive export for event_id={event_id!r} within {timeout_seconds}s",
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
    health_registry = ObservabilityExporterHealthRegistry()
    exporter_id = observability_export.backend_id
    app = create_governed_contractor_backend_app(
        registry_projection=build_governed_contractor_test_registry_projection(),
        settings=GovernedContractorBackendSettings.from_env(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "runtime_events.db",
        checkpoints_db_path=tmp_path / "checkpoints.db",
        document_store=document_store,
        observability_export=None,
    )
    runtime = app.state.harness_runtime
    if observability_export.enabled:
        export_plugin = build_observability_export_runtime_plugin(
            observability_export,
            health_registry=health_registry,
        )
        if export_plugin is not None:
            bootstrap_application_plugins(
                [export_plugin],
                nexus_loop=runtime.nexus_loop,
            )
    if inject_violation:
        attach_retry_violation_injector(runtime, tenant_id=tenant_id)
    return DiagFinalHostComposition(
        app=app,
        client=TestClient(app),
        runtime=runtime,
        document_store=document_store,
        observability_export=observability_export,
        health_registry=health_registry,
        exporter_id=exporter_id,
    )


def wait_for_exporter_health_snapshot(
    composition: DiagFinalHostComposition,
    *,
    expected_status: ObservabilityExporterHealthStatus,
    timeout_seconds: float = 30.0,
    min_consecutive_failures: int | None = None,
    min_recovery_count: int | None = None,
) -> ObservabilityExporterHealthSnapshot:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = composition.health_registry.get(composition.exporter_id)
        if snapshot is None:
            time.sleep(0.1)
            continue
        if snapshot.status is not expected_status:
            time.sleep(0.1)
            continue
        if (
            min_consecutive_failures is not None
            and snapshot.consecutive_failures < min_consecutive_failures
        ):
            time.sleep(0.1)
            continue
        if min_recovery_count is not None and snapshot.recovery_count < min_recovery_count:
            time.sleep(0.1)
            continue
        return snapshot
    raise AssertionError(
        "exporter health snapshot did not reach "
        f"status={expected_status.value!r} for exporter_id={composition.exporter_id!r} "
        f"within {timeout_seconds}s",
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
    attempt_id: str,
    execution_id: str,
    event_id: str,
    problem_id: str,
    terminal_event_type: str,
    collector_received: bool,
    collector_excerpt: str,
    collector_available: bool,
    restart_verified: bool,
    identity_verified: bool,
    privacy_verified: bool,
    real_collector_recovery_verified: bool = False,
    health_degraded_verified: bool = False,
    health_recovered_verified: bool = False,
    outage_event_replay_absent: bool = False,
    recovery_run_id: str | None = None,
    recovery_event_id: str | None = None,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "diag-final-e2e-proof.json"
    artifact_path.write_text(
        json.dumps(
            {
                "proof_id": "HARDEN-3F-EXTERNAL-OTLP",
                "run_id": run_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "execution_id": execution_id,
                "event_id": event_id,
                "problem_id": problem_id,
                "terminal_runtime_event_type": terminal_event_type,
                "collector_received_export": collector_received,
                "collector_available_after_vendor_failure": collector_available,
                "restart_persistence_verified": restart_verified,
                "identity_correlation_verified": identity_verified,
                "hos_privacy_verified": privacy_verified,
                "real_collector_recovery_verified": real_collector_recovery_verified,
                "health_degraded_verified": health_degraded_verified,
                "health_recovered_verified": health_recovered_verified,
                "outage_event_replay_absent": outage_event_replay_absent,
                "recovery_run_id": recovery_run_id,
                "recovery_event_id": recovery_event_id,
                "verification_summary": {
                    "real_docker_collector_success_verified": collector_received,
                    "real_docker_collector_outage_verified": health_degraded_verified,
                    "real_docker_collector_restart_verified": real_collector_recovery_verified,
                    "real_hos_export_after_restart_verified": real_collector_recovery_verified,
                    "process_local_exporter_health_recovery_verified": health_recovered_verified,
                    "no_replay_verified": outage_event_replay_absent,
                },
                "collector_excerpt": collector_excerpt[:2000],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_path


@pytest.fixture(scope="module")
def diag_final_collector_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DiagFinalCollectorStack]:
    require_docker_for_external_otlp_proof()
    tmp_path = tmp_path_factory.mktemp("diag-final-collector")
    stack = start_collector_stack(tmp_path)
    try:
        yield stack
    finally:
        stop_collector_stack()
