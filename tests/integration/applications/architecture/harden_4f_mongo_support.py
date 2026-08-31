# © Artur Czarnecki. All rights reserved.

"""HARDEN-4F — shared Mongo docker lifecycle + Mongo-backed product host helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest
from pymongo.errors import PyMongoError
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
from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemId
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from tests.integration.runtime.diag_final_otel_support import attach_retry_violation_injector
from tests.unit.runtime.diagnostics.problem_persistence_test_support import document_store_problem_persistence_for_tests

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE_FILE = _REPO_ROOT / "infra" / "docker" / "mongodb" / "docker-compose.yml"
_COMPOSE_PROJECT = "harden-4f-mongo-proof"
_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_harden_4f"
_COLLECTION_PREFIX = "harden_4f_"
_DOCUMENT_PARTITION_PREFIX = "intergrax.diagnostic_problem.v1"
_PROBE_PARTITION = "harden-4f-probe"
_PROBE_ROW_KEY = "reachability"
_REACHABILITY_TIMEOUT_SECONDS = 30.0
_REACHABILITY_POLL_SECONDS = 0.5
_MONGO_SELECTION_TIMEOUT_MS = 2000
_ROUTE_PREFIX = "/v1/governed_contractor"

_stopped_container_id: str | None = None


class ProductHostExecutionComposition(Protocol):
    """Minimal PRODUCT host surface for HTTP execution + runtime event truth checks."""

    client: TestClient
    runtime: HarnessHostRuntime


@dataclass(frozen=True, slots=True)
class Harden4FMongoHostComposition:
    """PRODUCT host with Mongo-backed DocumentStore for Problem persistence."""

    app: FastAPI
    client: TestClient
    runtime: HarnessHostRuntime
    document_store: ConditionalDocumentStore
    collection_name: str


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


def require_docker_for_harden_4f_proof() -> None:
    if docker_daemon_available():
        return
    pytest.skip("docker daemon unavailable for HARDEN-4F Mongo external proof")


def _with_bounded_mongo_timeout(uri: str) -> str:
    if "serverSelectionTimeoutMS=" in uri:
        return uri
    separator = "&" if "?" in uri else "?"
    return f"{uri}{separator}serverSelectionTimeoutMS={_MONGO_SELECTION_TIMEOUT_MS}"


def resolve_mongodb_uri() -> str:
    raw = os.environ.get("INTERGRAX_MONGODB_URI", _DEFAULT_URI).strip() or _DEFAULT_URI
    return _with_bounded_mongo_timeout(raw)


def proof_env(*, collection_name: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["INTERGRAX_MONGODB_URI"] = resolve_mongodb_uri()
    env["INTERGRAX_MONGODB_DATABASE"] = _DEFAULT_DATABASE
    env["INTERGRAX_MONGODB_COLLECTION"] = collection_name or (
        f"{_COLLECTION_PREFIX}{uuid.uuid4().hex}"
    )
    pythonpath = [
        str(_REPO_ROOT),
        str(_REPO_ROOT / "agents"),
        str(_REPO_ROOT / "applications"),
    ]
    if existing := env.get("PYTHONPATH", "").strip():
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _run_compose(*args: str, timeout: float | None = 120) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "compose",
        "-p",
        _COMPOSE_PROJECT,
        "-f",
        str(_COMPOSE_FILE),
        *args,
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _resolve_mongo_container_id() -> str | None:
    global _stopped_container_id
    if _stopped_container_id:
        return _stopped_container_id
    compose_ps = _run_compose("ps", "-q", "mongodb", timeout=30)
    if compose_ps.returncode == 0 and compose_ps.stdout.strip():
        return compose_ps.stdout.strip().splitlines()[0]
    listed = subprocess.run(
        ["docker", "ps", "-a", "--filter", "publish=27017", "-q"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if listed.returncode == 0 and listed.stdout.strip():
        return listed.stdout.strip().splitlines()[0]
    by_name = subprocess.run(
        ["docker", "ps", "-a", "-q", "-f", "name=intergrax-mongodb"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if by_name.returncode == 0 and by_name.stdout.strip():
        return by_name.stdout.strip().splitlines()[0]
    return None


def ensure_mongo_running() -> None:
    global _stopped_container_id
    if probe_mongo_document_store():
        _stopped_container_id = None
        return
    if _stopped_container_id is not None:
        start_mongo_container()
        return
    up = _run_compose("up", "-d", timeout=180)
    if up.returncode != 0:
        raise RuntimeError(
            "failed to start HARDEN-4F Mongo stack:\n"
            f"stdout={up.stdout}\nstderr={up.stderr}",
        )
    wait_until_mongo_reachable()


def stop_mongo_container() -> None:
    global _stopped_container_id
    container_id = _resolve_mongo_container_id()
    if container_id is None:
        raise RuntimeError("HARDEN-4F could not resolve Mongo container id for stop")
    completed = subprocess.run(
        ["docker", "stop", container_id],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to stop Mongo container for HARDEN-4F:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
    _stopped_container_id = container_id


def start_mongo_container() -> None:
    global _stopped_container_id
    container_id = _resolve_mongo_container_id()
    if container_id is None:
        raise RuntimeError("HARDEN-4F could not resolve Mongo container id for start")
    completed = subprocess.run(
        ["docker", "start", container_id],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to start Mongo container for HARDEN-4F:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
    wait_until_mongo_reachable()
    _stopped_container_id = None


def _open_platform_document_store() -> ConditionalDocumentStore:
    bootstrap_application_integration_catalog()
    store = create_mongodb_document_store()
    return assert_conditional_document_store(store)


def probe_mongo_document_store() -> bool:
    try:
        store = _open_platform_document_store()
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError):
        return False
    try:
        store.get(_PROBE_PARTITION, _PROBE_ROW_KEY)
        return True
    except (ConnectionError, TimeoutError, OSError, PyMongoError):
        return False
    finally:
        store.close()


def wait_until_mongo_reachable(*, timeout_seconds: float = _REACHABILITY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if probe_mongo_document_store():
            return
        time.sleep(_REACHABILITY_POLL_SECONDS)
    raise AssertionError(f"MongoDB did not become reachable within {timeout_seconds}s")


def wait_until_mongo_unreachable(*, timeout_seconds: float = _REACHABILITY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not probe_mongo_document_store():
            return
        time.sleep(_REACHABILITY_POLL_SECONDS)
    raise AssertionError(f"MongoDB did not become unreachable within {timeout_seconds}s")


def create_proof_document_store() -> ConditionalDocumentStore:
    bootstrap_application_integration_catalog()
    store = create_mongodb_document_store()
    return assert_conditional_document_store(store)


def build_harden_4f_mongo_product_host(
    *,
    tmp_path: Path,
    document_store: ConditionalDocumentStore,
    tenant_id: str,
    inject_violation: bool = True,
) -> Harden4FMongoHostComposition:
    collection_name = os.environ.get("INTERGRAX_MONGODB_COLLECTION", "")
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
    if inject_violation:
        attach_retry_violation_injector(runtime, tenant_id=tenant_id)
    return Harden4FMongoHostComposition(
        app=app,
        client=TestClient(app),
        runtime=runtime,
        document_store=document_store,
        collection_name=collection_name,
    )


def build_read_service(composition: Harden4FMongoHostComposition):
    deps = resolve_host_diagnostic_read_dependencies(composition.runtime)
    return build_diagnostic_read_service(deps)


def execute_mongo_host_run(
    composition: ProductHostExecutionComposition,
    *,
    tenant_id: str,
    message: str,
) -> dict[str, object]:
    response = composition.client.post(
        f"{_ROUTE_PREFIX}/run",
        json={
            "tenant_id": tenant_id,
            "user_id": "harden-4f-proof",
            "message": message,
            "capability": "external_contractor.adapt",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("state") == "completed", payload
    assert payload.get("run_id")
    return payload


def assert_mongo_host_runtime_event_truth(
    composition: ProductHostExecutionComposition,
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


@contextmanager
def mongo_outage_phase() -> Iterator[None]:
    """Stop Mongo for outage assertions; always restart Mongo before exiting."""
    stop_mongo_container()
    try:
        yield
    finally:
        start_mongo_container()


def read_problem_via_fresh_store_persistence(
    *,
    tenant_id: str,
    problem_id: str | ProblemId,
) -> Problem | None:
    """Provider-truth read through a new DocumentStore composition (not host cache)."""
    store = create_proof_document_store()
    try:
        persistence = document_store_problem_persistence_for_tests(store)
        return persistence.get(tenant_id=tenant_id, problem_id=problem_id)
    finally:
        store.close()


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"


def purge_tenant_documents(store: ConditionalDocumentStore, tenant_id: str) -> None:
    partition_key = _document_partition(tenant_id)
    cursor: str | None = None
    while True:
        page = store.query(partition_key, limit=5000, cursor=cursor)
        for document in page.documents:
            store.delete(document.partition_key, document.row_key)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor


def cleanup_proof_tenant(*, tenant_id: str) -> None:
    store = create_proof_document_store()
    try:
        purge_tenant_documents(store, tenant_id)
    finally:
        store.close()


def occurrence_run_ids(problem: Problem) -> set[str]:
    run_ids: set[str] = set()
    for occurrence in problem.occurrences:
        execution = occurrence.subject_ref.execution()
        if execution is not None:
            run_ids.add(str(execution.run_id))
    return run_ids


def assert_host_uses_document_store_problem_persistence(
    composition: Harden4FMongoHostComposition,
) -> DocumentStoreProblemPersistence:
    deps = resolve_host_diagnostic_read_dependencies(composition.runtime)
    persistence = deps.problem_persistence
    if not isinstance(persistence, DocumentStoreProblemPersistence):
        raise AssertionError(
            "expected DocumentStoreProblemPersistence on host diagnostic read dependencies",
        )
    return persistence
