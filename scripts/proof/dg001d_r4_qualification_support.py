# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""Shared helpers for DG-001D R4 real supervisor pre-engine failure qualification."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "applications"
_AGENTS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "agents"
for _bootstrap_entry in (
    str(_APPLICATIONS_BOOTSTRAP_ROOT),
    str(_AGENTS_BOOTSTRAP_ROOT),
    str(_REPO_BOOTSTRAP_ROOT),
):
    if _bootstrap_entry not in sys.path:
        sys.path.insert(0, _bootstrap_entry)

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.hosting.errors import HostedApplicationSupervisorError, HostedApplicationSupervisorFailureReason
from intergrax.hosting.process_bootstrap import HostedProcessBootstrapPhase
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.diagnostic_subject import DiagnosticSubjectKind
from intergrax.runtime.diagnostics.problem_grouping import DeterministicSignalFindingSignature
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemOccurrence
from intergrax.runtime.diagnostics.problem_occurrence_id import problem_occurrence_id_for
from local_workspace_application.host.background_worker_main import (
    activate_local_workspace_reference_production_authority,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from scripts.proof.dg001b_r5_qualification_support import (
    default_mongodb_uri,
    ensure_mongodb_replica_set_for_qualification,
    os_environ_get,
    os_environ_pop,
    os_environ_set,
    os_pathsep_join,
)
from scripts.proof.intergrax_proof_reachability import http_reachable, tcp_reachable
from scripts.proof.dg001d_r4_qualification_contracts import qualification_secret_sentinel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
_DEFAULT_ELASTICSEARCH_INDEX = "intergrax-lkw-observability"
_DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
_SUPERVISOR_CHILD_SCRIPT = _REPO_ROOT / "scripts" / "proof" / "dg001d_r4_supervisor_child.py"
_SECRET_SENTINEL = qualification_secret_sentinel()
_EXPECTED_APPLICATION_ID = LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id
_EXPECTED_PHASE = HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION.value
_EXPECTED_REASON_CODE = HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED.value
_EXPECTED_EXCEPTION_TYPE = HostedApplicationSupervisorError.__name__


class QualificationGate(StrEnum):
    E0_ENVIRONMENT = "E0"
    E1_SUPERVISOR_PROCESS = "E1"
    E2_SUPERVISOR_IDENTITY = "E2"
    E3_PRE_ENGINE_FAILURE = "E3"
    E4_FAILURE_EVENT = "E4"
    E5_OBSERVABILITY_EXPORT = "E5"
    E6_DIAGNOSTIC_PROJECTION = "E6"
    E7_DURABLE_PERSISTENCE = "E7"
    E8_OPERATOR_READ = "E8"
    E9_FAILURE_SAFETY = "E9"


@dataclass(frozen=True, slots=True)
class PrerequisiteStatus:
    mongodb_reachable: bool
    elasticsearch_reachable: bool
    redis_reachable: bool
    mongodb_uri_class: str
    mongodb_database: str
    mongodb_collection_authority: str
    observability_backend: str
    observability_index: str

    @property
    def environment_ready(self) -> bool:
        return self.mongodb_reachable and self.elasticsearch_reachable and self.redis_reachable


@dataclass(frozen=True, slots=True)
class SupervisorProcessEvidence:
    pid: int | None
    exit_code: int
    stdout: str
    stderr: str
    started_at: datetime
    finished_at: datetime
    instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class ObservabilityExportEvidence:
    event_id: str
    application_id: str
    instance_id: str
    lifecycle_state: str
    event_type: str
    occurred_at: str
    qualification_marker: str
    backend_id: str
    index: str


@dataclass(frozen=True, slots=True)
class DiagnosticReadEvidence:
    application_id: str
    instance_id: str
    tenant_id: str
    problem_id: str
    occurrence_id: str
    subject_kind: DiagnosticSubjectKind
    phase: str
    reason_code: str
    exception_type: str
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str
    occurrence_count: int


@dataclass(frozen=True, slots=True)
class QualificationAttemptResult:
    attempt_id: str
    marker: str
    prerequisites: PrerequisiteStatus
    supervisor: SupervisorProcessEvidence | None
    observability: ObservabilityExportEvidence | None
    diagnostic_read: DiagnosticReadEvidence | None
    gate_results: dict[QualificationGate, bool]
    blocked_reason: str | None = None


def repository_root() -> Path:
    return _REPO_ROOT


def qualification_mongodb_database() -> str:
    return "intergrax_dg001d_r4"


def qualification_mongodb_collection() -> str:
    return f"dg001d_r4_{uuid.uuid4().hex[:12]}"


def evaluate_prerequisites(
    *,
    mongodb_uri: str,
    mongodb_database: str,
    mongodb_collection: str,
    elasticsearch_url: str,
    redis_url: str,
) -> PrerequisiteStatus:
    mongo_host = "127.0.0.1"
    mongo_port = 27018
    if "127.0.0.1:27017" in mongodb_uri or "localhost:27017" in mongodb_uri:
        mongo_port = 27017
    redis_host = "127.0.0.1"
    redis_port = 6379
    if redis_url.startswith("redis://"):
        redis_tail = redis_url.removeprefix("redis://").split("/", 1)[0]
        if ":" in redis_tail:
            redis_host, redis_port_text = redis_tail.split(":", 1)
            redis_port = int(redis_port_text)

    return PrerequisiteStatus(
        mongodb_reachable=tcp_reachable(mongo_host, mongo_port),
        elasticsearch_reachable=http_reachable(f"{elasticsearch_url.rstrip('/')}/"),
        redis_reachable=tcp_reachable(redis_host, redis_port),
        mongodb_uri_class="mongodb+srv_or_standard_local",
        mongodb_database=mongodb_database,
        mongodb_collection_authority=mongodb_collection,
        observability_backend="elasticsearch",
        observability_index=_DEFAULT_ELASTICSEARCH_INDEX,
    )


def build_qualification_environment(
    *,
    marker: str,
    attempt_data_home: Path,
    base_environment: Mapping[str, str] | None = None,
    instance_id: str | None = None,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    if base_environment is not None:
        merged.update(base_environment)

    mongo_uri = merged.get("INTERGRAX_MONGODB_URI", "").strip() or default_mongodb_uri()
    mongo_database = merged.get("INTERGRAX_MONGODB_DATABASE", "").strip() or "intergrax_proofs"
    mongo_collection = (
        merged.get("LKW_MANAGED_WORKSPACE_COLLECTION", "").strip() or "lkw_managed_workspaces"
    )
    redis_url = merged.get("INTERGRAX_REDIS_URL", "").strip() or _DEFAULT_REDIS_URL
    elasticsearch_url = (
        merged.get("LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL", "").strip()
        or _DEFAULT_ELASTICSEARCH_URL
    )
    elasticsearch_index = (
        merged.get("LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX", "").strip()
        or _DEFAULT_ELASTICSEARCH_INDEX
    )

    qualification_env = dict(merged)
    qualification_env.update(
        {
            "INTERGRAX_HARNESS_API_KEY": merged.get(
                "INTERGRAX_HARNESS_API_KEY",
                "dg001d-r4-qualification-harness-key",
            ),
            "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET": merged.get(
                "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
                "dg001d-r4-qualification-cursor-secret-32b-min",
            ),
            "LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS": "true",
            "LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS": "true",
            "LOCAL_WORKSPACE_ENABLE_REDIS": "true",
            "INTERGRAX_REDIS_URL": redis_url,
            "INTERGRAX_MONGODB_URI": mongo_uri,
            "INTERGRAX_MONGODB_DATABASE": mongo_database,
            "INTERGRAX_MONGODB_COLLECTION": mongo_collection,
            "LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND": "mongodb",
            "LKW_MANAGED_WORKSPACE_COLLECTION": mongo_collection,
            "DATA_HOME": attempt_data_home.as_posix(),
            "LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED": "true",
            "LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND": "elasticsearch",
            "LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL": elasticsearch_url,
            "LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX": elasticsearch_index,
            "LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT": marker,
            "LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_NAME": "intergrax-lkw",
            "LOCAL_WORKSPACE_VECTOR_STORE": "inmemory",
            "INTERGRAX_KAFKA_BOOTSTRAP_SERVERS": merged.get(
                "INTERGRAX_KAFKA_BOOTSTRAP_SERVERS",
                "127.0.0.1:9094",
            ),
            "INTERGRAX_KAFKA_TOPIC": merged.get("INTERGRAX_KAFKA_TOPIC", "intergrax.tasks"),
            "INTERGRAX_KAFKA_EVENTS_TOPIC": merged.get(
                "INTERGRAX_KAFKA_EVENTS_TOPIC",
                "intergrax.task-events",
            ),
            "INTERGRAX_KAFKA_STATUS_TOPIC": merged.get(
                "INTERGRAX_KAFKA_STATUS_TOPIC",
                "intergrax.task-status",
            ),
            "INTERGRAX_KAFKA_RESULTS_TOPIC": merged.get(
                "INTERGRAX_KAFKA_RESULTS_TOPIC",
                "intergrax.task-results",
            ),
            "INTERGRAX_KAFKA_CONSUMER_GROUP": merged.get(
                "INTERGRAX_KAFKA_CONSUMER_GROUP",
                "dg001d-r4-qualification",
            ),
        },
    )
    if instance_id:
        qualification_env["DG001D_R4_INSTANCE_ID"] = instance_id
    existing_pythonpath = merged.get("PYTHONPATH", "").strip()
    computed_pythonpath = _application_pythonpath()
    if existing_pythonpath:
        qualification_env["PYTHONPATH"] = os_pathsep_join(
            [computed_pythonpath, existing_pythonpath],
        )
    else:
        qualification_env["PYTHONPATH"] = computed_pythonpath
    return qualification_env


def _application_pythonpath() -> str:
    applications_root = _REPO_ROOT / "applications"
    agents_root = _REPO_ROOT / "agents"
    existing_pythonpath = os_environ_get("PYTHONPATH") or ""
    separator = ";" if sys.platform.startswith("win") else ":"
    entries = [str(applications_root), str(agents_root)]
    for entry in existing_pythonpath.split(separator):
        stripped = entry.strip()
        if stripped and stripped not in entries:
            entries.append(stripped)
    return separator.join(entries)


def spawn_supervisor_child(
    environment: Mapping[str, str],
    *,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> SupervisorProcessEvidence:
    started_at = datetime.now(UTC)
    completed = subprocess_runner(
        [
            sys.executable,
            str(_SUPERVISOR_CHILD_SCRIPT),
        ],
        cwd=str(_REPO_ROOT),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    finished_at = datetime.now(UTC)
    instance_id = _extract_instance_id_from_output(completed.stdout, completed.stderr)
    return SupervisorProcessEvidence(
        pid=None,
        exit_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        started_at=started_at,
        finished_at=finished_at,
        instance_id=instance_id,
    )


def _extract_instance_id_from_output(stdout: str, stderr: str) -> str | None:
    combined = f"{stdout}\n{stderr}"
    for line in combined.splitlines():
        if line.startswith("SUPERVISOR_INSTANCE_ID="):
            candidate = line.removeprefix("SUPERVISOR_INSTANCE_ID=").strip()
            if candidate:
                return candidate
    return None


def build_canonical_diagnostic_read_service(
    environment: Mapping[str, str],
) -> tuple[DiagnosticReadService, str]:
    previous: dict[str, str | None] = {}
    for key, value in environment.items():
        previous[key] = os_environ_get(key)
        os_environ_set(key, value)
    try:
        settings = LocalWorkspaceBackendSettings.from_env()
        environment_profile = build_local_workspace_environment_profile(settings)
        _, registry_projection = activate_local_workspace_reference_production_authority(
            settings,
            environment_profile=environment_profile,
        )
        document_store = resolve_lkw_runtime_document_store(settings)
        runtime = build_harness_host_runtime(
            LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            LOCAL_WORKSPACE_APPLICATION_MANIFEST.resolved_environment(),
            settings=settings,
            idempotency_db_path=Path(settings.idempotency_db_path),
            document_store=document_store,
            registry_projection=registry_projection,
        )
        read_service = build_diagnostic_read_service(
            resolve_host_diagnostic_read_dependencies(runtime),
        )
        return read_service, environment_profile.profile_id
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os_environ_pop(key)
            else:
                os_environ_set(key, old_value)


def _signal_signature_fields(
    detail: DiagnosticProblemDetail,
) -> tuple[str, str, str] | None:
    signature = detail.grouping_provenance.deterministic_signature
    if signature is None:
        return None
    for finding in signature.findings:
        if type(finding) is not DeterministicSignalFindingSignature:
            continue
        return finding.source_component, finding.error_code or "", finding.exception_type or ""
    return None


def _occurrence_from_view(occurrence_view: DiagnosticProblemOccurrenceView) -> ProblemOccurrence:
    return ProblemOccurrence(
        subject_ref=occurrence_view.subject_ref,
        observed_at=occurrence_view.observed_at,
        strategy_id=occurrence_view.strategy_id,
        strategy_version=occurrence_view.strategy_version,
        method=occurrence_view.method,
    )


def read_diagnostic_evidence(
    environment: Mapping[str, str],
    *,
    marker: str,
    not_before: datetime | None = None,
    expected_instance_id: str | None = None,
) -> DiagnosticReadEvidence | None:
    read_service, tenant_id = build_canonical_diagnostic_read_service(environment)
    listed = read_service.list_problems(tenant_id=tenant_id)

    candidates: list[tuple[datetime, DiagnosticReadEvidence]] = []
    for summary in listed.problems:
        detail = read_service.get_problem(tenant_id=tenant_id, problem_id=summary.problem_id)
        if detail is None:
            continue
        signature_fields = _signal_signature_fields(detail)
        if signature_fields is None:
            continue
        phase, reason_code, exception_type = signature_fields
        if phase != _EXPECTED_PHASE:
            continue
        if reason_code != _EXPECTED_REASON_CODE:
            continue
        if exception_type != _EXPECTED_EXCEPTION_TYPE:
            continue
        for occurrence_view in detail.occurrences:
            if not_before is not None and occurrence_view.observed_at < not_before:
                continue
            application_subject = occurrence_view.subject_ref.application_instance()
            if application_subject is None:
                continue
            if application_subject.application_id != _EXPECTED_APPLICATION_ID:
                continue
            if expected_instance_id is not None and application_subject.instance_id != expected_instance_id:
                continue
            occurrence = _occurrence_from_view(occurrence_view)
            occurrence_id = str(problem_occurrence_id_for(occurrence))
            serialized = json.dumps(
                {
                    "problem_id": str(summary.problem_id),
                    "occurrence_id": occurrence_id,
                    "marker": marker,
                },
            )
            if _SECRET_SENTINEL in serialized:
                continue
            evidence = DiagnosticReadEvidence(
                application_id=application_subject.application_id,
                instance_id=application_subject.instance_id,
                tenant_id=tenant_id,
                problem_id=str(summary.problem_id),
                occurrence_id=occurrence_id,
                subject_kind=occurrence_view.subject_ref.kind,
                phase=phase,
                reason_code=reason_code,
                exception_type=exception_type,
                task_id="",
                run_id="",
                attempt_id="",
                execution_id="",
                occurrence_count=detail.occurrence_count,
            )
            candidates.append((occurrence_view.observed_at, evidence))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def read_problem_occurrence_count(
    environment: Mapping[str, str],
    *,
    problem_id: str,
) -> int | None:
    read_service, tenant_id = build_canonical_diagnostic_read_service(environment)
    detail = read_service.get_problem(tenant_id=tenant_id, problem_id=problem_id)
    if detail is None:
        return None
    return detail.occurrence_count


def query_observability_export_by_instance(
    *,
    elasticsearch_url: str,
    index: str,
    instance_id: str,
    marker: str,
) -> ObservabilityExportEvidence | None:
    from scripts.proof.dg001b_r5_qualification_support import query_observability_export_by_instance as _query

    return _query(
        elasticsearch_url=elasticsearch_url,
        index=index,
        instance_id=instance_id,
        marker=marker,
    )


__all__ = [
    "DiagnosticReadEvidence",
    "ObservabilityExportEvidence",
    "PrerequisiteStatus",
    "QualificationAttemptResult",
    "QualificationGate",
    "SupervisorProcessEvidence",
    "build_canonical_diagnostic_read_service",
    "build_qualification_environment",
    "default_mongodb_uri",
    "ensure_mongodb_replica_set_for_qualification",
    "evaluate_prerequisites",
    "qualification_mongodb_collection",
    "qualification_mongodb_database",
    "query_observability_export_by_instance",
    "read_diagnostic_evidence",
    "read_problem_occurrence_count",
    "repository_root",
    "spawn_supervisor_child",
]
