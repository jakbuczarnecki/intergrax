#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DG-001D R4 — real supervisor pre-engine failure qualification harness."""

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

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from intergrax.runtime.diagnostics.diagnostic_subject import DiagnosticSubjectKind
from scripts.proof.dg001d_r4_qualification_contracts import qualification_secret_sentinel
from scripts.proof.dg001d_r4_qualification_support import (
    DiagnosticReadEvidence,
    ObservabilityExportEvidence,
    PrerequisiteStatus,
    QualificationAttemptResult,
    QualificationGate,
    SupervisorProcessEvidence,
    build_qualification_environment,
    default_mongodb_uri,
    ensure_mongodb_replica_set_for_qualification,
    evaluate_prerequisites,
    qualification_mongodb_collection,
    qualification_mongodb_database,
    query_observability_export_by_instance,
    read_diagnostic_evidence,
    read_problem_occurrence_count,
    repository_root,
    spawn_supervisor_child,
)
from scripts.proof.intergrax_proof_environment import resolve_proof_environment

_READER_MODE = "reader"
_SUPERVISOR_MODE = "supervisor"
_DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
_DEFAULT_ELASTICSEARCH_INDEX = "intergrax-lkw-observability"
_DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
_REDIS_CONTAINER_NAME = "dg001d-r4-qual-redis"
_SECRET_SENTINEL = qualification_secret_sentinel()


@dataclass(frozen=True, slots=True)
class QualificationRunConfig:
    attempt_id: str
    marker: str
    attempt_data_home: Path
    evidence_dir: Path
    mongodb_uri: str
    mongodb_database: str
    mongodb_collection: str
    elasticsearch_url: str
    elasticsearch_index: str
    redis_url: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-id",
        default="",
        help="Immutable qualification attempt id (default: dg001d-r4-<utc-timestamp>).",
    )
    parser.add_argument(
        "--mode",
        choices=(_SUPERVISOR_MODE, _READER_MODE, "orchestrate"),
        default="orchestrate",
        help="Supervisor child, reader child, or full orchestration.",
    )
    parser.add_argument("--marker", default="", help="Qualification marker for reader mode.")
    parser.add_argument(
        "--env-json",
        default="",
        help="Reader/supervisor mode: JSON file containing the resolved qualification environment.",
    )
    parser.add_argument(
        "--not-before-iso",
        default="",
        help="Reader mode: ignore occurrences observed before this UTC timestamp.",
    )
    parser.add_argument(
        "--instance-id",
        default="",
        help="Supervisor mode: fixed instance_id for recurrence qualification.",
    )
    parser.add_argument(
        "--expected-instance-id",
        default="",
        help="Reader mode: filter to a specific supervisor instance_id.",
    )
    parser.add_argument(
        "--ensure-redis",
        action="store_true",
        help="Start ephemeral local Redis on 6379 when unreachable.",
    )
    parser.add_argument(
        "--ensure-mongo-replica-set",
        action="store_true",
        help="Start local Mongo replica-set stack required for durable Problem persistence.",
    )
    return parser.parse_args()


def _build_run_config(attempt_id: str) -> QualificationRunConfig:
    repo_root = repository_root()
    resolved = resolve_proof_environment(
        proof_package_dir=repo_root / "scripts" / "proof",
        repository_root=repo_root,
    )
    marker = attempt_id
    attempt_data_home = repo_root / ".tmp" / "session" / "dg001d-r4" / attempt_id / "data"
    evidence_dir = repo_root / ".tmp" / "session" / "dg001d-r4" / attempt_id
    attempt_data_home.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    mongodb_uri = resolved.environment.get("INTERGRAX_MONGODB_URI", "").strip() or default_mongodb_uri()
    mongodb_database = (
        resolved.environment.get("INTERGRAX_MONGODB_DATABASE", "").strip() or "intergrax_proofs"
    )
    mongodb_collection = (
        resolved.environment.get("LKW_MANAGED_WORKSPACE_COLLECTION", "").strip()
        or "lkw_managed_workspaces"
    )
    elasticsearch_url = (
        resolved.environment.get("LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL", "").strip()
        or _DEFAULT_ELASTICSEARCH_URL
    )
    elasticsearch_index = (
        resolved.environment.get("LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX", "").strip()
        or _DEFAULT_ELASTICSEARCH_INDEX
    )
    redis_url = resolved.environment.get("INTERGRAX_REDIS_URL", "").strip() or _DEFAULT_REDIS_URL
    return QualificationRunConfig(
        attempt_id=attempt_id,
        marker=marker,
        attempt_data_home=attempt_data_home,
        evidence_dir=evidence_dir,
        mongodb_uri=mongodb_uri,
        mongodb_database=mongodb_database,
        mongodb_collection=mongodb_collection,
        elasticsearch_url=elasticsearch_url,
        elasticsearch_index=elasticsearch_index,
        redis_url=redis_url,
    )


def _ensure_ephemeral_redis() -> str:
    probe = evaluate_prerequisites(
        mongodb_uri=default_mongodb_uri(),
        mongodb_database="intergrax_proofs",
        mongodb_collection="lkw_managed_workspaces",
        elasticsearch_url=_DEFAULT_ELASTICSEARCH_URL,
        redis_url=_DEFAULT_REDIS_URL,
    )
    if probe.redis_reachable:
        return _DEFAULT_REDIS_URL

    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", _REDIS_CONTAINER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0 and inspect.stdout.strip().lower() == "true":
        return _DEFAULT_REDIS_URL

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            _REDIS_CONTAINER_NAME,
            "-p",
            "6379:6379",
            "redis:7-alpine",
        ],
        check=True,
    )
    for _ in range(20):
        probe = evaluate_prerequisites(
            mongodb_uri=default_mongodb_uri(),
            mongodb_database="intergrax_proofs",
            mongodb_collection="lkw_managed_workspaces",
            elasticsearch_url=_DEFAULT_ELASTICSEARCH_URL,
            redis_url=_DEFAULT_REDIS_URL,
        )
        if probe.redis_reachable:
            return _DEFAULT_REDIS_URL
        time.sleep(0.5)
    raise RuntimeError("ephemeral_redis_unavailable")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_environment(path: Path) -> dict[str, str]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("qualification environment json must be an object")
    environment: dict[str, str] = {}
    for key, value in loaded.items():
        environment[str(key)] = str(value)
    return environment


def _diagnostic_to_json(evidence: DiagnosticReadEvidence) -> dict[str, str | int]:
    return {
        "application_id": evidence.application_id,
        "instance_id": evidence.instance_id,
        "tenant_id": evidence.tenant_id,
        "problem_id": evidence.problem_id,
        "occurrence_id": evidence.occurrence_id,
        "subject_kind": evidence.subject_kind.value,
        "phase": evidence.phase,
        "reason_code": evidence.reason_code,
        "exception_type": evidence.exception_type,
        "task_id": evidence.task_id,
        "run_id": evidence.run_id,
        "attempt_id": evidence.attempt_id,
        "execution_id": evidence.execution_id,
        "occurrence_count": evidence.occurrence_count,
    }


def _reader_main(args: argparse.Namespace) -> int:
    if not args.env_json:
        raise ValueError("reader mode requires --env-json")
    environment = _load_environment(Path(args.env_json))
    not_before = None
    if args.not_before_iso:
        not_before = datetime.fromisoformat(args.not_before_iso)
        if not_before.tzinfo is None:
            not_before = not_before.replace(tzinfo=UTC)
    expected_instance_id = args.expected_instance_id.strip() or None
    evidence = read_diagnostic_evidence(
        environment,
        marker=args.marker,
        not_before=not_before,
        expected_instance_id=expected_instance_id,
    )
    if evidence is None:
        print("READER_RESULT=null")
        return 2
    print("READER_RESULT=" + json.dumps(_diagnostic_to_json(evidence), sort_keys=True))
    return 0


def _supervisor_to_json(supervisor: SupervisorProcessEvidence) -> dict[str, str | int | None]:
    return {
        "pid": supervisor.pid,
        "exit_code": supervisor.exit_code,
        "stdout": supervisor.stdout,
        "stderr": supervisor.stderr,
        "started_at": supervisor.started_at.isoformat(),
        "finished_at": supervisor.finished_at.isoformat(),
        "instance_id": supervisor.instance_id,
    }


def _supervisor_main(config: QualificationRunConfig, environment: dict[str, str]) -> int:
    supervisor = spawn_supervisor_child(environment)
    _write_json(config.evidence_dir / "supervisor_process.json", _supervisor_to_json(supervisor))
    print(f"SUPERVISOR_EXIT_CODE={supervisor.exit_code}")
    return supervisor.exit_code


def _spawn_reader_subprocess(
    *,
    config: QualificationRunConfig,
    environment: dict[str, str],
    not_before: datetime,
    expected_instance_id: str | None = None,
) -> DiagnosticReadEvidence | None:
    env_path = config.evidence_dir / "qualification_environment.json"
    _write_json(env_path, environment)
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script_path),
        "--mode",
        _READER_MODE,
        "--marker",
        config.marker,
        "--env-json",
        str(env_path),
        "--not-before-iso",
        not_before.isoformat(),
    ]
    if expected_instance_id:
        command.extend(["--expected-instance-id", expected_instance_id])
    completed = subprocess.run(
        command,
        cwd=str(repository_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    _write_json(
        config.evidence_dir / "reader_process.json",
        {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    for line in completed.stdout.splitlines():
        if line.startswith("READER_RESULT="):
            payload = line.removeprefix("READER_RESULT=").strip()
            if payload == "null":
                return None
            decoded = json.loads(payload)
            return DiagnosticReadEvidence(
                application_id=str(decoded["application_id"]),
                instance_id=str(decoded["instance_id"]),
                tenant_id=str(decoded["tenant_id"]),
                problem_id=str(decoded["problem_id"]),
                occurrence_id=str(decoded["occurrence_id"]),
                subject_kind=DiagnosticSubjectKind(str(decoded["subject_kind"])),
                phase=str(decoded["phase"]),
                reason_code=str(decoded["reason_code"]),
                exception_type=str(decoded["exception_type"]),
                task_id=str(decoded.get("task_id", "")),
                run_id=str(decoded.get("run_id", "")),
                attempt_id=str(decoded.get("attempt_id", "")),
                execution_id=str(decoded.get("execution_id", "")),
                occurrence_count=int(decoded.get("occurrence_count", 0)),
            )
    return None


def _evaluate_gates(
    *,
    prerequisites: PrerequisiteStatus,
    supervisor: SupervisorProcessEvidence | None,
    observability: ObservabilityExportEvidence | None,
    diagnostic_read: DiagnosticReadEvidence | None,
    recurrence_passed: bool,
) -> dict[QualificationGate, bool]:
    return {
        QualificationGate.E0_ENVIRONMENT: prerequisites.environment_ready,
        QualificationGate.E1_SUPERVISOR_PROCESS: supervisor is not None,
        QualificationGate.E2_SUPERVISOR_IDENTITY: bool(
            diagnostic_read is not None
            and diagnostic_read.application_id
            and diagnostic_read.instance_id
            and diagnostic_read.tenant_id,
        ),
        QualificationGate.E3_PRE_ENGINE_FAILURE: supervisor is not None and supervisor.exit_code != 0,
        QualificationGate.E4_FAILURE_EVENT: bool(
            observability is not None
            and observability.event_type == "hosting.application.failed"
            and observability.lifecycle_state == "failed",
        ),
        QualificationGate.E5_OBSERVABILITY_EXPORT: observability is not None,
        QualificationGate.E6_DIAGNOSTIC_PROJECTION: diagnostic_read is not None,
        QualificationGate.E7_DURABLE_PERSISTENCE: diagnostic_read is not None,
        QualificationGate.E8_OPERATOR_READ: diagnostic_read is not None and recurrence_passed,
        QualificationGate.E9_FAILURE_SAFETY: bool(
            supervisor is not None
            and supervisor.exit_code != 0
            and diagnostic_read is not None
            and diagnostic_read.subject_kind.value == "application_instance"
            and diagnostic_read.task_id == ""
            and diagnostic_read.run_id == ""
            and diagnostic_read.attempt_id == ""
            and diagnostic_read.execution_id == ""
            and diagnostic_read.phase == "engine_construction"
            and diagnostic_read.reason_code == "engine_factory_failed"
            and diagnostic_read.exception_type == "HostedApplicationSupervisorError"
            and _SECRET_SENTINEL not in json.dumps(_diagnostic_to_json(diagnostic_read))
        ),
    }


def _identity_fidelity(
    observability: ObservabilityExportEvidence | None,
    diagnostic_read: DiagnosticReadEvidence | None,
) -> int:
    if observability is None or diagnostic_read is None:
        return 0
    checks = [
        observability.application_id == diagnostic_read.application_id,
        observability.instance_id == diagnostic_read.instance_id,
        diagnostic_read.phase == "engine_construction",
        diagnostic_read.reason_code == "engine_factory_failed",
    ]
    matched = sum(1 for item in checks if item)
    return int((matched / len(checks)) * 100)


def _emit_result(result: QualificationAttemptResult, *, identity_fidelity: int = 0) -> None:
    print(f"QUALIFICATION_ATTEMPT_ID={result.attempt_id}")
    print(f"QUALIFICATION_MARKER={result.marker}")
    if result.blocked_reason:
        print(f"QUALIFICATION_BLOCKED={result.blocked_reason}")
    for gate, passed in result.gate_results.items():
        print(f"{gate.value}={'PASS' if passed else 'FAIL'}")
    print(f"IDENTITY_FIDELITY={identity_fidelity}%")
    if all(result.gate_results.values()):
        print("QUALIFICATION_VERDICT=QUALIFIED")
    elif result.blocked_reason == "required_real_backends_unavailable":
        print("QUALIFICATION_VERDICT=BLOCKED")
    else:
        print("QUALIFICATION_VERDICT=FAILED")


def _orchestrate(args: argparse.Namespace) -> int:
    attempt_id = args.attempt_id.strip() or f"dg001d-r4-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    config = _build_run_config(attempt_id)
    if args.ensure_redis:
        config = QualificationRunConfig(
            attempt_id=config.attempt_id,
            marker=config.marker,
            attempt_data_home=config.attempt_data_home,
            evidence_dir=config.evidence_dir,
            mongodb_uri=config.mongodb_uri,
            mongodb_database=config.mongodb_database,
            mongodb_collection=config.mongodb_collection,
            elasticsearch_url=config.elasticsearch_url,
            elasticsearch_index=config.elasticsearch_index,
            redis_url=_ensure_ephemeral_redis(),
        )

    repo_root = repository_root()
    resolved = resolve_proof_environment(
        proof_package_dir=repo_root / "scripts" / "proof",
        repository_root=repo_root,
    )
    if args.ensure_mongo_replica_set:
        mongo_uri = ensure_mongodb_replica_set_for_qualification()
        mongo_database = qualification_mongodb_database()
        mongo_collection = qualification_mongodb_collection()
        resolved.environment["INTERGRAX_MONGODB_URI"] = mongo_uri
        resolved.environment["INTERGRAX_MONGODB_DATABASE"] = mongo_database
        resolved.environment["LKW_MANAGED_WORKSPACE_COLLECTION"] = mongo_collection
        config = QualificationRunConfig(
            attempt_id=config.attempt_id,
            marker=config.marker,
            attempt_data_home=config.attempt_data_home,
            evidence_dir=config.evidence_dir,
            mongodb_uri=mongo_uri,
            mongodb_database=mongo_database,
            mongodb_collection=mongo_collection,
            elasticsearch_url=config.elasticsearch_url,
            elasticsearch_index=config.elasticsearch_index,
            redis_url=config.redis_url,
        )

    prerequisites = evaluate_prerequisites(
        mongodb_uri=config.mongodb_uri,
        mongodb_database=config.mongodb_database,
        mongodb_collection=config.mongodb_collection,
        elasticsearch_url=config.elasticsearch_url,
        redis_url=config.redis_url,
    )
    if not prerequisites.environment_ready:
        result = QualificationAttemptResult(
            attempt_id=config.attempt_id,
            marker=config.marker,
            prerequisites=prerequisites,
            supervisor=None,
            observability=None,
            diagnostic_read=None,
            gate_results={gate: False for gate in QualificationGate},
            blocked_reason="required_real_backends_unavailable",
        )
        _emit_result(result)
        return 3

    resolved.environment["INTERGRAX_REDIS_URL"] = config.redis_url
    base_environment = build_qualification_environment(
        marker=config.marker,
        attempt_data_home=config.attempt_data_home,
        base_environment=resolved.environment,
    )
    _write_json(config.evidence_dir / "qualification_environment.json", base_environment)

    recurrence_passed = False
    recurrence_payload: dict[str, object] | None = None
    first_instance_id = "dg001d-r4-instance-001"
    second_instance_id = "dg001d-r4-instance-002"

    first_environment = build_qualification_environment(
        marker=config.marker,
        attempt_data_home=config.attempt_data_home,
        base_environment=base_environment,
        instance_id=first_instance_id,
    )
    first_supervisor = spawn_supervisor_child(first_environment)
    _write_json(
        config.evidence_dir / "supervisor_process_first.json",
        _supervisor_to_json(first_supervisor),
    )
    first_read = _spawn_reader_subprocess(
        config=config,
        environment=first_environment,
        not_before=first_supervisor.started_at,
        expected_instance_id=first_instance_id,
    )
    observability = None
    if first_read is not None:
        observability = query_observability_export_by_instance(
            elasticsearch_url=config.elasticsearch_url,
            index=config.elasticsearch_index,
            instance_id=first_read.instance_id,
            marker=config.marker,
        )

    if first_read is not None and first_read.occurrence_count == 1:
        second_environment = build_qualification_environment(
            marker=config.marker,
            attempt_data_home=config.attempt_data_home,
            base_environment=base_environment,
            instance_id=second_instance_id,
        )
        second_supervisor = spawn_supervisor_child(second_environment)
        _write_json(
            config.evidence_dir / "supervisor_process_second.json",
            _supervisor_to_json(second_supervisor),
        )
        second_read = _spawn_reader_subprocess(
            config=config,
            environment=second_environment,
            not_before=second_supervisor.started_at,
            expected_instance_id=second_instance_id,
        )
        final_count = read_problem_occurrence_count(
            second_environment,
            problem_id=first_read.problem_id,
        )
        recurrence_passed = (
            second_read is not None
            and final_count == 2
            and first_read.problem_id == second_read.problem_id
        )
        recurrence_payload = {
            "first_instance_id": first_instance_id,
            "second_instance_id": second_instance_id,
            "problem_id": first_read.problem_id,
            "first_occurrence_count": first_read.occurrence_count,
            "final_occurrence_count": final_count,
            "recurrence_passed": recurrence_passed,
        }
        _write_json(config.evidence_dir / "recurrence.json", recurrence_payload)

    diagnostic_read = first_read
    supervisor = first_supervisor
    gate_results = _evaluate_gates(
        prerequisites=prerequisites,
        supervisor=supervisor,
        observability=observability,
        diagnostic_read=diagnostic_read,
        recurrence_passed=recurrence_passed,
    )
    identity_fidelity = _identity_fidelity(observability, diagnostic_read)
    result = QualificationAttemptResult(
        attempt_id=config.attempt_id,
        marker=config.marker,
        prerequisites=prerequisites,
        supervisor=supervisor,
        observability=observability,
        diagnostic_read=diagnostic_read,
        gate_results=gate_results,
        blocked_reason=None if all(gate_results.values()) else "gate_failure",
    )
    payload = {
        "attempt_id": result.attempt_id,
        "marker": result.marker,
        "gate_results": {gate.value: passed for gate, passed in gate_results.items()},
        "identity_fidelity_percent": identity_fidelity,
        "blocked_reason": result.blocked_reason,
        "supervisor_exit_code": supervisor.exit_code,
        "diagnostic_read": (
            _diagnostic_to_json(diagnostic_read) if diagnostic_read is not None else None
        ),
        "observability": (
            {
                "event_id": observability.event_id,
                "application_id": observability.application_id,
                "instance_id": observability.instance_id,
                "lifecycle_state": observability.lifecycle_state,
                "event_type": observability.event_type,
                "occurred_at": observability.occurred_at,
            }
            if observability is not None
            else None
        ),
        "recurrence": recurrence_payload,
    }
    _write_json(config.evidence_dir / "qualification_result.json", payload)
    _emit_result(result, identity_fidelity=identity_fidelity)
    return 0 if all(gate_results.values()) else 4


def main() -> int:
    args = _parse_args()
    if args.mode == _READER_MODE:
        return _reader_main(args)
    if args.mode == _SUPERVISOR_MODE:
        attempt_id = args.attempt_id.strip() or args.marker.strip()
        if not attempt_id:
            raise ValueError("supervisor mode requires --attempt-id or --marker")
        config = _build_run_config(attempt_id)
        if not args.env_json:
            raise ValueError("supervisor mode requires --env-json")
        environment = _load_environment(Path(args.env_json))
        if args.instance_id.strip():
            environment["DG001D_R4_INSTANCE_ID"] = args.instance_id.strip()
        return _supervisor_main(config, environment)
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
