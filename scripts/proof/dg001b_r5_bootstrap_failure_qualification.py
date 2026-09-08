#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DG-001B R5 — real controlled worker bootstrap failure qualification harness."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "applications"
for _bootstrap_entry in (str(_APPLICATIONS_BOOTSTRAP_ROOT), str(_REPO_BOOTSTRAP_ROOT)):
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
from scripts.proof.dg001b_r5_qualification_contracts import qualification_secret_sentinel
from scripts.proof.dg001b_r5_qualification_support import (
    DiagnosticReadEvidence,
    ObservabilityExportEvidence,
    PrerequisiteStatus,
    QualificationAttemptResult,
    QualificationGate,
    WorkerProcessEvidence,
    build_qualification_environment,
    default_mongodb_uri,
    ensure_mongodb_replica_set_for_qualification,
    evaluate_prerequisites,
    qualification_mongodb_collection,
    qualification_mongodb_database,
    query_observability_export_by_instance,
    read_diagnostic_evidence,
    repository_root,
    spawn_worker_child,
)
from scripts.proof.intergrax_proof_environment import resolve_proof_environment

_READER_MODE = "reader"
_WORKER_MODE = "worker"
_DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
_DEFAULT_ELASTICSEARCH_INDEX = "intergrax-lkw-observability"
_DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
_REDIS_CONTAINER_NAME = "dg001b-r5-qual-redis"
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
        help="Immutable qualification attempt id (default: dg001b-r5-<utc-timestamp>).",
    )
    parser.add_argument(
        "--mode",
        choices=(_WORKER_MODE, _READER_MODE, "orchestrate"),
        default="orchestrate",
        help="Worker child, reader child, or full orchestration.",
    )
    parser.add_argument("--marker", default="", help="Qualification marker for reader mode.")
    parser.add_argument(
        "--env-json",
        default="",
        help="Reader mode: JSON file containing the resolved qualification environment.",
    )
    parser.add_argument(
        "--not-before-iso",
        default="",
        help="Reader mode: ignore occurrences observed before this UTC timestamp.",
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
    attempt_data_home = repo_root / ".tmp" / "session" / "dg001b-r5" / attempt_id / "data"
    evidence_dir = repo_root / ".tmp" / "session" / "dg001b-r5" / attempt_id
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


def _reader_main(args: argparse.Namespace) -> int:
    if not args.env_json:
        raise ValueError("reader mode requires --env-json")
    environment = _load_environment(Path(args.env_json))
    not_before = None
    if args.not_before_iso:
        not_before = datetime.fromisoformat(args.not_before_iso)
        if not_before.tzinfo is None:
            not_before = not_before.replace(tzinfo=UTC)
    evidence = read_diagnostic_evidence(
        environment,
        marker=args.marker,
        not_before=not_before,
    )
    if evidence is None:
        print("READER_RESULT=null")
        return 2
    print("READER_RESULT=" + json.dumps(_diagnostic_to_json(evidence), sort_keys=True))
    return 0


def _diagnostic_to_json(evidence: DiagnosticReadEvidence) -> dict[str, str]:
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
    }


def _worker_main(config: QualificationRunConfig, environment: dict[str, str]) -> int:
    worker = spawn_worker_child(environment)
    _write_json(config.evidence_dir / "worker_process.json", _worker_to_json(worker))
    print(f"WORKER_EXIT_CODE={worker.exit_code}")
    return worker.exit_code


def _worker_to_json(worker: WorkerProcessEvidence) -> dict[str, str | int | None]:
    return {
        "pid": worker.pid,
        "exit_code": worker.exit_code,
        "stdout": worker.stdout,
        "stderr": worker.stderr,
        "started_at": worker.started_at.isoformat(),
        "finished_at": worker.finished_at.isoformat(),
    }


def _spawn_reader_subprocess(
    *,
    config: QualificationRunConfig,
    environment: dict[str, str],
    not_before: datetime,
) -> DiagnosticReadEvidence | None:
    env_path = config.evidence_dir / "qualification_environment.json"
    _write_json(env_path, environment)
    script_path = Path(__file__).resolve()
    completed = subprocess.run(
        [
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
        ],
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
            )
    return None


def _evaluate_gates(
    *,
    prerequisites: PrerequisiteStatus,
    worker: WorkerProcessEvidence | None,
    observability: ObservabilityExportEvidence | None,
    diagnostic_read: DiagnosticReadEvidence | None,
) -> dict[QualificationGate, bool]:
    return {
        QualificationGate.E0_ENVIRONMENT: prerequisites.environment_ready,
        QualificationGate.E1_WORKER_PROCESS: worker is not None,
        QualificationGate.E2_BOOTSTRAP_IDENTITY: bool(
            diagnostic_read is not None
            and diagnostic_read.application_id
            and diagnostic_read.instance_id
            and diagnostic_read.tenant_id,
        ),
        QualificationGate.E3_B6_FAILURE: worker is not None and worker.exit_code != 0,
        QualificationGate.E4_FAILURE_EVENT: bool(
            observability is not None
            and observability.event_type == "hosting.application.failed"
            and observability.lifecycle_state == "failed",
        ),
        QualificationGate.E5_OBSERVABILITY_EXPORT: observability is not None,
        QualificationGate.E6_DIAGNOSTIC_PROJECTION: diagnostic_read is not None,
        QualificationGate.E7_DURABLE_PERSISTENCE: diagnostic_read is not None,
        QualificationGate.E8_OPERATOR_READ: diagnostic_read is not None,
        QualificationGate.E9_FAILURE_SAFETY: bool(
            worker is not None
            and worker.exit_code != 0
            and diagnostic_read is not None
            and diagnostic_read.subject_kind.value == "application_instance"
            and diagnostic_read.task_id == ""
            and diagnostic_read.run_id == ""
            and diagnostic_read.attempt_id == ""
            and diagnostic_read.execution_id == ""
            and _SECRET_SENTINEL not in json.dumps(_diagnostic_to_json(diagnostic_read))
        ),
    }


def _orchestrate(args: argparse.Namespace) -> int:
    attempt_id = args.attempt_id.strip() or f"dg001b-r5-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
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
            worker=None,
            observability=None,
            diagnostic_read=None,
            gate_results={gate: False for gate in QualificationGate},
            blocked_reason="required_real_backends_unavailable",
        )
        _emit_result(result)
        return 3

    resolved.environment["INTERGRAX_REDIS_URL"] = config.redis_url
    environment = build_qualification_environment(
        marker=config.marker,
        attempt_data_home=config.attempt_data_home,
        base_environment=resolved.environment,
    )
    _write_json(config.evidence_dir / "qualification_environment.json", environment)

    worker = spawn_worker_child(environment)
    _write_json(config.evidence_dir / "worker_process.json", _worker_to_json(worker))

    diagnostic_read = _spawn_reader_subprocess(
        config=config,
        environment=environment,
        not_before=worker.started_at,
    )
    observability = None
    if diagnostic_read is not None:
        observability = query_observability_export_by_instance(
            elasticsearch_url=config.elasticsearch_url,
            index=config.elasticsearch_index,
            instance_id=diagnostic_read.instance_id,
            marker=config.marker,
        )
        if observability is not None:
            _write_json(
                config.evidence_dir / "observability_export.json",
                {
                    "event_id": observability.event_id,
                    "application_id": observability.application_id,
                    "instance_id": observability.instance_id,
                    "lifecycle_state": observability.lifecycle_state,
                    "event_type": observability.event_type,
                    "occurred_at": observability.occurred_at,
                    "qualification_marker": observability.qualification_marker,
                    "backend_id": observability.backend_id,
                    "index": observability.index,
                },
            )

    gate_results = _evaluate_gates(
        prerequisites=prerequisites,
        worker=worker,
        observability=observability,
        diagnostic_read=diagnostic_read,
    )
    identity_fidelity = _identity_fidelity(observability, diagnostic_read)
    result = QualificationAttemptResult(
        attempt_id=config.attempt_id,
        marker=config.marker,
        prerequisites=prerequisites,
        worker=worker,
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
        "worker_exit_code": worker.exit_code,
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
    }
    _write_json(config.evidence_dir / "qualification_result.json", payload)
    _emit_result(result, identity_fidelity=identity_fidelity)
    return 0 if all(gate_results.values()) else 4


def _identity_fidelity(
    observability: ObservabilityExportEvidence | None,
    diagnostic_read: DiagnosticReadEvidence | None,
) -> int:
    if observability is None or diagnostic_read is None:
        return 0
    checks = [
        observability.application_id == diagnostic_read.application_id,
        observability.instance_id == diagnostic_read.instance_id,
        diagnostic_read.phase == "worker_construction",
    ]
    if not checks:
        return 0
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


def main() -> int:
    args = _parse_args()
    if args.mode == _READER_MODE:
        return _reader_main(args)
    if args.mode == _WORKER_MODE:
        attempt_id = args.attempt_id.strip() or args.marker.strip()
        if not attempt_id:
            raise ValueError("worker mode requires --attempt-id or --marker")
        config = _build_run_config(attempt_id)
        if not args.env_json:
            raise ValueError("worker mode requires --env-json")
        environment = _load_environment(Path(args.env_json))
        return _worker_main(config, environment)
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
