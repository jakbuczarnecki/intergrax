#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Shared cross-platform LKW Core Platform Proof runner.

OS launchers are transport-only. Proof orchestration and acceptance live here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"
_BASE_COMPOSE = _DOCKER_DIR / "docker-compose.yml"
_ES_COMPOSE = _DOCKER_DIR / "docker-compose.elasticsearch.yml"
_KAFKA_COMPOSE = _DOCKER_DIR / "docker-compose.kafka.yml"
_MONGODB_COMPOSE = _DOCKER_DIR / "docker-compose.mongodb.yml"
_WATCHER_COMPOSE = _DOCKER_DIR / "file-watcher-e2e.compose.yml"
_SENTRY_PROOF_DIR = _DOCKER_DIR / "sentry-proof"
_SAMPLE_DOCS_DIR = _APP_DIR / "sample_docs"
_PROOF_DOCS_DIR = _APP_DIR / ".proof_docs"
_WATCHER_STATE_DIR = _APP_DIR / ".file_watcher_e2e_state"

_SENTRY_PROOF_PY = _SCRIPT_DIR / "run-sentry-observability-proof.py"
_ES_INSPECT_PY = _SCRIPT_DIR / "inspect_elasticsearch_observability.py"
_BACKGROUND_PROOF_PY = _SCRIPT_DIR / "run-lkw-background-task-proof.py"
_HOSTING_PROOF_PY = _SCRIPT_DIR / "run-lkw-hosting-proof.py"
_FILE_WATCHER_PROOF_PY = _SCRIPT_DIR / "run-lkw-file-watcher-e2e-proof.py"

_DEFAULT_BASE_URL = "http://127.0.0.1:8020"
_DEFAULT_KAFKA_UI = "http://127.0.0.1:8085"
_DEFAULT_MONGO_EXPRESS = "http://127.0.0.1:8086"
_DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
_DEFAULT_KIBANA_URL = "http://127.0.0.1:5601"
_DEFAULT_SENTRY_URL = "http://127.0.0.1:9000"
_DEFAULT_ES_INDEX = "intergrax-lkw-observability"
_DEFAULT_PHASE_TIMEOUT = 600

_SEARCH_REASON_RETRIEVE_COMPLETE = "retrieve_complete"
_SAFE_REASON = re.compile(r"^[A-Za-z0-9_.-]+$")
_KV_LINE = re.compile(r"^([A-Za-z0-9_.-]+)=(.*)$")
_FILE_WATCHER_SCOPE_ID = "lkw-file-watcher-e2e"
_FILE_WATCHER_USER_ID = "lkw.file_watcher"
_FILE_WATCHER_SEED_NAME = "lkw_core_proof_embed_seed.txt"

ALL_PHASE_ORDER: tuple[str, ...] = (
    "startup",
    "sentry",
    "elasticsearch",
    "persistence",
    "background-task",
    "application-hosting",
    "file-watcher",
)


class OsFamily(str, Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"


class WrapperId(str, Enum):
    WINDOWS_BAT = "windows_bat"
    LINUX_SH = "linux_sh"
    MACOS_SH = "macos_sh"


VALID_OS_WRAPPER_PAIRS: frozenset[tuple[OsFamily, WrapperId]] = frozenset(
    {
        (OsFamily.WINDOWS, WrapperId.WINDOWS_BAT),
        (OsFamily.LINUX, WrapperId.LINUX_SH),
        (OsFamily.MACOS, WrapperId.MACOS_SH),
    }
)


class CoreProofError(Exception):
    def __init__(
        self,
        reason: str,
        *,
        phase: str | None = None,
        child_exit_code: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.phase = phase
        self.child_exit_code = child_exit_code


@dataclass
class ProofConfig:
    os_family: OsFamily
    wrapper_id: WrapperId
    phase: str
    run_id_prefix: str
    base_url: str
    kafka_ui: str
    mongo_express: str
    elasticsearch_url: str
    kibana_url: str
    sentry_url: str
    phase_timeout_seconds: int
    elasticsearch_index: str = _DEFAULT_ES_INDEX


@dataclass
class PhaseOutcome:
    name: str
    ok: bool
    receipt_id: str | None = None
    details: dict[str, str] = field(default_factory=dict)


def detect_os_family(system_name: str | None = None) -> OsFamily:
    name = (system_name if system_name is not None else platform.system()).strip()
    mapping = {
        "Windows": OsFamily.WINDOWS,
        "Linux": OsFamily.LINUX,
        "Darwin": OsFamily.MACOS,
    }
    detected = mapping.get(name)
    if detected is None:
        raise CoreProofError("unsupported_operating_system")
    return detected


def validate_os_wrapper_pair(
    os_family: OsFamily,
    wrapper_id: WrapperId,
    *,
    detected: OsFamily | None = None,
) -> None:
    if (os_family, wrapper_id) not in VALID_OS_WRAPPER_PAIRS:
        raise CoreProofError("invalid_os_wrapper_pair")
    actual = detected if detected is not None else detect_os_family()
    if actual is not os_family:
        raise CoreProofError("operating_system_mismatch")


def resolve_phases(phase: str) -> tuple[str, ...]:
    if phase == "all":
        return ALL_PHASE_ORDER
    if phase not in ALL_PHASE_ORDER:
        raise CoreProofError("unknown_phase")
    return (phase,)


def _env_default(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return fallback
    stripped = value.strip()
    return stripped if stripped else fallback


def _print_kv(key: str, value: object) -> None:
    print(f"{key}={value}", flush=True)


def _emit_phase_running(phase: str) -> None:
    _print_kv("core_phase", phase)
    _print_kv("core_phase_result", "RUNNING")


def _emit_phase_pass(phase: str) -> None:
    _print_kv("core_phase", phase)
    _print_kv("core_phase_result", "PASS")


def _emit_failure(
    phase: str,
    reason: str,
    *,
    child_exit_code: int | None = None,
) -> None:
    _print_kv("core_proof_result", "FAIL")
    _print_kv("failed_phase", phase)
    _print_kv("failure_reason", reason)
    if child_exit_code is not None:
        _print_kv("child_exit_code", child_exit_code)


def _which(command: str) -> str | None:
    return shutil.which(command)


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    completed = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise CoreProofError(
            "command_failed",
            child_exit_code=completed.returncode,
        )
    return completed


def http_get_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise CoreProofError("http_json_not_object")
    return parsed


def http_post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    body = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise CoreProofError("http_json_not_object")
    return parsed


def wait_for_http_reachable(
    url: str,
    *,
    timeout_seconds: int,
    accept_status: Callable[[int], bool] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    checker = accept_status or (lambda code: 200 <= code < 500)
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if checker(int(response.status)):
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise CoreProofError("http_endpoint_unreachable")


def wait_for_json_condition(
    url: str,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            payload = http_get_json(url, timeout=5.0)
            if predicate(payload):
                return payload
        except (
            CoreProofError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ):
            last_error = "json_condition_pending"
        time.sleep(2)
    raise CoreProofError(last_error or "json_condition_timeout")


def wait_for_lkw_health(base_url: str, *, timeout_seconds: int) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    wait_for_json_condition(
        health_url,
        lambda payload: str(payload.get("status", "")).strip() == "ok",
        timeout_seconds=timeout_seconds,
    )
    _print_kv("lkw_health", "ok")


def discover_compose_files() -> list[Path]:
    files = [_BASE_COMPOSE]
    extras = sorted(_DOCKER_DIR.glob("docker-compose.*.yml"))
    files.extend(extras)
    return files


def compose_args(compose_files: Sequence[Path]) -> list[str]:
    args = ["docker", "compose"]
    for path in compose_files:
        args.extend(["-f", str(path)])
    return args


def compose_config(compose_files: Sequence[Path], *, cwd: Path) -> None:
    completed = run_command(
        [*compose_args(compose_files), "config"],
        cwd=cwd,
        timeout=120,
    )
    if completed.returncode != 0:
        raise CoreProofError(
            "compose_config_failed",
            child_exit_code=completed.returncode,
        )


def compose_up(
    compose_files: Sequence[Path],
    services: Sequence[str],
    *,
    cwd: Path,
    build: bool = True,
) -> None:
    command = [*compose_args(compose_files), "up", "-d"]
    if build:
        command.append("--build")
    command.extend(services)
    completed = run_command(command, cwd=cwd, timeout=None)
    if completed.returncode != 0:
        raise CoreProofError(
            "compose_up_failed",
            child_exit_code=completed.returncode,
        )


def compose_stop(
    compose_files: Sequence[Path],
    services: Sequence[str],
    *,
    cwd: Path,
) -> None:
    completed = run_command(
        [*compose_args(compose_files), "stop", *services],
        cwd=cwd,
        timeout=300,
    )
    if completed.returncode != 0:
        raise CoreProofError(
            "compose_stop_failed",
            child_exit_code=completed.returncode,
        )


def compose_restart(
    compose_files: Sequence[Path],
    services: Sequence[str],
    *,
    cwd: Path,
) -> None:
    completed = run_command(
        [*compose_args(compose_files), "restart", *services],
        cwd=cwd,
        timeout=300,
    )
    if completed.returncode != 0:
        raise CoreProofError(
            "compose_restart_failed",
            child_exit_code=completed.returncode,
        )


def compose_down(
    compose_files: Sequence[Path],
    *,
    cwd: Path,
    volumes: bool = False,
    remove_orphans: bool = True,
) -> None:
    command = [*compose_args(compose_files), "down"]
    if volumes:
        command.append("-v")
    if remove_orphans:
        command.append("--remove-orphans")
    completed = run_command(command, cwd=cwd, timeout=None)
    if completed.returncode != 0:
        raise CoreProofError(
            "compose_down_failed",
            child_exit_code=completed.returncode,
        )


def compose_ps_json(
    compose_files: Sequence[Path],
    service: str,
    *,
    cwd: Path,
) -> list[dict[str, Any]]:
    completed = run_command(
        [*compose_args(compose_files), "ps", "--format", "json", service],
        cwd=cwd,
        timeout=60,
    )
    if completed.returncode != 0:
        raise CoreProofError(
            "compose_ps_failed",
            child_exit_code=completed.returncode,
        )
    text = completed.stdout.strip()
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    # Docker may emit one JSON object per line or a JSON array.
    if text.startswith("["):
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    rows.append(item)
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def wait_for_compose_health(
    compose_files: Sequence[Path],
    service: str,
    *,
    cwd: Path,
    timeout_seconds: int,
    require_running: bool = False,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            rows = compose_ps_json(compose_files, service, cwd=cwd)
        except CoreProofError:
            rows = []
        for row in rows:
            health = str(row.get("Health") or "").strip().lower()
            state = str(row.get("State") or "").strip().lower()
            status = str(row.get("Status") or "").strip().lower()
            running = state == "running" or status.startswith("up")
            if health == "healthy" and (running or not require_running):
                return
        time.sleep(2)
    raise CoreProofError("compose_health_timeout")


def ensure_env_file() -> None:
    env_file = _APP_DIR / ".env"
    example = _APP_DIR / ".env.example"
    if env_file.is_file():
        return
    if not example.is_file():
        raise CoreProofError("env_file_missing")
    shutil.copyfile(example, env_file)


def mongodb_child_env() -> dict[str, str]:
    username = _env_default("LKW_MONGODB_ROOT_USERNAME", "intergrax")
    password = _env_default("LKW_MONGODB_ROOT_PASSWORD", "intergrax-local-dev-only")
    database = _env_default("LKW_MONGODB_DATABASE", "intergrax_proofs")
    collection = _env_default("LKW_MONGODB_COLLECTION", "proof_receipts")
    host_port = _env_default("LKW_MONGODB_HOST_PORT", "27018")
    uri = (
        f"mongodb://{username}:{password}@127.0.0.1:{host_port}/"
        f"{database}?authSource=admin"
    )
    return {
        "INTERGRAX_MONGODB_URI": uri,
        "INTERGRAX_MONGODB_DATABASE": database,
        "INTERGRAX_MONGODB_COLLECTION": collection,
        "LKW_MONGODB_ROOT_USERNAME": username,
        "LKW_MONGODB_ROOT_PASSWORD": password,
        "LKW_MONGODB_DATABASE": database,
        "LKW_MONGODB_COLLECTION": collection,
        "LKW_MONGODB_HOST_PORT": host_port,
    }


def parse_kv_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _KV_LINE.match(line.strip())
        if match is None:
            continue
        values[match.group(1)] = match.group(2)
    return values


def run_python_child(
    script: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> tuple[int, str]:
    if not script.is_file():
        raise CoreProofError("proof_script_missing")
    completed = run_command(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        timeout=timeout,
    )
    combined = (completed.stdout or "") + (
        "\n" + completed.stderr if completed.stderr else ""
    )
    return completed.returncode, combined


def require_child_fields(
    output: Mapping[str, str],
    required: Mapping[str, str],
) -> None:
    for key, expected in required.items():
        actual = output.get(key)
        if actual is None:
            raise CoreProofError(f"missing_{key}")
        if actual != expected:
            raise CoreProofError(f"unexpected_{key}")


def extract_receipt_id(output: Mapping[str, str]) -> str:
    receipt_id = str(output.get("proof_receipt_id", "")).strip()
    if not receipt_id:
        raise CoreProofError("blank_proof_receipt_id")
    return receipt_id


def validate_background_task_child_output(output: Mapping[str, str]) -> str:
    require_child_fields(
        output,
        {
            "proof_result": "PASS",
            "proof_receipt_recorded": "true",
            "proof_receipt_verified": "true",
            "proof_receipt_query_verified": "true",
            "document_store_provider": "mongodb",
            "message_bus_provider": "kafka",
        },
    )
    return extract_receipt_id(output)


def validate_hosting_child_output(output: Mapping[str, str]) -> str:
    require_child_fields(
        output,
        {
            "proof_result": "PASS",
            "proof_kind": "platform_application_hosting",
            "proof_receipt_recorded": "true",
            "proof_receipt_verified": "true",
            "proof_receipt_query_verified": "true",
        },
    )
    return extract_receipt_id(output)


def validate_file_watcher_child_output(output: Mapping[str, str]) -> str:
    require_child_fields(
        output,
        {
            "proof_result": "PASS",
            "proof_kind": "file_watcher_persistent_search",
            "embedding_warmup_completed": "true",
            "reviewer_rerun_required": "false",
            "source_ref_found_before_restart": "true",
            "watcher_restored_after_restart": "true",
            "source_ref_found_after_restart": "true",
            "proof_receipt_recorded": "true",
            "proof_receipt_verified": "true",
            "proof_receipt_query_verified": "true",
        },
    )
    return extract_receipt_id(output)


def validate_sentry_child_output(output: Mapping[str, str]) -> None:
    require_child_fields(
        output,
        {
            "proof_result": "PASS",
            "safety_check": "passed",
        },
    )


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_index_signal_count(response: Mapping[str, Any]) -> int:
    evidence = _as_mapping(response.get("metadata")).get("lkw_evidence.v1")
    diagnostics = _as_mapping(_as_mapping(evidence).get("diagnostics"))
    index_summary = _as_mapping(diagnostics.get("lkw.index_summary.v1"))
    for field_name in ("ingested_count", "chunk_count", "accepted_count"):
        raw = index_summary.get(field_name)
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def extract_search_result_count(response: Mapping[str, Any]) -> int:
    evidence = _as_mapping(response.get("metadata")).get("lkw_evidence.v1")
    diagnostics = _as_mapping(_as_mapping(evidence).get("diagnostics"))
    search_summary = _as_mapping(diagnostics.get("lkw.search_summary.v1"))
    if not search_summary:
        return 0
    if search_summary.get("used") is not True:
        return 0
    reason = search_summary.get("reason")
    if not isinstance(reason, str):
        return 0
    if reason.strip() != _SEARCH_REASON_RETRIEVE_COMPLETE:
        return 0
    for field_name in ("evidence_count", "num_results"):
        raw = search_summary.get(field_name)
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def require_positive_ingest(response: Mapping[str, Any]) -> int:
    count = extract_index_signal_count(response)
    if count <= 0:
        raise CoreProofError("index_not_ingested")
    return count


def require_positive_search(response: Mapping[str, Any]) -> int:
    count = extract_search_result_count(response)
    if count <= 0:
        raise CoreProofError("search_results_missing")
    return count


def search_retrieve_ready(response: Mapping[str, Any]) -> bool:
    """True when typed search summary proves retrieve completed (zero hits OK)."""
    evidence = _as_mapping(response.get("metadata")).get("lkw_evidence.v1")
    diagnostics = _as_mapping(_as_mapping(evidence).get("diagnostics"))
    search_summary = _as_mapping(diagnostics.get("lkw.search_summary.v1"))
    if search_summary.get("used") is not True:
        return False
    reason = search_summary.get("reason")
    return (
        isinstance(reason, str) and reason.strip() == _SEARCH_REASON_RETRIEVE_COMPLETE
    )


def safe_failure_reason(output: Mapping[str, str], *, fallback: str) -> str:
    reason = str(output.get("failure_reason", "")).strip()
    if reason and _SAFE_REASON.fullmatch(reason):
        return reason
    return fallback


def ensure_file_watcher_retrieve_ready(config: ProofConfig) -> None:
    """Seed the watcher collection after volume reset so embedding warm-up can pass.

    After ``compose down -v``, the watcher collection is empty. Platform retrieve
    currently maps empty results to ``used=false`` / ``retrieve_failed`` /
    ``no_hits``, which makes the accepted file-watcher warm-up fail closed.
    Indexing one seed document restores a working retrieve path for warm-up.
    """
    _PROOF_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    seed_path = _PROOF_DOCS_DIR / _FILE_WATCHER_SEED_NAME
    seed_path.write_text(
        (
            "Intergrax LKW core proof embedding seed document.\n"
            "Pre-warms the file-watcher collection after a clean volume reset.\n"
        ),
        encoding="utf-8",
    )
    container_path = f"/data/user_docs/{_FILE_WATCHER_SEED_NAME}"
    run_url = f"{config.base_url.rstrip('/')}/v1/local_workspace/run"
    index_response = http_post_json(
        run_url,
        {
            "tenant_id": _FILE_WATCHER_SCOPE_ID,
            "workspace_id": _FILE_WATCHER_SCOPE_ID,
            "user_id": _FILE_WATCHER_USER_ID,
            "message": "index core proof embedding seed",
            "capability": "local.workspace.index",
            "metadata": {
                "source_paths": [container_path],
                "collection_id": _FILE_WATCHER_SCOPE_ID,
            },
        },
        timeout=300.0,
    )
    require_positive_ingest(index_response)
    deadline = time.monotonic() + min(300, config.phase_timeout_seconds)
    while time.monotonic() < deadline:
        try:
            probe = http_post_json(
                run_url,
                {
                    "tenant_id": _FILE_WATCHER_SCOPE_ID,
                    "workspace_id": _FILE_WATCHER_SCOPE_ID,
                    "user_id": _FILE_WATCHER_USER_ID,
                    "message": "core proof retrieve readiness probe",
                    "capability": "local.workspace.search",
                    "metadata": {
                        "tenant_id": _FILE_WATCHER_SCOPE_ID,
                        "user_id": _FILE_WATCHER_USER_ID,
                        "workspace_id": _FILE_WATCHER_SCOPE_ID,
                        "collection_id": _FILE_WATCHER_SCOPE_ID,
                        "query": "core proof embedding seed",
                        "top_k": 1,
                        "proof_phase": "embedding_warmup",
                    },
                },
                timeout=120.0,
            )
        except (
            CoreProofError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ):
            time.sleep(2)
            continue
        if search_retrieve_ready(probe):
            _print_kv("file_watcher_retrieve_ready", "true")
            return
        time.sleep(2)
    raise CoreProofError("file_watcher_retrieve_not_ready")


def validate_environment(config: ProofConfig) -> None:
    if _which("uv") is None:
        raise CoreProofError("uv_missing")
    if _which("docker") is None:
        raise CoreProofError("docker_missing")
    compose_probe = run_command(
        ["docker", "compose", "version"], cwd=_REPO_ROOT, timeout=30
    )
    if compose_probe.returncode != 0:
        raise CoreProofError(
            "docker_compose_unavailable",
            child_exit_code=compose_probe.returncode,
        )
    for path in (
        _BASE_COMPOSE,
        _ES_COMPOSE,
        _KAFKA_COMPOSE,
        _MONGODB_COMPOSE,
        _WATCHER_COMPOSE,
        _SENTRY_PROOF_PY,
        _ES_INSPECT_PY,
        _BACKGROUND_PROOF_PY,
        _HOSTING_PROOF_PY,
        _FILE_WATCHER_PROOF_PY,
    ):
        if not path.exists():
            raise CoreProofError("required_path_missing")
    ensure_env_file()
    _print_kv("environment_validation", "PASS")
    _print_kv("detected_os_family", detect_os_family().value)
    _print_kv("requested_os_family", config.os_family.value)
    _print_kv("wrapper_id", config.wrapper_id.value)


def clear_sentry_runtime_state() -> None:
    for name in ("generated.env", "generated.env.tmp", ".bootstrapped"):
        path = _SENTRY_PROOF_DIR / name
        if path.is_file():
            path.unlink()
    startup_log = _DOCKER_DIR / "lkw-platform-proof-startup.log"
    if startup_log.is_file():
        startup_log.unlink()


def phase_startup(config: ProofConfig) -> PhaseOutcome:
    compose_files = discover_compose_files()
    for path in compose_files:
        if not path.is_file():
            raise CoreProofError("required_path_missing")
    compose_config(compose_files, cwd=_REPO_ROOT)
    compose_down(compose_files, cwd=_REPO_ROOT, volumes=True, remove_orphans=True)
    clear_sentry_runtime_state()
    compose_up(compose_files, [], cwd=_REPO_ROOT, build=True)
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    return PhaseOutcome(name="startup", ok=True)


def phase_sentry(config: ProofConfig) -> PhaseOutcome:
    compose_files = discover_compose_files()
    compose_up(compose_files, [], cwd=_REPO_ROOT, build=False)
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    run_id = f"{config.run_id_prefix}sentry"
    correlation_id = run_id
    exit_code, text = run_python_child(
        _SENTRY_PROOF_PY,
        [
            "--base-url",
            config.base_url,
            "--sentry-ui",
            config.sentry_url,
            "--run-id",
            run_id,
            "--correlation-id",
            correlation_id,
        ],
        cwd=_REPO_ROOT,
        timeout=config.phase_timeout_seconds,
    )
    if exit_code != 0:
        raise CoreProofError(
            "sentry_child_failed",
            child_exit_code=exit_code,
        )
    validate_sentry_child_output(parse_kv_output(text))
    return PhaseOutcome(name="sentry", ok=True)


def phase_elasticsearch(config: ProofConfig) -> PhaseOutcome:
    compose_files = [_BASE_COMPOSE, _ES_COMPOSE]
    compose_config(compose_files, cwd=_REPO_ROOT)
    compose_up(compose_files, ["local_workspace"], cwd=_REPO_ROOT, build=True)
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    wait_for_http_reachable(
        f"{config.elasticsearch_url.rstrip('/')}/_cluster/health",
        timeout_seconds=min(180, config.phase_timeout_seconds),
    )
    run_payload = {
        "message": "Find documents about local workspace observability proof",
        "capability": "local.workspace.search",
        "metadata": {
            "proof": "LKW_PLATFORM_PROOF",
            "proof_helper": "run-lkw-core-platform-proof",
        },
    }
    response = http_post_json(
        f"{config.base_url.rstrip('/')}/v1/local_workspace/run",
        run_payload,
        timeout=180.0,
    )
    run_id = str(response.get("run_id") or "").strip()
    if not run_id:
        raise CoreProofError("run_id_missing")
    _print_kv("run_id", run_id)
    inspect_base = [
        "--url",
        config.elasticsearch_url,
        "--index",
        config.elasticsearch_index,
        "--run-id",
        run_id,
    ]
    for extra in (
        [],
        ["--check-duplicates"],
        ["--check-safety"],
        ["--check-duplicates", "--check-safety"],
    ):
        exit_code, _text = run_python_child(
            _ES_INSPECT_PY,
            [*inspect_base, *extra],
            cwd=_REPO_ROOT,
            timeout=config.phase_timeout_seconds,
        )
        if exit_code != 0:
            raise CoreProofError(
                "elasticsearch_validation_failed",
                child_exit_code=exit_code,
            )
    _print_kv("elasticsearch_validation", "passed")
    _print_kv("run_id", run_id)
    return PhaseOutcome(
        name="elasticsearch",
        ok=True,
        details={"run_id": run_id},
    )


def phase_persistence(config: ProofConfig) -> PhaseOutcome:
    compose_files = discover_compose_files()
    tenant_id = "lkw-persistence-proof"
    workspace_id = "lkw-persistence-proof"
    collection_id = "lkw-persistence-proof"
    marker_timestamp = time.strftime("%Y%m%d%H%M%S")
    marker = f"LKW_PERSISTENCE_PROOF_{marker_timestamp}"
    proof_file_name = f"lkw_persistence_proof_{marker_timestamp}.txt"
    container_source_path = f"/data/user_docs/{proof_file_name}"
    _SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    proof_doc_path = _SAMPLE_DOCS_DIR / proof_file_name
    proof_doc_path.write_text(
        (
            "Intergrax LKW persistence proof document.\n"
            f"Unique marker: {marker}\n"
            "This document verifies indexed local knowledge survives a "
            "non-destructive restart."
        ),
        encoding="utf-8",
    )
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    index_response = http_post_json(
        f"{config.base_url.rstrip('/')}/v1/local_workspace/run",
        {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "message": "index persistence proof document",
            "capability": "local.workspace.index",
            "metadata": {
                "source_paths": [container_source_path],
                "collection_id": collection_id,
            },
        },
        timeout=300.0,
    )
    index_signal = require_positive_ingest(index_response)
    _print_kv("index_signal_count", index_signal)

    def _search() -> dict[str, Any]:
        return http_post_json(
            f"{config.base_url.rstrip('/')}/v1/local_workspace/run",
            {
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "message": marker,
                "capability": "local.workspace.search",
                "metadata": {
                    "collection_id": collection_id,
                    "query": marker,
                    "top_k": 5,
                },
            },
            timeout=180.0,
        )

    before_count = require_positive_search(_search())
    _print_kv("before_restart_results", before_count)
    compose_restart(
        compose_files,
        ["local_workspace", "qdrant"],
        cwd=_REPO_ROOT,
    )
    _print_kv("restart_mode", "non_destructive")
    _print_kv("volumes_removed", "false")
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    after_count = require_positive_search(_search())
    _print_kv("after_restart_results", after_count)
    _print_kv("proof_kind", "persistent_vector_storage")
    _print_kv("reindexed_after_restart", "false")
    return PhaseOutcome(
        name="persistence",
        ok=True,
        details={
            "before_restart_results": str(before_count),
            "after_restart_results": str(after_count),
        },
    )


def phase_background_task(config: ProofConfig) -> PhaseOutcome:
    compose_files = [_BASE_COMPOSE, _KAFKA_COMPOSE, _MONGODB_COMPOSE]
    compose_config(compose_files, cwd=_REPO_ROOT)
    _PROOF_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    compose_up(
        compose_files,
        [
            "local_workspace",
            "lkw-background-worker",
            "lkw-kafka",
            "lkw-kafka-topics",
            "lkw-kafka-ui",
            "lkw-redis",
            "lkw-mongodb",
            "lkw-mongo-express",
        ],
        cwd=_REPO_ROOT,
        build=True,
    )
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    wait_for_compose_health(
        compose_files,
        "lkw-mongodb",
        cwd=_REPO_ROOT,
        timeout_seconds=min(180, config.phase_timeout_seconds),
    )
    wait_for_http_reachable(config.kafka_ui, timeout_seconds=120)
    wait_for_http_reachable(config.mongo_express, timeout_seconds=120)
    exit_code, text = run_python_child(
        _BACKGROUND_PROOF_PY,
        [
            "--base-url",
            config.base_url,
            "--kafka-ui",
            config.kafka_ui,
            "--mongo-express",
            config.mongo_express,
        ],
        cwd=_REPO_ROOT,
        env=mongodb_child_env(),
        timeout=config.phase_timeout_seconds,
    )
    if exit_code != 0:
        raise CoreProofError(
            "background_task_child_failed",
            child_exit_code=exit_code,
        )
    receipt_id = validate_background_task_child_output(parse_kv_output(text))
    return PhaseOutcome(
        name="background-task",
        ok=True,
        receipt_id=receipt_id,
    )


def phase_application_hosting(config: ProofConfig) -> PhaseOutcome:
    compose_files = [_BASE_COMPOSE, _MONGODB_COMPOSE]
    compose_config(compose_files, cwd=_REPO_ROOT)
    compose_up(
        compose_files,
        ["lkw-mongodb", "lkw-mongo-express"],
        cwd=_REPO_ROOT,
        build=False,
    )
    wait_for_compose_health(
        compose_files,
        "lkw-mongodb",
        cwd=_REPO_ROOT,
        timeout_seconds=min(180, config.phase_timeout_seconds),
    )
    wait_for_http_reachable(config.mongo_express, timeout_seconds=120)
    exit_code, text = run_python_child(
        _HOSTING_PROOF_PY,
        [],
        cwd=_REPO_ROOT,
        env=mongodb_child_env(),
        timeout=config.phase_timeout_seconds,
    )
    if exit_code != 0:
        raise CoreProofError(
            "application_hosting_child_failed",
            child_exit_code=exit_code,
        )
    receipt_id = validate_hosting_child_output(parse_kv_output(text))
    return PhaseOutcome(
        name="application-hosting",
        ok=True,
        receipt_id=receipt_id,
    )


def phase_file_watcher(config: ProofConfig) -> PhaseOutcome:
    compose_files = [
        _BASE_COMPOSE,
        _KAFKA_COMPOSE,
        _WATCHER_COMPOSE,
        _MONGODB_COMPOSE,
    ]
    compose_config(compose_files, cwd=_REPO_ROOT)
    _PROOF_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if _WATCHER_STATE_DIR.exists():
        shutil.rmtree(_WATCHER_STATE_DIR)
    _WATCHER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _print_kv("watcher_state_reset", "true")
    compose_up(
        compose_files,
        [
            "local_workspace",
            "lkw-background-worker",
            "lkw-file-watcher",
            "lkw-kafka",
            "lkw-kafka-topics",
            "lkw-kafka-ui",
            "lkw-redis",
            "qdrant",
            "ollama",
            "lkw-mongodb",
            "lkw-mongo-express",
        ],
        cwd=_REPO_ROOT,
        build=True,
    )
    wait_for_lkw_health(config.base_url, timeout_seconds=config.phase_timeout_seconds)
    wait_for_compose_health(
        compose_files,
        "lkw-file-watcher",
        cwd=_REPO_ROOT,
        timeout_seconds=min(240, config.phase_timeout_seconds),
        require_running=True,
    )
    wait_for_compose_health(
        compose_files,
        "lkw-mongodb",
        cwd=_REPO_ROOT,
        timeout_seconds=min(180, config.phase_timeout_seconds),
    )
    wait_for_http_reachable(config.mongo_express, timeout_seconds=120)
    wait_for_http_reachable(config.kafka_ui, timeout_seconds=120)
    ensure_file_watcher_retrieve_ready(config)
    kafka_bootstrap = _env_default(
        "LKW_FILE_WATCHER_E2E_KAFKA_BOOTSTRAP",
        "127.0.0.1:9094",
    )
    exit_code, text = run_python_child(
        _FILE_WATCHER_PROOF_PY,
        [
            "--base-url",
            config.base_url,
            "--kafka-bootstrap",
            kafka_bootstrap,
            "--topic",
            "intergrax.tasks",
            "--repo-root",
            str(_REPO_ROOT),
            "--proof-docs-dir",
            str(_PROOF_DOCS_DIR),
            "--base-compose",
            str(_BASE_COMPOSE),
            "--kafka-compose",
            str(_KAFKA_COMPOSE),
            "--watcher-compose",
            str(_WATCHER_COMPOSE),
            "--mongodb-compose",
            str(_MONGODB_COMPOSE),
            "--mongo-express",
            config.mongo_express,
        ],
        cwd=_REPO_ROOT,
        env=mongodb_child_env(),
        timeout=None,
    )
    parsed_child = parse_kv_output(text)
    if exit_code != 0:
        raise CoreProofError(
            safe_failure_reason(
                parsed_child,
                fallback="file_watcher_child_failed",
            ),
            child_exit_code=exit_code,
        )
    receipt_id = validate_file_watcher_child_output(parsed_child)
    return PhaseOutcome(
        name="file-watcher",
        ok=True,
        receipt_id=receipt_id,
    )


PHASE_RUNNERS: dict[str, Callable[[ProofConfig], PhaseOutcome]] = {
    "startup": phase_startup,
    "sentry": phase_sentry,
    "elasticsearch": phase_elasticsearch,
    "persistence": phase_persistence,
    "background-task": phase_background_task,
    "application-hosting": phase_application_hosting,
    "file-watcher": phase_file_watcher,
}


def emit_final_pass(
    config: ProofConfig,
    *,
    background_task_receipt_id: str,
    application_hosting_receipt_id: str,
    file_watcher_receipt_id: str,
) -> None:
    _print_kv("core_proof_result", "PASS")
    _print_kv("core_proof_os_family", config.os_family.value)
    _print_kv("core_proof_wrapper_id", config.wrapper_id.value)
    _print_kv("core_proof_shared_python_runner", "true")
    _print_kv("core_proof_all_phases_passed", "true")
    _print_kv("startup_phase", "PASS")
    _print_kv("sentry_phase", "PASS")
    _print_kv("elasticsearch_phase", "PASS")
    _print_kv("persistence_phase", "PASS")
    _print_kv("background_task_phase", "PASS")
    _print_kv("application_hosting_phase", "PASS")
    _print_kv("file_watcher_phase", "PASS")
    _print_kv("individual_proof_receipts_authoritative", "true")
    _print_kv("aggregate_terminal_summary_authoritative", "false")
    _print_kv("optional_os_interaction_proof_executed", "false")
    _print_kv("background_task_proof_receipt_id", background_task_receipt_id)
    _print_kv(
        "application_hosting_proof_receipt_id",
        application_hosting_receipt_id,
    )
    _print_kv("file_watcher_proof_receipt_id", file_watcher_receipt_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared LKW Core Platform Proof runner.",
    )
    parser.add_argument(
        "--os-family",
        required=True,
        choices=[item.value for item in OsFamily],
    )
    parser.add_argument(
        "--wrapper-id",
        required=True,
        choices=[item.value for item in WrapperId],
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", *ALL_PHASE_ORDER],
    )
    parser.add_argument(
        "--run-id-prefix",
        default=_env_default("LKW_CORE_PROOF_RUN_ID_PREFIX", "lkw-core-"),
    )
    parser.add_argument(
        "--base-url",
        default=_env_default("LOCAL_WORKSPACE_BACKEND_BASE_URL", _DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--kafka-ui",
        default=_env_default(
            "LKW_BACKGROUND_TASK_PROOF_KAFKA_UI_URL", _DEFAULT_KAFKA_UI
        ),
    )
    parser.add_argument(
        "--mongo-express",
        default=_env_default("LKW_MONGO_EXPRESS_URL", _DEFAULT_MONGO_EXPRESS),
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=_env_default(
            "LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_URL",
            _DEFAULT_ELASTICSEARCH_URL,
        ),
    )
    parser.add_argument(
        "--kibana-url",
        default=_env_default(
            "LOCAL_WORKSPACE_OBSERVABILITY_PROOF_KIBANA_URL",
            _DEFAULT_KIBANA_URL,
        ),
    )
    parser.add_argument(
        "--sentry-url",
        default=_env_default("LKW_SENTRY_PROOF_UI_URL", _DEFAULT_SENTRY_URL),
    )
    parser.add_argument(
        "--phase-timeout-seconds",
        type=int,
        default=int(
            _env_default(
                "LKW_CORE_PROOF_PHASE_TIMEOUT_SECONDS",
                str(_DEFAULT_PHASE_TIMEOUT),
            )
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ProofConfig:
    return ProofConfig(
        os_family=OsFamily(args.os_family),
        wrapper_id=WrapperId(args.wrapper_id),
        phase=str(args.phase),
        run_id_prefix=str(args.run_id_prefix),
        base_url=str(args.base_url).rstrip("/"),
        kafka_ui=str(args.kafka_ui).rstrip("/"),
        mongo_express=str(args.mongo_express).rstrip("/"),
        elasticsearch_url=str(args.elasticsearch_url).rstrip("/"),
        kibana_url=str(args.kibana_url).rstrip("/"),
        sentry_url=str(args.sentry_url).rstrip("/"),
        phase_timeout_seconds=int(args.phase_timeout_seconds),
    )


def run_core_proof(
    config: ProofConfig,
    *,
    phase_runners: Mapping[str, Callable[[ProofConfig], PhaseOutcome]] | None = None,
) -> int:
    runners = dict(PHASE_RUNNERS if phase_runners is None else phase_runners)
    try:
        validate_os_wrapper_pair(config.os_family, config.wrapper_id)
        validate_environment(config)
        phases = resolve_phases(config.phase)
    except CoreProofError as exc:
        phase = exc.phase or "startup"
        _emit_failure(phase, exc.reason, child_exit_code=exc.child_exit_code)
        return 1

    outcomes: dict[str, PhaseOutcome] = {}
    for phase in phases:
        _emit_phase_running(phase)
        runner = runners.get(phase)
        if runner is None:
            _emit_failure(phase, "phase_runner_missing")
            return 1
        try:
            outcome = runner(config)
        except CoreProofError as exc:
            _emit_failure(
                phase,
                exc.reason,
                child_exit_code=exc.child_exit_code,
            )
            return 1
        except Exception as exc:  # noqa: BLE001 - fail closed with safe type only
            _emit_failure(phase, type(exc).__name__)
            return 1
        if not outcome.ok:
            _emit_failure(phase, "phase_reported_failure")
            return 1
        _emit_phase_pass(phase)
        outcomes[phase] = outcome

    if config.phase != "all":
        _print_kv("core_proof_result", "PARTIAL")
        _print_kv("core_proof_all_phases_passed", "false")
        _print_kv("optional_os_interaction_proof_executed", "false")
        return 0

    background_receipt = outcomes["background-task"].receipt_id
    hosting_receipt = outcomes["application-hosting"].receipt_id
    watcher_receipt = outcomes["file-watcher"].receipt_id
    if not background_receipt or not hosting_receipt or not watcher_receipt:
        _emit_failure("file-watcher", "receipt_ids_incomplete")
        return 1
    emit_final_pass(
        config,
        background_task_receipt_id=background_receipt,
        application_hosting_receipt_id=hosting_receipt,
        file_watcher_receipt_id=watcher_receipt,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)
    return run_core_proof(config)


if __name__ == "__main__":
    sys.exit(main())
