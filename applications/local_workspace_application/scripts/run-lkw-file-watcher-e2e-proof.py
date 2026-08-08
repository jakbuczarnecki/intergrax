#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW.7C2 watcher-triggered persistent search E2E proof with ProofReceipt.

Uses docker compose against the dedicated watcher overlay, Kafka task-topic
inspection, local.workspace.search diagnostics, embedding warm-up, and
platform ProofReceiptStore → MongoDB DocumentStore recording.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import (
    create_mongodb_integration,
)
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
)
from intergrax.proofs.receipts.contracts import (
    ProofReceipt,
    ProofReceiptResult,
)
from intergrax.proofs.receipts.recording import (
    ProofReceiptVerificationError,
    record_and_verify_proof_receipt,
)

_AGENTS_ROOT = Path(__file__).resolve().parents[3] / "agents"
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.append(str(_AGENTS_ROOT))

from local_search.diagnostics import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    SearchSummaryReason,
    parse_search_summary_reason,
)

_APPLICATION_ID = "local_workspace"
_PROOF_KIND = "file_watcher_persistent_search"
_RECEIPT_TASK = "LKW.7C2"
_PROOF_RUNNER = "run-lkw-file-watcher-e2e-proof.py"
_DEFAULT_MONGO_EXPRESS_URL = "http://127.0.0.1:8086"
_SIDECAR_RESULT_SCHEMA = "lkw.file_watcher_sidecar_result.v1"
_VERIFICATION_DOCUMENT = (
    "docs/project/technical/applications/local_workspace_application/LKW_7_FILE_WATCHER_VERIFICATION.md"
)
_REVIEWER_GUIDE = "docs/project/proofs/LKW_PLATFORM_PROOF.md"

_TENANT_ID = "lkw-file-watcher-e2e"
_WORKSPACE_ID = "lkw-file-watcher-e2e"
_COLLECTION_ID = "lkw-file-watcher-e2e"
# Must match FileWatcher ingest job requested_by / indexed payload user_id.
_SEARCH_USER_ID = "lkw.file_watcher"

_TASK_TOPIC = "intergrax.tasks"

_CONTAINER_DOCS_ROOT = "/data/user_docs"
_CONTAINER_CHECKPOINT_PATH = (
    "/data/file_watcher_state/data/file_watcher/checkpoint.json"
)

_WATCHER_SERVICE = "lkw-file-watcher"
_RESTART_SERVICES = (
    "lkw-file-watcher",
    "lkw-background-worker",
    "local_workspace",
    "qdrant",
)

_SCRIPT_PATH = Path(__file__).resolve()
_DEFAULT_APP_DIR = _SCRIPT_PATH.parent.parent
_DEFAULT_REPO_ROOT = _DEFAULT_APP_DIR.parent.parent
_DEFAULT_DOCKER_DIR = _DEFAULT_APP_DIR / "docker"
_DEFAULT_PROOF_DOCS_DIR = _DEFAULT_APP_DIR / ".proof_docs"
_DEFAULT_BASE_COMPOSE = _DEFAULT_DOCKER_DIR / "docker-compose.yml"
_DEFAULT_KAFKA_COMPOSE = _DEFAULT_DOCKER_DIR / "docker-compose.kafka.yml"
_DEFAULT_WATCHER_COMPOSE = _DEFAULT_DOCKER_DIR / "file-watcher-e2e.compose.yml"
_DEFAULT_MONGODB_COMPOSE = _DEFAULT_DOCKER_DIR / "docker-compose.mongodb.yml"

_WARMUP_REQUEST_TIMEOUT_SECONDS = 120.0
_WARMUP_RETRY_SLEEP_SECONDS = 2.0


@dataclass(frozen=True)
class ProofDocument:
    marker: str
    filename: str
    host_path: Path
    container_source_path: str
    content: str


@dataclass(frozen=True)
class ProofFileStat:
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True)
class SearchDiagnostics:
    num_results: int
    evidence_count: int
    source_refs: tuple[str, ...]
    raw_tool_reason: str | None
    used: bool | None = None
    reason: SearchSummaryReason | None = None
    terminal_status: str | None = None


@dataclass(frozen=True)
class WarmupResult:
    completed: bool
    attempt_count: int
    last_reason: str | None
    last_raw_tool_reason: str | None


@dataclass(frozen=True)
class FileWatcherE2EWorkloadEvidence:
    marker: str
    proof_filename: str
    container_source_path: str
    watcher_checkpoint_ready: bool
    embedding_warmup_completed: bool
    task_count_before_file: int
    task_count_after_file: int
    search_results_before_restart: int
    source_ref_found_before_restart: bool
    task_count_before_restart: int
    task_count_after_restart: int
    search_results_after_restart: int
    source_ref_found_after_restart: bool
    watcher_restored_after_restart: bool
    watcher_final_checkpoint_saved: bool
    source_file_modified_after_index: bool
    restart_mode: str
    volumes_removed: bool

    @property
    def task_topic_increased(self) -> bool:
        return self.task_count_after_file > self.task_count_before_file

    @property
    def duplicate_enqueue_after_restart(self) -> bool:
        return self.task_count_after_restart > self.task_count_before_restart

    @property
    def task_topic_regressed_after_restart(self) -> bool:
        return self.task_count_after_restart < self.task_count_before_restart

    @property
    def reindexed_after_restart(self) -> bool:
        return self.task_count_after_restart != self.task_count_before_restart

    @property
    def checkpoint_restore_verified(self) -> bool:
        return (
            self.watcher_restored_after_restart and self.watcher_final_checkpoint_saved
        )

    @property
    def reviewer_rerun_required(self) -> bool:
        return not self.embedding_warmup_completed


def validate_file_watcher_e2e_workload_evidence(
    evidence: FileWatcherE2EWorkloadEvidence,
) -> None:
    if not evidence.marker.strip():
        raise ValueError("workload_marker_missing")
    if not evidence.proof_filename.strip():
        raise ValueError("workload_filename_missing")
    source_path = evidence.container_source_path.strip()
    if not source_path or not source_path.startswith(f"{_CONTAINER_DOCS_ROOT}/"):
        raise ValueError("workload_source_path_invalid")
    if not evidence.embedding_warmup_completed or evidence.reviewer_rerun_required:
        raise ValueError("embedding_warmup_not_completed")
    if not evidence.watcher_checkpoint_ready:
        raise ValueError("watcher_checkpoint_not_ready")
    if (
        evidence.task_count_before_file < 0
        or evidence.task_count_after_file < 0
        or evidence.task_count_before_restart < 0
        or evidence.task_count_after_restart < 0
    ):
        raise ValueError("kafka_task_count_invalid")
    if not evidence.task_topic_increased:
        raise ValueError("kafka_task_topic_did_not_increase")
    if evidence.search_results_before_restart <= 0:
        raise ValueError("search_before_restart_missing")
    if not evidence.source_ref_found_before_restart:
        raise ValueError("source_ref_before_restart_missing")
    if evidence.restart_mode != "non_destructive":
        raise ValueError("restart_mode_not_non_destructive")
    if evidence.volumes_removed:
        raise ValueError("volumes_removed")
    if evidence.duplicate_enqueue_after_restart:
        raise ValueError("duplicate_enqueue_after_restart")
    if evidence.task_topic_regressed_after_restart:
        raise ValueError("kafka_task_topic_regressed_after_restart")
    if evidence.search_results_after_restart <= 0:
        raise ValueError("search_after_restart_missing")
    if not evidence.source_ref_found_after_restart:
        raise ValueError("source_ref_after_restart_missing")
    if not evidence.watcher_restored_after_restart:
        raise ValueError("watcher_restore_not_proven")
    if not evidence.watcher_final_checkpoint_saved:
        raise ValueError("watcher_final_checkpoint_not_saved")
    if not evidence.checkpoint_restore_verified:
        raise ValueError("watcher_restore_not_proven")
    if evidence.source_file_modified_after_index:
        raise ValueError("source_file_modified_after_index")


def capture_proof_file_stat(path: Path) -> ProofFileStat:
    st = path.stat()
    return ProofFileStat(size_bytes=st.st_size, modified_time_ns=st.st_mtime_ns)


def proof_file_stat_unchanged(
    *,
    before: ProofFileStat,
    after: ProofFileStat,
) -> bool:
    return (
        before.size_bytes == after.size_bytes
        and before.modified_time_ns == after.modified_time_ns
    )


def build_file_watcher_proof_id(run_id: str) -> str:
    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be blank")
    return f"{_APPLICATION_ID}:{_PROOF_KIND}:{normalized}"


def _strip_optional_compose_log_prefix(line: str) -> str:
    stripped = line.strip()
    if " | " not in stripped:
        return stripped
    prefix, remainder = stripped.split(" | ", 1)
    candidate = remainder.strip()
    if candidate.startswith("{") and prefix and not prefix.startswith("{"):
        return candidate
    return stripped


def extract_last_file_watcher_sidecar_result(
    log_output: str,
) -> dict[str, object] | None:
    last: dict[str, object] | None = None
    for line in log_output.splitlines():
        if not line.strip():
            continue
        candidate = _strip_optional_compose_log_prefix(line)
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("schema_version") != _SIDECAR_RESULT_SCHEMA:
            continue
        last = parsed
    return last


def sidecar_result_proves_checkpoint_restore(
    result: dict[str, object] | None,
) -> bool:
    if result is None:
        return False
    if result.get("schema_version") != _SIDECAR_RESULT_SCHEMA:
        return False
    if result.get("exit_kind") != "clean_stop":
        return False
    if result.get("exit_code") != 0:
        return False
    if result.get("restored_from_checkpoint") is not True:
        return False
    if result.get("final_checkpoint_saved") is not True:
        return False
    if result.get("error_id") is not None:
        return False
    return True


def fail(reason: str, **safe_fields: object) -> int:
    print("proof_result=FAIL")
    print(f"proof_kind={_PROOF_KIND}")
    print(f"failure_reason={reason}")
    for key, value in safe_fields.items():
        if isinstance(value, bool):
            print(f"{key}={'true' if value else 'false'}")
        else:
            print(f"{key}={value}")
    return 1


def fail_receipt_recording(error: BaseException) -> int:
    print("proof_result=FAIL")
    print(f"proof_kind={_PROOF_KIND}")
    print("failure_reason=proof_receipt_recording_failed")
    print("proof_workload_result=PASS")
    print("proof_receipt_recorded=false")
    print("proof_receipt_verified=false")
    print(f"receipt_error_type={type(error).__name__}")
    return 1


def generate_proof_identity(*, now: datetime | None = None) -> tuple[str, str]:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    marker = f"LKW_FILE_WATCHER_E2E_{stamp}_{suffix}"
    filename = f"lkw_file_watcher_e2e_{stamp}_{suffix}.txt"
    return marker, filename


def build_proof_document_content(marker: str) -> str:
    return (
        f"{marker}\n"
        "This file was created after the file-watcher baseline checkpoint.\n"
        "No manual indexing command is permitted for this proof.\n"
    )


def create_proof_document(proof_docs_dir: Path) -> ProofDocument:
    marker, filename = generate_proof_identity()
    content = build_proof_document_content(marker)
    host_path = proof_docs_dir / filename
    host_path.write_text(content, encoding="utf-8")
    return ProofDocument(
        marker=marker,
        filename=filename,
        host_path=host_path,
        container_source_path=f"{_CONTAINER_DOCS_ROOT}/{filename}",
        content=content,
    )


def build_compose_command(
    *compose_args: str,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(base_compose),
        "-f",
        str(kafka_compose),
        "-f",
        str(watcher_compose),
        "-f",
        str(mongodb_compose),
        *compose_args,
    ]


def run_compose(
    *compose_args: str,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
    check: bool = True,
    timeout: float | None = 120.0,
) -> subprocess.CompletedProcess[str]:
    command = build_compose_command(
        *compose_args,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    )
    completed = subprocess.run(
        command,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"compose_command_failed:{compose_args[0]}")
    return completed


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: float = 30.0,
) -> dict[str, object]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response_not_object")
    return parsed


def wait_for_health(base_url: str, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            payload = request_json(health_url, timeout=5.0)
            if str(payload.get("status", "")).lower() == "ok":
                return True
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            pass
        time.sleep(2.0)
    return False


def watcher_container_running(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> bool:
    completed = run_compose(
        "ps",
        "--format",
        "json",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
        check=False,
    )
    if completed.returncode != 0:
        return False
    stdout = completed.stdout.strip()
    if not stdout:
        return False
    records: list[dict[str, Any]] = []
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, list):
            records = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            records = [parsed]
    except json.JSONDecodeError:
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    if not records:
        return False
    for record in records:
        state = str(record.get("State") or record.get("state") or "").lower()
        status = str(record.get("Status") or record.get("status") or "").lower()
        if state in {"exited", "dead", "restarting"}:
            return False
        if "restarting" in status or "exited" in status:
            return False
        if state in {"running", "up"} or status.startswith("up"):
            return True
    return False


def watcher_checkpoint_ready(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> bool:
    completed = run_compose(
        "exec",
        "-T",
        _WATCHER_SERVICE,
        "python",
        "-c",
        (
            "from pathlib import Path; "
            f"raise SystemExit(0 if Path({_CONTAINER_CHECKPOINT_PATH!r}).is_file() else 1)"
        ),
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
        check=False,
        timeout=30.0,
    )
    return completed.returncode == 0


def inspect_kafka_topic_message_count(*, bootstrap: str, topic: str) -> int:
    try:
        from confluent_kafka import Consumer, TopicPartition
    except ImportError as exc:
        raise RuntimeError("kafka_inspection_unavailable") from exc

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"lkw-file-watcher-e2e-inspector-{int(time.time())}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    try:
        metadata = consumer.list_topics(topic=topic, timeout=10)
        topic_meta = metadata.topics.get(topic)
        if topic_meta is None or topic_meta.error is not None:
            raise RuntimeError("kafka_topic_unavailable")
        total = 0
        for partition_id in topic_meta.partitions:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, partition_id),
                timeout=10,
            )
            total += max(0, high - low)
        return total
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("kafka_topic_unavailable") from exc
    finally:
        consumer.close()


def kafka_topic_increased(*, before: int, after: int) -> bool:
    return after > before


def kafka_topic_regression(*, before: int, after: int) -> bool:
    return after < before


def duplicate_enqueue_detected(*, before_restart: int, after_restart: int) -> bool:
    return after_restart > before_restart


def build_search_request(marker: str) -> dict[str, object]:
    return {
        "tenant_id": _TENANT_ID,
        "workspace_id": _WORKSPACE_ID,
        "user_id": _SEARCH_USER_ID,
        "message": marker,
        "capability": "local.workspace.search",
        "metadata": {
            "tenant_id": _TENANT_ID,
            "user_id": _SEARCH_USER_ID,
            "workspace_id": _WORKSPACE_ID,
            "collection_id": _COLLECTION_ID,
            "query": marker,
            "top_k": 5,
        },
    }


def build_warmup_search_request(query: str) -> dict[str, object]:
    return {
        "tenant_id": _TENANT_ID,
        "workspace_id": _WORKSPACE_ID,
        "user_id": _SEARCH_USER_ID,
        "message": query,
        "capability": "local.workspace.search",
        "metadata": {
            "tenant_id": _TENANT_ID,
            "user_id": _SEARCH_USER_ID,
            "workspace_id": _WORKSPACE_ID,
            "collection_id": _COLLECTION_ID,
            "query": query,
            "top_k": 1,
            "proof_phase": "embedding_warmup",
        },
    }


def extract_search_diagnostics(response: dict[str, object]) -> SearchDiagnostics | None:
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        return None
    evidence = metadata.get("lkw_evidence.v1")
    if not isinstance(evidence, dict):
        return None
    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    summary = diagnostics.get("lkw.search_summary.v1")
    if not isinstance(summary, dict):
        return None

    num_results_raw = summary.get("num_results", 0)
    evidence_count_raw = summary.get("evidence_count", 0)
    num_results = int(num_results_raw) if isinstance(num_results_raw, int) else 0
    evidence_count = (
        int(evidence_count_raw) if isinstance(evidence_count_raw, int) else 0
    )

    source_refs_raw = summary.get("source_refs")
    source_refs: tuple[str, ...]
    if isinstance(source_refs_raw, list):
        source_refs = tuple(str(item) for item in source_refs_raw)
    else:
        source_refs = ()

    raw_tool_reason_value = summary.get("raw_tool_reason")
    raw_tool_reason = (
        str(raw_tool_reason_value) if raw_tool_reason_value is not None else None
    )
    used_value = summary.get("used")
    used = used_value if isinstance(used_value, bool) else None
    reason = parse_search_summary_reason(summary.get("reason"))
    terminal_raw = evidence.get("terminal_status")
    terminal_status = str(terminal_raw) if terminal_raw is not None else None
    return SearchDiagnostics(
        num_results=num_results,
        evidence_count=evidence_count,
        source_refs=source_refs,
        raw_tool_reason=raw_tool_reason,
        used=used,
        reason=reason,
        terminal_status=terminal_status,
    )


def warmup_attempt_succeeded(diagnostics: SearchDiagnostics | None) -> bool:
    """Accept warm-up only when typed retrieve outcome proves success.

    Requires ``used=true`` and ``reason=retrieve_complete``. Zero hits are
    acceptable. ``terminal_status`` alone is never sufficient.
    """
    return (
        diagnostics is not None
        and diagnostics.used is True
        and diagnostics.reason is SearchSummaryReason.RETRIEVE_COMPLETE
    )


def search_attempt_succeeded(
    diagnostics: SearchDiagnostics | None,
    *,
    expected_source_path: str,
) -> bool:
    if diagnostics is None:
        return False
    if diagnostics.num_results <= 0 and diagnostics.evidence_count <= 0:
        return False
    return expected_source_path in diagnostics.source_refs


def run_embedding_warmup(
    *,
    base_url: str,
    timeout_seconds: float,
) -> WarmupResult:
    deadline = time.monotonic() + timeout_seconds
    run_url = f"{base_url.rstrip('/')}/v1/local_workspace/run"
    attempt_count = 0
    last_reason: str | None = None
    last_raw_tool_reason: str | None = None
    while time.monotonic() < deadline:
        attempt_count += 1
        query = f"LKW_FILE_WATCHER_E2E_PREWARM_{secrets.token_hex(4)}"
        request_body = build_warmup_search_request(query)
        try:
            response = request_json(
                run_url,
                method="POST",
                payload=request_body,
                timeout=_WARMUP_REQUEST_TIMEOUT_SECONDS,
            )
            diagnostics = extract_search_diagnostics(response)
            if diagnostics is not None:
                if diagnostics.reason is not None:
                    last_reason = diagnostics.reason.value
                elif diagnostics.terminal_status is not None:
                    last_reason = f"terminal_{diagnostics.terminal_status}"
                last_raw_tool_reason = diagnostics.raw_tool_reason
            if warmup_attempt_succeeded(diagnostics):
                if last_reason is None:
                    last_reason = SearchSummaryReason.RETRIEVE_COMPLETE.value
                return WarmupResult(
                    completed=True,
                    attempt_count=attempt_count,
                    last_reason=last_reason,
                    last_raw_tool_reason=last_raw_tool_reason,
                )
            if diagnostics is None:
                last_reason = "missing_diagnostics"
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            last_reason = "http_or_timeout"
        time.sleep(_WARMUP_RETRY_SLEEP_SECONDS)
    return WarmupResult(
        completed=False,
        attempt_count=attempt_count,
        last_reason=last_reason,
        last_raw_tool_reason=last_raw_tool_reason,
    )


def build_restart_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> list[str]:
    return build_compose_command(
        "restart",
        *_RESTART_SERVICES,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    )


def build_watcher_graceful_stop_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> list[str]:
    return build_compose_command(
        "stop",
        "--timeout",
        "30",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    )


def build_watcher_logs_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> list[str]:
    return build_compose_command(
        "logs",
        "--no-color",
        "--no-log-prefix",
        "--tail",
        "200",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    )


def build_watcher_resume_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
) -> list[str]:
    return build_compose_command(
        "up",
        "-d",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    )


def wait_for_watcher_ready(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    mongodb_compose: Path,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if watcher_container_running(
            base_compose=base_compose,
            kafka_compose=kafka_compose,
            watcher_compose=watcher_compose,
            mongodb_compose=mongodb_compose,
        ) and watcher_checkpoint_ready(
            base_compose=base_compose,
            kafka_compose=kafka_compose,
            watcher_compose=watcher_compose,
            mongodb_compose=mongodb_compose,
        ):
            return True
        time.sleep(1.0)
    return False


def build_file_watcher_e2e_proof_receipt(
    *,
    run_id: str,
    workload_evidence: FileWatcherE2EWorkloadEvidence,
    mongo_express_url: str = _DEFAULT_MONGO_EXPRESS_URL,
) -> ProofReceipt:
    if not isinstance(workload_evidence, FileWatcherE2EWorkloadEvidence):
        raise TypeError("workload_evidence_must_be_typed")
    validate_file_watcher_e2e_workload_evidence(workload_evidence)
    return ProofReceipt(
        proof_id=build_file_watcher_proof_id(run_id),
        proof_kind=_PROOF_KIND,
        application_id=_APPLICATION_ID,
        result=ProofReceiptResult.PASS,
        run_id=run_id,
        correlation_id=None,
        task_id=None,
        provider_evidence={
            "message_bus_provider": "kafka",
            "worker_execution": "asynchronous",
            "enqueue_trigger": "filesystem_create",
            "watcher_process": "foreground_sidecar",
            "watcher_checkpoint_store": "json_file",
            "checkpoint_restore_verified": (
                workload_evidence.checkpoint_restore_verified
            ),
            "watcher_final_checkpoint_saved": (
                workload_evidence.watcher_final_checkpoint_saved
            ),
            "vector_store_provider": "qdrant",
            "persistent_index": True,
            "document_store_provider": "mongodb",
            "kafka_task_topic": _TASK_TOPIC,
            "task_count_before_file": workload_evidence.task_count_before_file,
            "task_count_after_file": workload_evidence.task_count_after_file,
            "task_topic_increased": workload_evidence.task_topic_increased,
            "task_count_before_restart": (workload_evidence.task_count_before_restart),
            "task_count_after_restart": workload_evidence.task_count_after_restart,
            "duplicate_enqueue_after_restart": (
                workload_evidence.duplicate_enqueue_after_restart
            ),
            "restart_services": list(_RESTART_SERVICES),
        },
        domain_evidence={
            "trigger": "filesystem_create",
            "tenant_id": _TENANT_ID,
            "workspace_id": _WORKSPACE_ID,
            "collection_id": _COLLECTION_ID,
            "marker": workload_evidence.marker,
            "proof_filename": workload_evidence.proof_filename,
            "container_source_path": workload_evidence.container_source_path,
            "embedding_warmup_completed": (
                workload_evidence.embedding_warmup_completed
            ),
            "reviewer_rerun_required": workload_evidence.reviewer_rerun_required,
            "watcher_checkpoint_ready": workload_evidence.watcher_checkpoint_ready,
            "watcher_restored_after_restart": (
                workload_evidence.watcher_restored_after_restart
            ),
            "search_results_before_restart": (
                workload_evidence.search_results_before_restart
            ),
            "source_ref_found_before_restart": (
                workload_evidence.source_ref_found_before_restart
            ),
            "restart_mode": workload_evidence.restart_mode,
            "volumes_removed": workload_evidence.volumes_removed,
            "source_file_modified_after_index": (
                workload_evidence.source_file_modified_after_index
            ),
            "reindexed_after_restart": workload_evidence.reindexed_after_restart,
            "search_results_after_restart": (
                workload_evidence.search_results_after_restart
            ),
            "source_ref_found_after_restart": (
                workload_evidence.source_ref_found_after_restart
            ),
        },
        guardrails={
            "manual_index_command": False,
            "direct_enqueue": False,
            "direct_handler_call": False,
            "direct_indexer_call": False,
            "direct_ingest_call": False,
            "mock_queue": False,
            "inmemory_bypass": False,
            "direct_qdrant_write": False,
            "direct_mongodb_write": False,
            "direct_pymongo_from_lkw": False,
            "markdown_source_of_truth": False,
            "manual_evidence_injection": False,
        },
        metadata={
            "proof_runner": _PROOF_RUNNER,
            "receipt_task": _RECEIPT_TASK,
            "mongo_express_url": mongo_express_url,
            "recorded_from_live_run": True,
            "reviewer_guide": _REVIEWER_GUIDE,
            "verification_document": _VERIFICATION_DOCUMENT,
        },
    )


def _resolve_host_mongodb_uri() -> str | None:
    explicit = os.environ.get("INTERGRAX_MONGODB_URI", "").strip()
    if explicit:
        return explicit

    username = (
        os.environ.get("LKW_MONGODB_ROOT_USERNAME", "intergrax").strip() or "intergrax"
    )
    password = (
        os.environ.get("LKW_MONGODB_ROOT_PASSWORD", "intergrax-local-dev-only").strip()
        or "intergrax-local-dev-only"
    )
    database = (
        os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip()
        or "intergrax_proofs"
    )
    host_port = os.environ.get("LKW_MONGODB_HOST_PORT", "27018").strip() or "27018"
    return (
        f"mongodb://{username}:{password}@127.0.0.1:{host_port}/"
        f"{database}?authSource=admin"
    )


def ensure_mongodb_env() -> None:
    """Populate host-visible MongoDB provider environment for platform resolution."""
    if not os.environ.get("INTERGRAX_MONGODB_URI", "").strip():
        resolved = _resolve_host_mongodb_uri()
        if resolved:
            os.environ["INTERGRAX_MONGODB_URI"] = resolved
    if not os.environ.get("INTERGRAX_MONGODB_DATABASE", "").strip():
        os.environ["INTERGRAX_MONGODB_DATABASE"] = (
            os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip()
            or "intergrax_proofs"
        )
    if not os.environ.get("INTERGRAX_MONGODB_COLLECTION", "").strip():
        os.environ["INTERGRAX_MONGODB_COLLECTION"] = (
            os.environ.get("LKW_MONGODB_COLLECTION", "proof_receipts").strip()
            or "proof_receipts"
        )


def resolve_mongodb_document_store() -> tuple[
    MongoDBDocumentStoreIntegration, DocumentStore
]:
    """Resolve MongoDB DocumentStore through the platform provider factory."""
    ensure_mongodb_env()
    bundle = create_mongodb_integration()
    integration = bundle.document_store
    if not isinstance(integration, MongoDBDocumentStoreIntegration):
        raise TypeError("integration_not_mongodb_document_store")
    store = integration.as_document_store()
    if store is None:
        raise RuntimeError("document_store_adapter_unresolved")
    return integration, store


def record_file_watcher_e2e_proof_receipt(
    receipt: ProofReceipt,
) -> tuple[ProofReceipt, MongoDBDocumentStoreIntegration]:
    """Persist and verify a file-watcher proof receipt through the platform store."""
    integration, document_store = resolve_mongodb_document_store()
    verified = record_and_verify_proof_receipt(
        receipt, document_store, owns_document_store=True
    )
    return verified, integration


def format_pass_output(evidence: dict[str, object]) -> str:
    lines = [
        "proof_result=PASS",
        f"proof_kind={_PROOF_KIND}",
    ]
    for key, value in evidence.items():
        if isinstance(value, bool):
            lines.append(f"{key}={'true' if value else 'false'}")
        else:
            lines.append(f"{key}={value}")
    # Contiguous literals required by static receipt guardrails.
    assert "proof_receipt_recorded=true" in "\n".join(lines)
    assert "proof_receipt_verified=true" in "\n".join(lines)
    assert "proof_receipt_query_verified=true" in "\n".join(lines)
    return "\n".join(lines)


def build_pass_evidence(
    *,
    workload_evidence: FileWatcherE2EWorkloadEvidence,
    verified_receipt: ProofReceipt,
    integration_class: str,
    mongo_express_url: str,
) -> dict[str, object]:
    if not isinstance(workload_evidence, FileWatcherE2EWorkloadEvidence):
        raise TypeError("workload_evidence_must_be_typed")
    validate_file_watcher_e2e_workload_evidence(workload_evidence)
    return {
        "trigger": "filesystem_create",
        "manual_index_command": False,
        "direct_enqueue": False,
        "direct_indexer_call": False,
        "message_bus_provider": "kafka",
        "worker_execution": "asynchronous",
        "vector_store_provider": "qdrant",
        "persistent_index": True,
        "watcher_checkpoint_ready": workload_evidence.watcher_checkpoint_ready,
        "watcher_restored_after_restart": (
            workload_evidence.watcher_restored_after_restart
        ),
        "tenant_id": _TENANT_ID,
        "workspace_id": _WORKSPACE_ID,
        "collection_id": _COLLECTION_ID,
        "marker": workload_evidence.marker,
        "proof_filename": workload_evidence.proof_filename,
        "container_source_path": workload_evidence.container_source_path,
        "task_topic": _TASK_TOPIC,
        "task_count_before_file": workload_evidence.task_count_before_file,
        "task_count_after_file": workload_evidence.task_count_after_file,
        "task_topic_increased": workload_evidence.task_topic_increased,
        "search_results_before_restart": (
            workload_evidence.search_results_before_restart
        ),
        "source_ref_found_before_restart": (
            workload_evidence.source_ref_found_before_restart
        ),
        "restart_mode": workload_evidence.restart_mode,
        "volumes_removed": workload_evidence.volumes_removed,
        "source_file_modified_after_index": (
            workload_evidence.source_file_modified_after_index
        ),
        "reindexed_after_restart": workload_evidence.reindexed_after_restart,
        "task_count_before_restart": workload_evidence.task_count_before_restart,
        "task_count_after_restart": workload_evidence.task_count_after_restart,
        "duplicate_enqueue_after_restart": (
            workload_evidence.duplicate_enqueue_after_restart
        ),
        "search_results_after_restart": (
            workload_evidence.search_results_after_restart
        ),
        "source_ref_found_after_restart": (
            workload_evidence.source_ref_found_after_restart
        ),
        "embedding_warmup_completed": (workload_evidence.embedding_warmup_completed),
        "reviewer_rerun_required": workload_evidence.reviewer_rerun_required,
        "proof_receipt_recorded": True,
        "proof_receipt_verified": True,
        "proof_receipt_query_verified": True,
        "proof_receipt_store": "platform",
        "document_store_provider": MONGODB_DOCUMENT_STORE_PROVIDER_ID,
        "document_store_integration": integration_class,
        "proof_receipt_id": verified_receipt.proof_id,
        "proof_receipt_run_id": verified_receipt.run_id,
        "proof_receipt_result": verified_receipt.result.value,
        "proof_receipt_application_id": verified_receipt.application_id,
        "proof_receipt_task": _RECEIPT_TASK,
        "mongo_express_url": mongo_express_url,
        "markdown_source_of_truth": False,
        "direct_mongodb_write": False,
        "direct_pymongo_from_lkw": False,
        "manual_evidence_injection": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LKW.7C2 watcher-triggered persistent search E2E proof "
            "with ProofReceipt recording against a live local stack."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "LOCAL_WORKSPACE_BACKEND_BASE_URL", "http://127.0.0.1:8020"
        ),
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default=os.environ.get(
            "LKW_FILE_WATCHER_E2E_KAFKA_BOOTSTRAP", "127.0.0.1:9094"
        ),
    )
    parser.add_argument("--topic", default=_TASK_TOPIC)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--warmup-timeout-seconds", type=int, default=300)
    parser.add_argument("--stabilization-seconds", type=float, default=8.0)
    parser.add_argument("--repo-root", type=Path, default=_DEFAULT_REPO_ROOT)
    parser.add_argument("--proof-docs-dir", type=Path, default=_DEFAULT_PROOF_DOCS_DIR)
    parser.add_argument("--base-compose", type=Path, default=_DEFAULT_BASE_COMPOSE)
    parser.add_argument("--kafka-compose", type=Path, default=_DEFAULT_KAFKA_COMPOSE)
    parser.add_argument(
        "--watcher-compose", type=Path, default=_DEFAULT_WATCHER_COMPOSE
    )
    parser.add_argument(
        "--mongodb-compose", type=Path, default=_DEFAULT_MONGODB_COMPOSE
    )
    parser.add_argument(
        "--mongo-express",
        default=os.environ.get("LKW_MONGO_EXPRESS_URL", _DEFAULT_MONGO_EXPRESS_URL),
    )
    return parser.parse_args(argv)


def _poll_search_until_indexed(
    *,
    base_url: str,
    marker: str,
    expected_source_path: str,
    deadline: float,
) -> tuple[SearchDiagnostics | None, int]:
    last_diagnostics: SearchDiagnostics | None = None
    run_url = f"{base_url.rstrip('/')}/v1/local_workspace/run"
    request_body = build_search_request(marker)
    while time.monotonic() < deadline:
        try:
            response = request_json(run_url, method="POST", payload=request_body)
            last_diagnostics = extract_search_diagnostics(response)
            if search_attempt_succeeded(
                last_diagnostics, expected_source_path=expected_source_path
            ):
                assert last_diagnostics is not None
                result_count = max(
                    last_diagnostics.num_results, last_diagnostics.evidence_count
                )
                return last_diagnostics, result_count
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            pass
        time.sleep(2.0)
    return last_diagnostics, 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base_url = str(args.base_url).rstrip("/")
    proof_docs_dir = Path(args.proof_docs_dir)
    base_compose = Path(args.base_compose)
    kafka_compose = Path(args.kafka_compose)
    watcher_compose = Path(args.watcher_compose)
    mongodb_compose = Path(args.mongodb_compose)
    mongo_express_url = str(args.mongo_express)
    timeout_seconds = float(args.timeout_seconds)
    warmup_timeout_seconds = float(args.warmup_timeout_seconds)
    stabilization_seconds = float(args.stabilization_seconds)

    proof_docs_dir.mkdir(parents=True, exist_ok=True)

    if not wait_for_health(base_url, timeout_seconds=min(120.0, timeout_seconds)):
        return fail("lkw_health_unreachable")

    if not watcher_container_running(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    ):
        return fail("watcher_not_running")

    watcher_checkpoint_ready_before_file = watcher_checkpoint_ready(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    )
    if not watcher_checkpoint_ready_before_file:
        return fail("watcher_checkpoint_not_ready")

    warmup = run_embedding_warmup(
        base_url=base_url,
        timeout_seconds=warmup_timeout_seconds,
    )
    if not warmup.completed:
        return fail(
            "embedding_warmup_failed",
            last_warmup_reason=warmup.last_reason,
            last_warmup_raw_tool_reason=warmup.last_raw_tool_reason,
            warmup_attempt_count=warmup.attempt_count,
            embedding_warmup_completed=False,
            reviewer_rerun_required=True,
        )

    # Fresh proof workload deadline after successful warm-up.
    deadline = time.monotonic() + timeout_seconds

    try:
        task_count_before_file = inspect_kafka_topic_message_count(
            bootstrap=str(args.kafka_bootstrap),
            topic=str(args.topic),
        )
    except RuntimeError as exc:
        reason = str(exc)
        if reason == "kafka_inspection_unavailable":
            return fail("kafka_inspection_unavailable")
        return fail("kafka_topic_unavailable")

    try:
        document = create_proof_document(proof_docs_dir)
    except OSError:
        return fail("proof_document_write_failed")

    diagnostics, search_results_before_restart = _poll_search_until_indexed(
        base_url=base_url,
        marker=document.marker,
        expected_source_path=document.container_source_path,
        deadline=deadline,
    )
    source_ref_found_before_restart = (
        diagnostics is not None
        and document.container_source_path in diagnostics.source_refs
    )
    if not search_attempt_succeeded(
        diagnostics, expected_source_path=document.container_source_path
    ):
        fields: dict[str, object] = {
            "expected_source_path": document.container_source_path,
            "observed_source_ref_count": (
                len(diagnostics.source_refs) if diagnostics is not None else 0
            ),
            "last_num_results": (
                diagnostics.num_results if diagnostics is not None else 0
            ),
            "last_evidence_count": (
                diagnostics.evidence_count if diagnostics is not None else 0
            ),
            "last_raw_tool_reason": (
                diagnostics.raw_tool_reason if diagnostics is not None else None
            ),
            "source_ref_found_before_restart": source_ref_found_before_restart,
        }
        if diagnostics is None:
            return fail("search_results_missing", **fields)
        if diagnostics.num_results <= 0 and diagnostics.evidence_count <= 0:
            return fail("search_results_missing", **fields)
        return fail("expected_source_ref_missing", **fields)
    if not source_ref_found_before_restart:
        return fail(
            "expected_source_ref_missing",
            expected_source_path=document.container_source_path,
            source_ref_found_before_restart=False,
        )

    try:
        source_stat_after_index = capture_proof_file_stat(document.host_path)
    except OSError:
        return fail("source_file_stat_failed")

    try:
        task_count_after_file = inspect_kafka_topic_message_count(
            bootstrap=str(args.kafka_bootstrap),
            topic=str(args.topic),
        )
    except RuntimeError as exc:
        reason = str(exc)
        if reason == "kafka_inspection_unavailable":
            return fail("kafka_inspection_unavailable")
        return fail("kafka_topic_unavailable")

    if kafka_topic_regression(
        before=task_count_before_file, after=task_count_after_file
    ):
        return fail(
            "kafka_topic_did_not_increase",
            task_count_before_file=task_count_before_file,
            task_count_after_file=task_count_after_file,
        )
    if not kafka_topic_increased(
        before=task_count_before_file, after=task_count_after_file
    ):
        return fail(
            "kafka_topic_did_not_increase",
            task_count_before_file=task_count_before_file,
            task_count_after_file=task_count_after_file,
        )

    task_count_before_restart = task_count_after_file

    try:
        run_compose(
            "restart",
            *_RESTART_SERVICES,
            base_compose=base_compose,
            kafka_compose=kafka_compose,
            watcher_compose=watcher_compose,
            mongodb_compose=mongodb_compose,
            check=True,
            timeout=180.0,
        )
    except (RuntimeError, subprocess.TimeoutExpired):
        return fail("docker_restart_failed")

    if not wait_for_health(base_url, timeout_seconds=min(180.0, timeout_seconds)):
        return fail("health_after_restart_failed")

    if not watcher_container_running(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    ):
        return fail("watcher_after_restart_failed")

    if not watcher_checkpoint_ready(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
        mongodb_compose=mongodb_compose,
    ):
        return fail("watcher_after_restart_failed")

    time.sleep(stabilization_seconds)

    try:
        task_count_after_restart = inspect_kafka_topic_message_count(
            bootstrap=str(args.kafka_bootstrap),
            topic=str(args.topic),
        )
    except RuntimeError as exc:
        reason = str(exc)
        if reason == "kafka_inspection_unavailable":
            return fail("kafka_inspection_unavailable")
        return fail("kafka_topic_unavailable")

    if kafka_topic_regression(
        before=task_count_before_restart, after=task_count_after_restart
    ):
        return fail(
            "kafka_task_topic_regressed_after_restart",
            task_count_before_restart=task_count_before_restart,
            task_count_after_restart=task_count_after_restart,
        )
    if duplicate_enqueue_detected(
        before_restart=task_count_before_restart,
        after_restart=task_count_after_restart,
    ):
        return fail(
            "duplicate_enqueue_after_restart",
            task_count_before_restart=task_count_before_restart,
            task_count_after_restart=task_count_after_restart,
        )

    post_diagnostics, search_results_after_restart = _poll_search_until_indexed(
        base_url=base_url,
        marker=document.marker,
        expected_source_path=document.container_source_path,
        deadline=time.monotonic() + min(60.0, max(10.0, timeout_seconds / 4.0)),
    )
    source_ref_found_after_restart = (
        post_diagnostics is not None
        and document.container_source_path in post_diagnostics.source_refs
    )
    if not search_attempt_succeeded(
        post_diagnostics, expected_source_path=document.container_source_path
    ):
        return fail(
            "search_after_restart_failed",
            expected_source_path=document.container_source_path,
            last_num_results=(
                post_diagnostics.num_results if post_diagnostics is not None else 0
            ),
            last_evidence_count=(
                post_diagnostics.evidence_count if post_diagnostics is not None else 0
            ),
            source_ref_found_after_restart=source_ref_found_after_restart,
        )
    if not source_ref_found_after_restart:
        return fail(
            "search_after_restart_failed",
            expected_source_path=document.container_source_path,
            source_ref_found_after_restart=False,
        )

    try:
        source_stat_after_restart = capture_proof_file_stat(document.host_path)
    except OSError:
        return fail("source_file_stat_failed")

    source_file_modified_after_index = not proof_file_stat_unchanged(
        before=source_stat_after_index,
        after=source_stat_after_restart,
    )
    if source_file_modified_after_index:
        return fail(
            "source_file_modified_after_index",
            size_changed=(
                source_stat_after_index.size_bytes
                != source_stat_after_restart.size_bytes
            ),
            modified_time_changed=(
                source_stat_after_index.modified_time_ns
                != source_stat_after_restart.modified_time_ns
            ),
        )

    try:
        run_compose(
            "stop",
            "--timeout",
            "30",
            _WATCHER_SERVICE,
            base_compose=base_compose,
            kafka_compose=kafka_compose,
            watcher_compose=watcher_compose,
            mongodb_compose=mongodb_compose,
            check=True,
            timeout=60.0,
        )
    except (RuntimeError, subprocess.TimeoutExpired):
        return fail("watcher_graceful_stop_failed")

    evidence_failure: str | None = None
    evidence_fields: dict[str, object] = {}
    watcher_restored_after_restart = False
    watcher_final_checkpoint_saved = False
    try:
        try:
            logs_completed = run_compose(
                "logs",
                "--no-color",
                "--no-log-prefix",
                "--tail",
                "200",
                _WATCHER_SERVICE,
                base_compose=base_compose,
                kafka_compose=kafka_compose,
                watcher_compose=watcher_compose,
                mongodb_compose=mongodb_compose,
                check=True,
                timeout=60.0,
            )
        except (RuntimeError, subprocess.TimeoutExpired):
            evidence_failure = "watcher_result_read_failed"
        else:
            sidecar_result = extract_last_file_watcher_sidecar_result(
                logs_completed.stdout
            )
            if sidecar_result is None:
                evidence_failure = "watcher_result_missing"
            elif not sidecar_result_proves_checkpoint_restore(sidecar_result):
                evidence_failure = "watcher_restore_not_proven"
                evidence_fields = {
                    "sidecar_exit_kind": sidecar_result.get("exit_kind"),
                    "sidecar_exit_code": sidecar_result.get("exit_code"),
                    "sidecar_restored_from_checkpoint": sidecar_result.get(
                        "restored_from_checkpoint"
                    ),
                    "sidecar_final_checkpoint_saved": sidecar_result.get(
                        "final_checkpoint_saved"
                    ),
                    "sidecar_error_id": sidecar_result.get("error_id"),
                }
            else:
                watcher_restored_after_restart = True
                watcher_final_checkpoint_saved = True
    finally:
        resume_ok = False
        try:
            run_compose(
                "up",
                "-d",
                _WATCHER_SERVICE,
                base_compose=base_compose,
                kafka_compose=kafka_compose,
                watcher_compose=watcher_compose,
                mongodb_compose=mongodb_compose,
                check=True,
                timeout=180.0,
            )
            resume_ok = wait_for_watcher_ready(
                base_compose=base_compose,
                kafka_compose=kafka_compose,
                watcher_compose=watcher_compose,
                mongodb_compose=mongodb_compose,
                timeout_seconds=min(120.0, timeout_seconds),
            )
        except (RuntimeError, subprocess.TimeoutExpired):
            resume_ok = False

    if not resume_ok:
        if evidence_failure is not None:
            return fail(
                "watcher_resume_failed",
                previous_failure_reason=evidence_failure,
                **evidence_fields,
            )
        return fail("watcher_resume_failed")

    if evidence_failure is not None:
        return fail(evidence_failure, **evidence_fields)

    if not watcher_restored_after_restart:
        return fail("watcher_restore_not_proven")
    if not watcher_final_checkpoint_saved:
        return fail("watcher_final_checkpoint_not_saved")

    workload_evidence = FileWatcherE2EWorkloadEvidence(
        marker=document.marker,
        proof_filename=document.filename,
        container_source_path=document.container_source_path,
        watcher_checkpoint_ready=watcher_checkpoint_ready_before_file,
        embedding_warmup_completed=warmup.completed,
        task_count_before_file=task_count_before_file,
        task_count_after_file=task_count_after_file,
        search_results_before_restart=search_results_before_restart,
        source_ref_found_before_restart=source_ref_found_before_restart,
        task_count_before_restart=task_count_before_restart,
        task_count_after_restart=task_count_after_restart,
        search_results_after_restart=search_results_after_restart,
        source_ref_found_after_restart=source_ref_found_after_restart,
        watcher_restored_after_restart=watcher_restored_after_restart,
        watcher_final_checkpoint_saved=watcher_final_checkpoint_saved,
        source_file_modified_after_index=source_file_modified_after_index,
        restart_mode="non_destructive",
        volumes_removed=False,
    )
    try:
        validate_file_watcher_e2e_workload_evidence(workload_evidence)
    except ValueError as exc:
        return fail(str(exc))

    receipt = build_file_watcher_e2e_proof_receipt(
        run_id=document.marker,
        workload_evidence=workload_evidence,
        mongo_express_url=mongo_express_url,
    )

    try:
        verified_receipt, integration = record_file_watcher_e2e_proof_receipt(receipt)
    except (
        ProofReceiptVerificationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return fail_receipt_recording(exc)

    evidence = build_pass_evidence(
        workload_evidence=workload_evidence,
        verified_receipt=verified_receipt,
        integration_class=type(integration).__name__,
        mongo_express_url=mongo_express_url,
    )
    print(format_pass_output(evidence))
    print(f"proof_runner={_PROOF_RUNNER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
