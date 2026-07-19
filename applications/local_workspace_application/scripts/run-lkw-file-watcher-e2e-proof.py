#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW.7C1 watcher-triggered persistent search E2E proof.

Uses docker compose against the dedicated watcher overlay, Kafka task-topic
inspection, and local.workspace.search diagnostics only. Receipt recording is
out of scope for this workload.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROOF_KIND = "file_watcher_persistent_search"
_PROOF_RUNNER = "run-lkw-file-watcher-e2e-proof.py"
_SIDECAR_RESULT_SCHEMA = "lkw.file_watcher_sidecar_result.v1"

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
        print(f"{key}={value}")
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
        *compose_args,
    ]


def run_compose(
    *compose_args: str,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    check: bool = True,
    timeout: float | None = 120.0,
) -> subprocess.CompletedProcess[str]:
    command = build_compose_command(
        *compose_args,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
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
) -> bool:
    completed = run_compose(
        "ps",
        "--format",
        "json",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
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
    return SearchDiagnostics(
        num_results=num_results,
        evidence_count=evidence_count,
        source_refs=source_refs,
        raw_tool_reason=raw_tool_reason,
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


def build_restart_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
) -> list[str]:
    return build_compose_command(
        "restart",
        *_RESTART_SERVICES,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
    )


def build_watcher_graceful_stop_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
) -> list[str]:
    return build_compose_command(
        "stop",
        "--timeout",
        "30",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
    )


def build_watcher_logs_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
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
    )


def build_watcher_resume_command(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
) -> list[str]:
    return build_compose_command(
        "up",
        "-d",
        _WATCHER_SERVICE,
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
    )


def wait_for_watcher_ready(
    *,
    base_compose: Path,
    kafka_compose: Path,
    watcher_compose: Path,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if watcher_container_running(
            base_compose=base_compose,
            kafka_compose=kafka_compose,
            watcher_compose=watcher_compose,
        ) and watcher_checkpoint_ready(
            base_compose=base_compose,
            kafka_compose=kafka_compose,
            watcher_compose=watcher_compose,
        ):
            return True
        time.sleep(1.0)
    return False


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
    return "\n".join(lines)


def build_pass_evidence(
    *,
    marker: str,
    filename: str,
    container_source_path: str,
    task_count_before_file: int,
    task_count_after_file: int,
    search_results_before_restart: int,
    task_count_before_restart: int,
    task_count_after_restart: int,
    search_results_after_restart: int,
    watcher_restored_after_restart: bool,
    source_file_modified_after_index: bool,
) -> dict[str, object]:
    return {
        "trigger": "filesystem_create",
        "manual_index_command": False,
        "direct_enqueue": False,
        "direct_indexer_call": False,
        "message_bus_provider": "kafka",
        "worker_execution": "asynchronous",
        "vector_store_provider": "qdrant",
        "persistent_index": True,
        "watcher_checkpoint_ready": True,
        "watcher_restored_after_restart": watcher_restored_after_restart,
        "tenant_id": _TENANT_ID,
        "workspace_id": _WORKSPACE_ID,
        "collection_id": _COLLECTION_ID,
        "marker": marker,
        "proof_filename": filename,
        "container_source_path": container_source_path,
        "task_topic": _TASK_TOPIC,
        "task_count_before_file": task_count_before_file,
        "task_count_after_file": task_count_after_file,
        "task_topic_increased": True,
        "search_results_before_restart": search_results_before_restart,
        "source_ref_found_before_restart": True,
        "restart_mode": "non_destructive",
        "volumes_removed": False,
        "source_file_modified_after_index": source_file_modified_after_index,
        "reindexed_after_restart": False,
        "task_count_before_restart": task_count_before_restart,
        "task_count_after_restart": task_count_after_restart,
        "duplicate_enqueue_after_restart": False,
        "search_results_after_restart": search_results_after_restart,
        "source_ref_found_after_restart": True,
        "proof_receipt_recorded": False,
        "proof_receipt_task": "LKW.7C2",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LKW.7C1 watcher-triggered persistent search E2E proof "
            "against a live local stack."
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
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--stabilization-seconds", type=float, default=8.0)
    parser.add_argument("--repo-root", type=Path, default=_DEFAULT_REPO_ROOT)
    parser.add_argument("--proof-docs-dir", type=Path, default=_DEFAULT_PROOF_DOCS_DIR)
    parser.add_argument("--base-compose", type=Path, default=_DEFAULT_BASE_COMPOSE)
    parser.add_argument("--kafka-compose", type=Path, default=_DEFAULT_KAFKA_COMPOSE)
    parser.add_argument(
        "--watcher-compose", type=Path, default=_DEFAULT_WATCHER_COMPOSE
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
    timeout_seconds = float(args.timeout_seconds)
    stabilization_seconds = float(args.stabilization_seconds)
    deadline = time.monotonic() + timeout_seconds

    proof_docs_dir.mkdir(parents=True, exist_ok=True)

    if not wait_for_health(base_url, timeout_seconds=min(120.0, timeout_seconds)):
        return fail("lkw_health_unreachable")

    if not watcher_container_running(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
    ):
        return fail("watcher_not_running")

    if not watcher_checkpoint_ready(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
    ):
        return fail("watcher_checkpoint_not_ready")

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
        }
        if diagnostics is None:
            return fail("search_results_missing", **fields)
        if diagnostics.num_results <= 0 and diagnostics.evidence_count <= 0:
            return fail("search_results_missing", **fields)
        return fail("expected_source_ref_missing", **fields)

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
    ):
        return fail("watcher_after_restart_failed")

    if not watcher_checkpoint_ready(
        base_compose=base_compose,
        kafka_compose=kafka_compose,
        watcher_compose=watcher_compose,
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
            "duplicate_enqueue_after_restart",
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
            check=True,
            timeout=60.0,
        )
    except (RuntimeError, subprocess.TimeoutExpired):
        return fail("watcher_graceful_stop_failed")

    evidence_failure: str | None = None
    evidence_fields: dict[str, object] = {}
    watcher_restored_after_restart = False
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
                check=True,
                timeout=180.0,
            )
            resume_ok = wait_for_watcher_ready(
                base_compose=base_compose,
                kafka_compose=kafka_compose,
                watcher_compose=watcher_compose,
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

    evidence = build_pass_evidence(
        marker=document.marker,
        filename=document.filename,
        container_source_path=document.container_source_path,
        task_count_before_file=task_count_before_file,
        task_count_after_file=task_count_after_file,
        search_results_before_restart=search_results_before_restart,
        task_count_before_restart=task_count_before_restart,
        task_count_after_restart=task_count_after_restart,
        search_results_after_restart=search_results_after_restart,
        watcher_restored_after_restart=watcher_restored_after_restart,
        source_file_modified_after_index=source_file_modified_after_index,
    )
    print(format_pass_output(evidence))
    print(f"proof_runner={_PROOF_RUNNER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
