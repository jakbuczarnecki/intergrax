#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Bounded live proof: WEB_URL intake through real SecureHttpWebContentCapture.

Validates:
canonical Compose stack
→ POST /knowledge/web-urls for a stable public HTTPS origin (example.com)
→ real SecureHttpWebContentCapture + WEB_URL ingestion/indexing in LKW
→ real tenant/workspace Qdrant vector scope
→ POST /v2/local_workspace/workspaces/{id}/ask (mode=indexed_only)
→ indexed Hybrid Ask branch with grounded answer and indexed citation/evidence
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import time
import urllib.error
import urllib.request
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
_PROOF_URL = "https://example.com/"
_DISPLAY_URL = "https://example.com"
_RETRIEVAL_MARKER = "Example Domain"
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


def assert_web_url_acceptance(body: dict[str, object]) -> tuple[str, str]:
    safe_display = str(body.get("safe_display_url", ""))
    if safe_display != _DISPLAY_URL:
        raise ProofFailure("web_url_accept", "safe_display_url_mismatch")
    if "track=secret" in json.dumps(body):
        raise ProofFailure("web_url_accept", "unsafe_query_leak")
    source_id = str(body.get("source_id", "")).strip()
    operation_id = str(body.get("operation_id", "")).strip()
    if not source_id or not operation_id:
        raise ProofFailure("web_url_accept", "missing_source_or_operation")
    return source_id, operation_id


def assert_indexed_hybrid_ask_run(
    run: dict[str, object],
    *,
    source_id: str,
    marker: str = _RETRIEVAL_MARKER,
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
    answer = run.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ProofFailure("hybrid_ask", "empty_answer")
    citations = run.get("citations") or []
    if not isinstance(citations, list) or not citations:
        raise ProofFailure("hybrid_ask", "missing_citations")
    citation = citations[0]
    if not isinstance(citation, dict):
        raise ProofFailure("hybrid_ask", "citation_not_object")
    if str(citation.get("source_id", "")) != source_id:
        raise ProofFailure("hybrid_ask", "citation_source_mismatch")
    evidence_id = str(citation.get("evidence_id", ""))
    if not evidence_id.startswith(_INDEXED_EVIDENCE_PREFIX):
        raise ProofFailure("hybrid_ask", "citation_not_indexed_evidence")
    if citation.get("evidence_type") != "indexed":
        raise ProofFailure("hybrid_ask", "citation_evidence_type_not_indexed")
    file_name = str(citation.get("file_name", ""))
    safe_display = str(citation.get("safe_display_name", file_name))
    if _DISPLAY_URL not in {file_name, safe_display}:
        raise ProofFailure("hybrid_ask", "citation_display_url_mismatch")
    excerpt = str(citation.get("excerpt", ""))
    if marker not in excerpt:
        raise ProofFailure("hybrid_ask", "citation_excerpt_missing_marker")
    if marker not in answer:
        raise ProofFailure("hybrid_ask", "answer_missing_marker")


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
    tenant_id = "lkw-web-url-indexed-ask"
    idempotency_key = f"web-url-proof-{suffix}"
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
            payload={"name": f"WEB URL Indexed Ask {suffix}"},
        )
        if status != 201:
            raise ProofFailure("workspace_create", f"http_{status}")
        workspace_id = str(workspace["workspace_id"])

        failing_phase = "web_url_accept"
        status, accepted = _request_json(
            f"{base_url}{_API_V1}/workspaces/{workspace_id}/knowledge/web-urls",
            method="POST",
            headers={
                "X-Tenant-Id": tenant_id,
                "Idempotency-Key": idempotency_key,
            },
            payload={"url": _PROOF_URL},
        )
        if status != 202:
            raise ProofFailure("web_url_accept", f"http_{status}")
        source_id, operation_id = assert_web_url_acceptance(accepted)

        failing_phase = "web_url_index"
        operation = wait_operation(base_url, operation_id, tenant_id=tenant_id)
        if operation.get("status") != "completed":
            error_code = operation.get("error_code", operation.get("error"))
            raise ProofFailure("web_url_index", f"status={operation.get('status')}:{error_code}")
        if int(operation.get("documents_indexed") or 0) < 1:
            raise ProofFailure("web_url_index", "documents_not_indexed")

        failing_phase = "search"
        status, search = _request_json(
            f"{base_url}{_API_V1}/workspaces/{workspace_id}/search",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"query": _RETRIEVAL_MARKER, "limit": 5},
        )
        if status != 200:
            raise ProofFailure("search", f"http_{status}")
        if len(search.get("results") or []) < 1:
            raise ProofFailure("search", "no_verified_results")

        failing_phase = "hybrid_ask"
        status, ask_run = _request_json(
            f"{base_url}{_API_V2}/workspaces/{workspace_id}/ask",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={
                "question": "What title text appears on the example.com page?",
                "mode": "indexed_only",
                "indexed_max_results": 5,
            },
            timeout=300.0,
        )
        if status != 200:
            raise ProofFailure("hybrid_ask", f"http_{status}")
        assert_indexed_hybrid_ask_run(ask_run, source_id=source_id)
        run_id = str(ask_run["run_id"])

        _print_kv("proof_result", "PASS")
        _print_kv("proof_kind", "web_url_indexed_ask")
        _print_kv("proof_id", "LKW-WEB-URL-INDEXED-ASK")
        _print_kv("capture_implementation", "SecureHttpWebContentCapture")
        _print_kv("ssrf_policy_weakened", "false")
        _print_kv("controlled_origin", _PROOF_URL)
        _print_kv("vector_store_provider", vector_store_provider)
        _print_kv("tenant_id", tenant_id)
        _print_kv("workspace_id", workspace_id)
        _print_kv("source_id", source_id)
        _print_kv("operation_id", operation_id)
        _print_kv("run_id", run_id)
        _print_kv("indexed_retrieval_status", "completed")
        _print_kv("live_execution_status", "skipped")
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
