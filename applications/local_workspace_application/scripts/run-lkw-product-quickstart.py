#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Shared cross-platform LKW product quickstart runner.

OS launchers are transport-only. Product orchestration and acceptance live here.
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
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_SAMPLE_FILE = _APP_DIR / "sample_docs" / "lkw_product_quickstart.txt"
_ENV_FILE = _APP_DIR / ".env"
_ENV_EXAMPLE = _APP_DIR / ".env.example"
_BOOTSTRAP_BAT = _SCRIPT_DIR / "build-local-docker.bat"
_BOOTSTRAP_SH = _SCRIPT_DIR / "build-local-docker.sh"
_COMPOSE_FILE = _APP_DIR / "docker" / "docker-compose.yml"
_COMPOSE_PROJECT = "intergrax_lkw"
_NEW_ENV_OLLAMA_EMBED_MODEL = "nomic-embed-text"
_MODEL_RESOLUTION_CODE = (
    "import os; "
    "from intergrax.rag.embedding.providers.ollama_embedding_provider "
    "import OllamaEmbeddingProvider; "
    "print(os.getenv(OllamaEmbeddingProvider.ENV_MODEL) "
    "or OllamaEmbeddingProvider.DEFAULT_MODEL)"
)
_MAX_EMBEDDING_MODEL_LENGTH = 256

_DEFAULT_BASE_URL = "http://127.0.0.1:8020"
_DEFAULT_TIMEOUT = 600
_API_PREFIX = "/v1/local_workspace"
_TENANT_ID = "lkw-product-quickstart"
_QUESTION = "What is the project codename?"
_ANSWER_MARKER = "AURORA-17"
_CITATION_FILE = "lkw_product_quickstart.txt"
_SAFE_REASON = re.compile(r"^[A-Za-z0-9_.-]+$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_FORBIDDEN_OUTPUT_SNIPPETS = (
    "storage_key",
    "object_id",
    "s3://",
    "presigned",
    "sha256:",
    "managed_files",
    "managed_upload_staging",
    "source_path",
    "INTERGRAX_",
    "mongodb",
    "qdrant",
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


class QuickstartError(Exception):
    def __init__(self, reason: str, *, stage: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


@dataclass
class QuickstartConfig:
    os_family: OsFamily
    wrapper_id: WrapperId
    base_url: str
    timeout_seconds: int
    skip_stack_start: bool


def detect_os_family(system_name: str | None = None) -> OsFamily:
    name = system_name if system_name is not None else platform.system()
    mapping = {
        "Windows": OsFamily.WINDOWS,
        "Linux": OsFamily.LINUX,
        "Darwin": OsFamily.MACOS,
    }
    detected = mapping.get(name)
    if detected is None:
        raise QuickstartError("unsupported_operating_system", stage="preflight")
    return detected


def validate_os_wrapper_pair(
    os_family: OsFamily,
    wrapper_id: WrapperId,
    *,
    detected: OsFamily | None = None,
) -> None:
    if (os_family, wrapper_id) not in VALID_OS_WRAPPER_PAIRS:
        raise QuickstartError("invalid_os_wrapper_pair", stage="preflight")
    actual = detected if detected is not None else detect_os_family()
    if actual is not os_family:
        raise QuickstartError("operating_system_mismatch", stage="preflight")


def validate_loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise QuickstartError("invalid_base_url_scheme", stage="preflight")
    host = (parsed.hostname or "").strip().lower()
    if host not in _LOOPBACK_HOSTS:
        raise QuickstartError("non_loopback_base_url", stage="preflight")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _print_kv(key: str, value: object) -> None:
    print(f"{key}={value}", flush=True)


def _emit_failure(stage: str, reason: str) -> None:
    safe_reason = reason if _SAFE_REASON.fullmatch(reason) else "unsafe_failure_reason"
    _print_kv("lkw_quickstart_result", "FAIL")
    _print_kv("failed_stage", stage)
    _print_kv("failure_reason", safe_reason)


def _assert_safe_user_text(text: str) -> None:
    lowered = text.lower()
    for snippet in _FORBIDDEN_OUTPUT_SNIPPETS:
        if snippet.lower() in lowered:
            raise QuickstartError("unsafe_output_detected", stage="preflight")


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    stage: str = "stack_start",
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise QuickstartError("command_timeout", stage=stage) from None
    except OSError:
        raise QuickstartError("command_start_failed", stage=stage) from None


def ensure_env_file() -> bool:
    if _ENV_FILE.is_file():
        return False
    if not _ENV_EXAMPLE.is_file():
        raise QuickstartError("env_example_missing", stage="preflight")
    try:
        shutil.copyfile(_ENV_EXAMPLE, _ENV_FILE)
        with _ENV_FILE.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\nINTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL="
                f"{_NEW_ENV_OLLAMA_EMBED_MODEL}\n"
            )
    except OSError:
        raise QuickstartError("env_materialization_failed", stage="preflight") from None
    return True


def bootstrap_args(os_family: OsFamily) -> list[str]:
    if os_family is OsFamily.WINDOWS:
        return ["cmd.exe", "/c", str(_BOOTSTRAP_BAT)]
    return ["sh", str(_BOOTSTRAP_SH)]


def compose_exec_args(*compose_command: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        _COMPOSE_PROJECT,
        "-f",
        str(_COMPOSE_FILE),
        *compose_command,
    ]


def _validate_resolved_embedding_model(output: str) -> str:
    if not isinstance(output, str):
        raise QuickstartError(
            "embedding_model_resolution_failed",
            stage="stack_start",
        )
    non_empty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(non_empty_lines) != 1:
        raise QuickstartError(
            "embedding_model_resolution_failed",
            stage="stack_start",
        )
    model_name = non_empty_lines[0]
    if (
        not model_name
        or len(model_name) > _MAX_EMBEDDING_MODEL_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in model_name)
        or any(character.isspace() for character in model_name)
    ):
        raise QuickstartError(
            "embedding_model_resolution_failed",
            stage="stack_start",
        )
    return model_name


def resolve_ollama_embedding_model(*, timeout_seconds: int) -> str:
    completed = run_command(
        compose_exec_args(
            "exec",
            "-T",
            "local_workspace",
            "python",
            "-c",
            _MODEL_RESOLUTION_CODE,
        ),
        cwd=_APP_DIR,
        timeout=timeout_seconds,
        stage="stack_start",
    )
    if completed.returncode != 0:
        raise QuickstartError(
            "embedding_model_resolution_failed",
            stage="stack_start",
        )
    return _validate_resolved_embedding_model(completed.stdout)


def ensure_ollama_embedding_model(
    model_name: str,
    *,
    timeout_seconds: int,
) -> None:
    completed = run_command(
        compose_exec_args(
            "exec",
            "-T",
            "ollama",
            "ollama",
            "pull",
            model_name,
        ),
        cwd=_APP_DIR,
        timeout=timeout_seconds,
        stage="stack_start",
    )
    if completed.returncode != 0:
        raise QuickstartError("embedding_model_pull_failed", stage="stack_start")


def _decode_json_object(raw: bytes, *, stage: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise QuickstartError("invalid_json_response", stage=stage) from None
    if not isinstance(payload, dict):
        raise QuickstartError("invalid_response_shape", stage=stage)
    return payload


def response_integer(
    payload: Mapping[str, Any],
    field: str,
    *,
    stage: str,
    default: int = 0,
) -> int:
    value = payload.get(field, default)
    if isinstance(value, bool):
        raise QuickstartError("invalid_response_shape", stage=stage)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise QuickstartError("invalid_response_shape", stage=stage) from None


def http_get_json(
    url: str,
    headers: Mapping[str, str],
    *,
    timeout: float = 30.0,
    stage: str = "health",
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = b""
    except (urllib.error.URLError, TimeoutError, OSError):
        raise QuickstartError("http_transport_failed", stage=stage) from None
    if status < 200 or status >= 300:
        raise QuickstartError(f"http_status_{status}", stage=stage)
    return _decode_json_object(body, stage=stage)


def http_post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    timeout: float = 60.0,
    stage: str,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(dict(body)).encode("utf-8")
    merged = dict(headers)
    merged["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = b""
    except (urllib.error.URLError, TimeoutError, OSError):
        raise QuickstartError("http_transport_failed", stage=stage) from None
    if status < 200 or status >= 300:
        return status, {}
    return status, _decode_json_object(raw, stage=stage)


def encode_multipart_file(
    field_name: str,
    filename: str,
    content: bytes,
    *,
    content_type: str = "text/plain",
) -> tuple[str, bytes]:
    boundary = f"----LkwQuickstart{uuid.uuid4().hex}"
    lines: list[bytes] = []
    lines.append(f"--{boundary}\r\n".encode("ascii"))
    lines.append(
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n".encode(
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
    headers: Mapping[str, str],
    *,
    timeout: float = 120.0,
    stage: str,
) -> tuple[int, dict[str, Any]]:
    merged = dict(headers)
    request = urllib.request.Request(url, data=body, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        raise QuickstartError("invalid_json_response", stage=stage) from None
    if not isinstance(payload, dict):
        raise QuickstartError("invalid_json_object", stage=stage)
    return status, payload


def wait_for_health(base_url: str, *, timeout_seconds: int) -> None:
    health_url = f"{base_url}/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            payload = http_get_json(health_url, {}, timeout=5.0)
            if str(payload.get("status", "")).strip() == "ok":
                return
        except QuickstartError as exc:
            if exc.reason != "http_transport_failed":
                raise
        time.sleep(2)
    raise QuickstartError("health_timeout", stage="health")


def wait_for_operation(
    base_url: str,
    operation_id: str,
    headers: Mapping[str, str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = f"{base_url}{_API_PREFIX}/operations/{operation_id}"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = http_get_json(url, headers, timeout=15.0, stage="ingestion")
        status = str(payload.get("status", "")).strip().lower()
        if status == "completed":
            documents_indexed = response_integer(
                payload,
                "documents_indexed",
                stage="ingestion",
            )
            files_failed = response_integer(
                payload,
                "files_failed",
                stage="ingestion",
            )
            if documents_indexed < 1:
                raise QuickstartError("documents_not_indexed", stage="ingestion")
            if files_failed != 0:
                raise QuickstartError("files_failed_nonzero", stage="ingestion")
            if payload.get("error") is not None:
                raise QuickstartError("operation_error_present", stage="ingestion")
            return payload
        if status == "failed":
            raise QuickstartError("operation_failed", stage="ingestion")
        time.sleep(2)
    raise QuickstartError("operation_timeout", stage="ingestion")


def _tenant_headers() -> dict[str, str]:
    return {"X-Tenant-Id": _TENANT_ID}


def create_workspace(base_url: str) -> str:
    suffix = uuid.uuid4().hex[:8]
    body = {
        "name": f"LKW Product Quickstart {suffix}",
        "description": "Local managed-file and grounded Ask evaluation",
    }
    status, payload = http_post_json(
        f"{base_url}{_API_PREFIX}/workspaces",
        body,
        _tenant_headers(),
        stage="workspace",
    )
    if status != 201:
        raise QuickstartError("workspace_create_failed", stage="workspace")
    workspace_id = str(payload.get("workspace_id", "")).strip()
    if not workspace_id:
        raise QuickstartError("workspace_id_missing", stage="workspace")
    return workspace_id


def upload_sample_file(base_url: str, workspace_id: str) -> str:
    if not _SAMPLE_FILE.is_file():
        raise QuickstartError("sample_file_missing", stage="upload")
    content = _SAMPLE_FILE.read_bytes()
    content_type, body = encode_multipart_file("files", _CITATION_FILE, content)
    idempotency = f"lkw-product-quickstart-{uuid.uuid4().hex}"
    headers = {
        **_tenant_headers(),
        "Idempotency-Key": idempotency,
        "Content-Type": content_type,
    }
    status, payload = http_post_bytes(
        f"{base_url}{_API_PREFIX}/workspaces/{workspace_id}/knowledge/files",
        body,
        headers,
        stage="upload",
    )
    if status != 202:
        raise QuickstartError("upload_http_failed", stage="upload")
    batch_status = str(payload.get("status", "")).strip().lower()
    accepted_count = response_integer(
        payload,
        "accepted_count",
        stage="upload",
    )
    failed_count = response_integer(payload, "failed_count", stage="upload")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise QuickstartError("upload_item_count_invalid", stage="upload")
    if batch_status not in {"accepted", "partial"}:
        raise QuickstartError("upload_status_invalid", stage="upload")
    if accepted_count != 1 or failed_count != 0:
        raise QuickstartError("upload_counts_invalid", stage="upload")
    item = items[0]
    if not isinstance(item, dict):
        raise QuickstartError("upload_item_invalid", stage="upload")
    operation_id = str(item.get("operation_id", "")).strip()
    source_id = str(item.get("source_id", "")).strip()
    if not operation_id or not source_id:
        raise QuickstartError("upload_operation_missing", stage="upload")
    return operation_id


def ask_workspace(base_url: str, workspace_id: str) -> dict[str, Any]:
    status, payload = http_post_json(
        f"{base_url}{_API_PREFIX}/workspaces/{workspace_id}/ask",
        {"question": _QUESTION, "limit": 5},
        _tenant_headers(),
        timeout=180.0,
        stage="ask",
    )
    if status != 200:
        raise QuickstartError("ask_http_failed", stage="ask")
    ask_status = str(payload.get("status", "")).strip().lower()
    if ask_status == "insufficient_evidence":
        raise QuickstartError("insufficient_evidence", stage="ask")
    if ask_status == "failed":
        raise QuickstartError("ask_failed", stage="ask")
    if ask_status != "completed":
        raise QuickstartError("ask_status_invalid", stage="ask")
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise QuickstartError("answer_empty", stage="ask")
    if _ANSWER_MARKER.lower() not in answer.lower():
        raise QuickstartError("answer_marker_missing", stage="ask")
    citations = payload.get("citations")
    if not isinstance(citations, list) or not citations:
        raise QuickstartError("citations_empty", stage="ask")
    citation_names = [
        str(item.get("file_name", "")).strip()
        for item in citations
        if isinstance(item, dict)
    ]
    if _CITATION_FILE not in citation_names:
        raise QuickstartError("citation_file_missing", stage="ask")
    run_id = str(payload.get("run_id", "")).strip()
    if not run_id:
        raise QuickstartError("run_id_missing", stage="ask")
    return payload


def verify_persisted_ask(base_url: str, run_id: str, workspace_id: str) -> None:
    payload = http_get_json(
        f"{base_url}{_API_PREFIX}/asks/{run_id}",
        _tenant_headers(),
        timeout=30.0,
        stage="persisted_read",
    )
    if str(payload.get("run_id", "")).strip() != run_id:
        raise QuickstartError("persisted_run_id_mismatch", stage="persisted_read")
    if str(payload.get("workspace_id", "")).strip() != workspace_id:
        raise QuickstartError("persisted_workspace_mismatch", stage="persisted_read")
    if str(payload.get("status", "")).strip().lower() != "completed":
        raise QuickstartError("persisted_status_invalid", stage="persisted_read")
    answer = str(payload.get("answer", "")).strip()
    if _ANSWER_MARKER.lower() not in answer.lower():
        raise QuickstartError("persisted_answer_marker_missing", stage="persisted_read")
    citations = payload.get("citations")
    if not isinstance(citations, list):
        raise QuickstartError("persisted_citations_invalid", stage="persisted_read")
    citation_names = [
        str(item.get("file_name", "")).strip()
        for item in citations
        if isinstance(item, dict)
    ]
    if _CITATION_FILE not in citation_names:
        raise QuickstartError("persisted_citation_missing", stage="persisted_read")


def emit_success(answer: str, workspace_id: str, run_id: str) -> None:
    print("LKW quickstart: PASS", flush=True)
    print(f"Question:\n{_QUESTION}", flush=True)
    print(f"Answer:\n{answer}", flush=True)
    print(f"Source:\n{_CITATION_FILE}", flush=True)
    print(f"Workspace:\n{workspace_id}", flush=True)
    print(f"Ask run:\n{run_id}", flush=True)
    print("Persisted Ask run verified:\nyes", flush=True)
    _print_kv("lkw_quickstart_result", "PASS")
    _print_kv("answer_marker", _ANSWER_MARKER)
    _print_kv("citation_file", _CITATION_FILE)
    _print_kv("persisted_run_verified", "true")
    _print_kv("stack_left_running", "true")
    print(
        "Stack remains running for inspection. See applications/local_workspace_application/docs/QUICKSTART.md "
        "for stop and troubleshooting commands.",
        flush=True,
    )


def run_quickstart(config: QuickstartConfig) -> int:
    current_stage = "preflight"
    try:
        validate_os_wrapper_pair(config.os_family, config.wrapper_id)
        base_url = validate_loopback_base_url(config.base_url)
        if not _SAMPLE_FILE.is_file():
            raise QuickstartError("sample_file_missing", stage="preflight")
        created_env = ensure_env_file()
        if created_env:
            print(
                "Created applications/local_workspace_application/.env from .env.example "
                "for local evaluation.",
                flush=True,
            )
        if not config.skip_stack_start:
            current_stage = "stack_start"
            if not _BOOTSTRAP_BAT.is_file() or not _BOOTSTRAP_SH.is_file():
                raise QuickstartError("bootstrap_script_missing", stage="preflight")
            completed = run_command(
                bootstrap_args(config.os_family),
                cwd=_REPO_ROOT,
                timeout=config.timeout_seconds,
                stage="stack_start",
            )
            if completed.returncode != 0:
                raise QuickstartError("stack_start_failed", stage="stack_start")
        current_stage = "health"
        wait_for_health(base_url, timeout_seconds=config.timeout_seconds)
        current_stage = "stack_start"
        model_name = resolve_ollama_embedding_model(
            timeout_seconds=config.timeout_seconds,
        )
        ensure_ollama_embedding_model(
            model_name,
            timeout_seconds=config.timeout_seconds,
        )
        current_stage = "workspace"
        workspace_id = create_workspace(base_url)
        current_stage = "upload"
        operation_id = upload_sample_file(base_url, workspace_id)
        current_stage = "ingestion"
        wait_for_operation(
            base_url,
            operation_id,
            _tenant_headers(),
            timeout_seconds=config.timeout_seconds,
        )
        current_stage = "ask"
        ask_payload = ask_workspace(base_url, workspace_id)
        run_id = str(ask_payload.get("run_id", "")).strip()
        answer = str(ask_payload.get("answer", "")).strip()
        current_stage = "persisted_read"
        verify_persisted_ask(base_url, run_id, workspace_id)
        _assert_safe_user_text(answer)
        emit_success(answer, workspace_id, run_id)
        return 0
    except QuickstartError as exc:
        _emit_failure(exc.stage, exc.reason)
        return 1
    except Exception:
        _emit_failure(current_stage, "unexpected_internal_error")
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LKW product quickstart runner")
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
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--timeout-seconds", type=int, default=_DEFAULT_TIMEOUT)
    parser.add_argument("--skip-stack-start", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        _emit_failure("preflight", "invalid_timeout")
        return 1
    config = QuickstartConfig(
        os_family=OsFamily(args.os_family),
        wrapper_id=WrapperId(args.wrapper_id),
        base_url=str(args.base_url),
        timeout_seconds=int(args.timeout_seconds),
        skip_stack_start=bool(args.skip_stack_start),
    )
    return run_quickstart(config)


if __name__ == "__main__":
    sys.exit(main())
