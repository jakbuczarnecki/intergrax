#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Controlled live proof for Trusted Ask Workspace (MVP-2).

Validates:
real host → sync → POST /ask → grounded answer + citations → stop → restart → GET /asks/{run_id}
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
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
_AGENTS_ROOT = _REPO_ROOT / "agents"
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"


def _print_kv(key: str, value: object) -> None:
    print(f"{key}={value}")


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _resolve_host_mongodb_uri() -> str:
    explicit = os.environ.get("INTERGRAX_MONGODB_URI", "").strip()
    if explicit:
        return explicit
    username = os.environ.get("LKW_MONGODB_ROOT_USERNAME", "intergrax").strip() or "intergrax"
    password = (
        os.environ.get("LKW_MONGODB_ROOT_PASSWORD", "intergrax-local-dev-only").strip()
        or "intergrax-local-dev-only"
    )
    database = os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip() or "intergrax_proofs"
    host_port = os.environ.get("LKW_MONGODB_HOST_PORT", "27018").strip() or "27018"
    return (
        f"mongodb://{username}:{password}@127.0.0.1:{host_port}/{database}?authSource=admin"
    )


def ensure_mongodb_env() -> None:
    os.environ["INTERGRAX_MONGODB_URI"] = _resolve_host_mongodb_uri()
    os.environ.setdefault("LKW_MANAGED_WORKSPACE_COLLECTION", "lkw_managed_workspaces")


def _docker_compose(*args: str) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        str(_BASE_COMPOSE),
        "-f",
        str(_MONGODB_COMPOSE),
        *args,
    ]
    completed = subprocess.run(command, cwd=str(_REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"docker_compose_failed:{' '.join(args)}")


def start_mongodb_if_needed(*, skip_docker: bool) -> None:
    if skip_docker:
        return
    _docker_compose("up", "-d", "lkw-mongodb")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(_BASE_COMPOSE),
                "-f",
                str(_MONGODB_COMPOSE),
                "exec",
                "-T",
                "lkw-mongodb",
                "mongosh",
                "--quiet",
                "--eval",
                "db.runCommand({ ping: 1 }).ok",
            ],
            cwd=str(_REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and "1" in (probe.stdout or ""):
            return
        time.sleep(1.0)
    raise RuntimeError("mongodb_not_ready")


def build_host_env(
    *,
    port: int,
    data_home: Path,
    allow_root: Path,
    shadow_root: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(_REPO_ROOT), str(_AGENTS_ROOT), str(_APPLICATIONS_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["DATA_HOME"] = str(data_home)
    env["LKW_DATA_HOME"] = str(data_home)
    env["INTERGRAX_SQLITE_DATA_DIR"] = str(data_home / "sqlite")
    env["INTERGRAX_SHADOW_ROOT"] = str(shadow_root)
    env["INTERGRAX_ALLOWED_READ_ROOTS"] = str(allow_root)
    env["LOCAL_WORKSPACE_BACKEND_HOST"] = "127.0.0.1"
    env["LOCAL_WORKSPACE_BACKEND_PORT"] = str(port)
    env["LOCAL_WORKSPACE_VECTOR_STORE"] = "inmemory"
    env["LOCAL_WORKSPACE_ENABLE_RAG"] = "true"
    env["LOCAL_WORKSPACE_ENABLE_RAG_INGEST"] = "true"
    env["LOCAL_WORKSPACE_INCLUDE_MCP"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_SCHEDULER"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_INTERACTIONS"] = "false"
    env["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "false"
    env["INTERGRAX_MONGODB_URI"] = _resolve_host_mongodb_uri()
    env["LKW_MANAGED_WORKSPACE_COLLECTION"] = "lkw_managed_workspaces"
    env["INTERGRAX_DOCUMENT_STORE_TASK_WORKER_START_DELAY_SECONDS"] = "0"
    env.setdefault("INTERGRAX_LLM_PROVIDER", "ollama")
    env.setdefault(
        "INTERGRAX_LLM_MODEL",
        os.environ.get("INTERGRAX_DEFAULT_OLLAMA_MODEL", "qwen2.5-coder:latest"),
    )
    env.setdefault("INTERGRAX_DEFAULT_OLLAMA_MODEL", env["INTERGRAX_LLM_MODEL"])
    return env


def _subprocess_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        sigbreak = getattr(signal, "SIGBREAK", None)
        previous = signal.signal(sigbreak, signal.SIG_IGN) if sigbreak is not None else None
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        finally:
            if sigbreak is not None and previous is not None:
                signal.signal(sigbreak, previous)
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_ready(
    process: subprocess.Popen[str],
    port: int,
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float = 180.0,
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/local_workspace/readiness"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout_tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"host_exited:{process.returncode}:stdout={stdout_tail!r}:stderr={stderr_tail!r}"
            )
        try:
            status, body = _request_json(url, timeout=5.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.5)
            continue
        if status == 200 and body.get("ready") is True and body.get("accepts_new_work") is True:
            return
        time.sleep(0.5)
    raise RuntimeError("host_not_ready")


def wait_operation(
    base_url: str,
    operation_id: str,
    *,
    tenant_id: str,
    timeout: float = 180.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        status, body = _request_json(
            f"{base_url}/v1/local_workspace/operations/{operation_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200:
            raise RuntimeError(f"operation_status_http_{status}:{body}")
        last = body
        if body.get("status") in {"completed", "failed"}:
            return body
        time.sleep(0.5)
    raise RuntimeError(f"operation_timeout:{last}")


def _start_host(
    *,
    port: int,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen[str]:
    host_command = [
        "uv",
        "run",
        "--extra",
        "integrations-mongodb",
        "python",
        "-m",
        "local_workspace_application.hosting",
    ]
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        host_command,
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        **_subprocess_group_kwargs(),
    )
    wait_ready(process, port, stdout_path=stdout_path, stderr_path=stderr_path)
    return process


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    ensure_mongodb_env()
    suffix = secrets.token_hex(4)
    marker = f"ask workspace payment term {suffix}"
    tenant_id = "lkw-ask-workspace-live"
    port = _reserve_free_port()
    base_url = f"http://127.0.0.1:{port}"
    temp_root = Path(tempfile.mkdtemp(prefix="lkw-ask-workspace-"))
    source_dir = temp_root / "source"
    data_home = temp_root / "data_home"
    shadow_root = temp_root / "shadow"
    source_dir.mkdir(parents=True)
    data_home.mkdir(parents=True)
    shadow_root.mkdir(parents=True)

    doc = source_dir / "payment-terms.txt"
    doc.write_text(
        (
            "Buildlogic supplier agreement.\n"
            f"Payment is due within 14 days of invoice. Marker: {marker}.\n"
            "Late fees accrue after the due date.\n"
        ),
        encoding="utf-8",
    )

    process: subprocess.Popen[str] | None = None
    stdout_path = temp_root / "host-stdout.log"
    stderr_path = temp_root / "host-stderr.log"
    restart_stdout = temp_root / "host-restart-stdout.log"
    restart_stderr = temp_root / "host-restart-stderr.log"

    try:
        start_mongodb_if_needed(skip_docker=args.skip_docker)
        env = build_host_env(
            port=port,
            data_home=data_home,
            allow_root=source_dir,
            shadow_root=shadow_root,
        )
        process = _start_host(
            port=port,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        status, workspace = _request_json(
            f"{base_url}/v1/local_workspace/workspaces",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"name": "Ask Live Proof Workspace"},
        )
        if status != 201:
            raise RuntimeError(f"workspace_create_failed:{status}:{workspace}")
        workspace_id = str(workspace["workspace_id"])

        status, source = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={
                "source_type": "local_folder",
                "path": str(source_dir.resolve()),
                "recursive": True,
            },
        )
        if status != 201:
            raise RuntimeError(f"source_register_failed:{status}:{source}")
        source_id = str(source["source_id"])

        status, sync = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources/{source_id}/sync",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 202:
            raise RuntimeError(f"sync_failed:{status}:{sync}")
        operation = wait_operation(
            base_url,
            str(sync["operation_id"]),
            tenant_id=tenant_id,
        )
        if operation.get("status") != "completed":
            raise RuntimeError(f"sync_not_completed:{operation}")

        status, search = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/search",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"query": marker, "limit": 5},
        )
        if status != 200:
            raise RuntimeError(f"search_failed:{status}:{search}")
        evidence_count = len(search.get("results") or [])
        if evidence_count < 1:
            raise RuntimeError(f"search_empty:{search}")

        status, ask = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/ask",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={
                "question": f"According to the documents, what is the payment term related to {marker}?",
                "limit": 5,
            },
            timeout=180.0,
        )
        if status != 200:
            raise RuntimeError(f"ask_failed:{status}:{ask}")
        if ask.get("status") != "completed":
            raise RuntimeError(f"ask_not_completed:{ask}")
        if not ask.get("answer"):
            raise RuntimeError(f"ask_missing_answer:{ask}")
        citations = ask.get("citations") or []
        if not isinstance(citations, list) or not citations:
            raise RuntimeError(f"ask_missing_citations:{ask}")
        run_id = str(ask["run_id"])
        citation_count = len(citations)

        # Stop host and restart against the same durable DocumentStore.
        _stop_process(process)
        process = None
        time.sleep(1.0)
        process = _start_host(
            port=port,
            env=env,
            stdout_path=restart_stdout,
            stderr_path=restart_stderr,
        )

        status, restarted = _request_json(
            f"{base_url}/v1/local_workspace/asks/{run_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200:
            raise RuntimeError(f"restart_read_failed:{status}:{restarted}")
        if restarted.get("run_id") != run_id:
            raise RuntimeError(f"restart_run_id_mismatch:{restarted}")
        if restarted.get("status") != "completed":
            raise RuntimeError(f"restart_status_mismatch:{restarted}")
        if restarted.get("answer") != ask.get("answer"):
            raise RuntimeError(f"restart_answer_mismatch:{restarted}")
        if len(restarted.get("citations") or []) != citation_count:
            raise RuntimeError(f"restart_citation_mismatch:{restarted}")

        _print_kv("workspace_id", workspace_id)
        _print_kv("run_id", run_id)
        _print_kv("ask_status", ask.get("status"))
        _print_kv("evidence_count", evidence_count)
        _print_kv("citation_count", citation_count)
        _print_kv("restart_read_result", "ok")
        _print_kv("proof_result", "PASS")
        return 0
    except Exception as exc:
        _print_kv("proof_result", "FAIL")
        _print_kv("error", f"{exc.__class__.__name__}: {exc}")
        if stdout_path.exists():
            _print_kv("stdout_tail", stdout_path.read_text(encoding="utf-8", errors="replace")[-2000:])
        if stderr_path.exists():
            _print_kv("stderr_tail", stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:])
        if restart_stderr.exists():
            _print_kv(
                "restart_stderr_tail",
                restart_stderr.read_text(encoding="utf-8", errors="replace")[-2000:],
            )
        return 1
    finally:
        if process is not None:
            _stop_process(process)
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
