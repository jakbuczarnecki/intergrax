# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8C — real LKW foreground process proof."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STARTUP_DEADLINE_SECONDS = 60.0
_POLL_INTERVAL_SECONDS = 0.1
_LOG_TAIL_CHARS = 8000
_BOUNDARY_NAME = "local_workspace_hosting_boundary"


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _bounded_text(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= _LOG_TAIL_CHARS:
        return text
    return text[-_LOG_TAIL_CHARS:]


def _build_process_env(tmp_path: Path, port: int) -> dict[str, str]:
    home = tmp_path / "home"
    data_home = tmp_path / "lkw-data"
    sqlite = tmp_path / "sqlite"
    shadow = tmp_path / "shadow"
    workspace = tmp_path / "workspace"
    for path in (home, data_home, sqlite, shadow, workspace):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath_entries = [
        str(_REPO_ROOT),
        str(_REPO_ROOT / "agents"),
        str(_REPO_ROOT / "applications"),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["DATA_HOME"] = str(data_home)
    env["LKW_DATA_HOME"] = str(data_home)
    env["INTERGRAX_SQLITE_DATA_DIR"] = str(sqlite)
    env["INTERGRAX_SHADOW_ROOT"] = str(shadow)
    env["INTERGRAX_ALLOWED_READ_ROOTS"] = str(workspace)
    env["LOCAL_WORKSPACE_BACKEND_HOST"] = "127.0.0.1"
    env["LOCAL_WORKSPACE_BACKEND_PORT"] = str(port)
    env["LOCAL_WORKSPACE_VECTOR_STORE"] = "inmemory"
    env["LOCAL_WORKSPACE_ENABLE_RAG"] = "true"
    env["LOCAL_WORKSPACE_ENABLE_RAG_INGEST"] = "true"
    env["LOCAL_WORKSPACE_INCLUDE_MCP"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_SCHEDULER"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_INTERACTIONS"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL"] = "false"
    env["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "false"
    return env


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        return int(response.status), json.loads(body) if body else None


def _wait_until_ready(
    *,
    process: subprocess.Popen[str],
    port: int,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + _STARTUP_DEADLINE_SECONDS
    url = f"http://127.0.0.1:{port}/v1/local_workspace/readiness"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "first hosted process exited before READY\n"
                f"exit={process.returncode}\n"
                f"stdout:\n{_bounded_text(stdout_path)}\n"
                f"stderr:\n{_bounded_text(stderr_path)}"
            )
        try:
            status, body = _http_json("GET", url)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(exc)
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue
        if (
            status == 200
            and isinstance(body, dict)
            and body.get("ready") is True
            and body.get("accepts_new_work") is True
            and body.get("state") == "ready"
        ):
            return body
        last_error = f"status={status} body={body!r}"
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(
        "first hosted process did not reach READY before deadline\n"
        f"last_error={last_error}\n"
        f"stdout:\n{_bounded_text(stdout_path)}\n"
        f"stderr:\n{_bounded_text(stderr_path)}"
    )


def _assert_boundary_component(readiness: dict[str, Any]) -> None:
    components = readiness.get("components")
    assert isinstance(components, list)
    matches = [item for item in components if item.get("name") == _BOUNDARY_NAME]
    assert len(matches) == 1, components
    component = matches[0]
    assert component.get("enabled") is True
    assert component.get("required") is True
    assert component.get("healthy") is True
    assert component.get("detail") == "before_ready hook completed"


def _last_json_line(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines, f"expected JSON stdout, got empty:\n{text}"
    payload = json.loads(lines[-1])
    assert isinstance(payload, dict)
    return payload


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def test_hosted_foreground_process_ready_index_and_instance_conflict(
    tmp_path: Path,
) -> None:
    port = _reserve_free_port()
    env = _build_process_env(tmp_path, port)
    workspace = Path(env["INTERGRAX_ALLOWED_READ_ROOTS"])
    fixture_path = workspace / "hosted-proof-fixture.txt"
    fixture_path.write_text(
        "APP-HOST-8C hosted proof fixture text for local.workspace.index\n",
        encoding="utf-8",
    )

    stdout_path = tmp_path / "first-stdout.log"
    stderr_path = tmp_path / "first-stderr.log"
    command = [sys.executable, "-m", "local_workspace_application.hosting"]
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    first_process: subprocess.Popen[str] | None = None
    try:
        first_process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        readiness = _wait_until_ready(
            process=first_process,
            port=port,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        _assert_boundary_component(readiness)
        assert first_process.poll() is None

        status, body = _http_json(
            "POST",
            f"http://127.0.0.1:{port}/v1/local_workspace/run",
            {
                "tenant_id": "tenant-hosting-proof",
                "workspace_id": "workspace-hosting-proof",
                "message": "index hosted proof fixture",
                "capability": "local.workspace.index",
                "metadata": {
                    "source_paths": [str(fixture_path.resolve())],
                    "collection_id": "workspace-hosting-proof",
                },
            },
        )
        assert status == 200
        assert isinstance(body, dict)
        assert body.get("state") == "completed"
        metadata = body.get("metadata")
        assert isinstance(metadata, dict)
        summary = metadata.get("application_run_summary.v1")
        assert isinstance(summary, dict)
        assert "terminal_status" in summary

        second = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 2, (
            f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
        )
        payload = _last_json_line(second.stdout)
        assert payload["schema_version"] == "local_workspace.hosted_process_result.v1"
        assert payload["application_id"] == "local_workspace"
        final_exit = payload["final_exit"]
        assert final_exit["exit_kind"] == "instance_conflict"
        assert final_exit["reason_code"] == "instance_conflict"
        assert final_exit["retryable"] is False
        assert payload["restart_exhausted"] is False
        assert len(payload["attempts"]) == 1
        assert payload["attempts"][0]["attempt_number"] == 0
        assert payload["attempts"][0]["exit_kind"] == "instance_conflict"
        assert payload["attempts"][0]["reason_code"] == "instance_conflict"
        for forbidden in (
            "startup_failure",
            "runtime_failure",
            "configuration_error",
            "port conflict",
        ):
            assert forbidden not in json.dumps(payload)

        status, after = _http_json(
            "GET",
            f"http://127.0.0.1:{port}/v1/local_workspace/readiness",
        )
        assert status == 200
        assert isinstance(after, dict)
        assert after.get("ready") is True
        assert after.get("accepts_new_work") is True
        assert after.get("state") == "ready"
        _assert_boundary_component(after)
        assert first_process.poll() is None
    finally:
        if first_process is not None:
            _terminate_process(first_process)
        stdout_handle.close()
        stderr_handle.close()
