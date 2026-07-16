# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8C/8D — real LKW foreground process proofs."""

from __future__ import annotations

import json
import os
import signal
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


def _install_windows_console_ctrl_shield() -> None:
    """Keep the pytest process alive when CTRL_BREAK is delivered on a shared console."""
    if os.name != "nt":
        return
    import ctypes

    if getattr(_install_windows_console_ctrl_shield, "_installed", False):
        return

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    def _handler(ctrl_type: int) -> bool:
        # 0 = CTRL_C_EVENT, 1 = CTRL_BREAK_EVENT
        return ctrl_type in (0, 1)

    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True):
        return
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signal.signal(sigbreak, signal.SIG_IGN)
    setattr(_install_windows_console_ctrl_shield, "_installed", True)
    setattr(_install_windows_console_ctrl_shield, "_handler", _handler)


_install_windows_console_ctrl_shield()

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STARTUP_DEADLINE_SECONDS = 60.0
_SHUTDOWN_DEADLINE_SECONDS = 30.0
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


def _subprocess_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _send_graceful_stop_signal(
    process: subprocess.Popen[str],
) -> str:
    if os.name == "nt":
        # Ignore SIGBREAK in the test process so GenerateConsoleCtrlEvent does
        # not terminate the pytest parent while targeting the child process group.
        sigbreak = getattr(signal, "SIGBREAK", None)
        previous = (
            signal.signal(sigbreak, signal.SIG_IGN) if sigbreak is not None else None
        )
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        finally:
            if sigbreak is not None and previous is not None:
                signal.signal(sigbreak, previous)
        return "signal.sigbreak"
    process.send_signal(signal.SIGTERM)
    return "signal.sigterm"


def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> tuple[int, Any]:
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
    label: str = "hosted process",
) -> dict[str, Any]:
    deadline = time.monotonic() + _STARTUP_DEADLINE_SECONDS
    url = f"http://127.0.0.1:{port}/v1/local_workspace/readiness"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"{label} exited before READY\n"
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
        f"{label} did not reach READY before deadline\n"
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


def _wait_for_process_exit(
    process: subprocess.Popen[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    label: str,
) -> int:
    deadline = time.monotonic() + _SHUTDOWN_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            return int(code)
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(
        f"{label} did not exit after graceful signal before deadline\n"
        f"stdout:\n{_bounded_text(stdout_path)}\n"
        f"stderr:\n{_bounded_text(stderr_path)}"
    )


def _cleanup_process(
    process: subprocess.Popen[str] | None,
) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        _send_graceful_stop_signal(process)
        process.wait(timeout=10)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    _cleanup_process(process)


def _assert_clean_stop_payload(
    payload: dict[str, Any],
    *,
    expected_reason: str,
) -> None:
    assert payload["schema_version"] == "local_workspace.hosted_process_result.v1"
    assert payload["application_id"] == "local_workspace"
    final_exit = payload["final_exit"]
    assert final_exit["exit_kind"] == "clean_stop"
    assert final_exit["reason_code"] == expected_reason
    assert final_exit["retryable"] is False
    assert final_exit["terminal_lifecycle_state"] == "stopped"
    assert payload["restart_exhausted"] is False
    assert len(payload["attempts"]) == 1
    attempt = payload["attempts"][0]
    assert attempt["attempt_number"] == 0
    assert attempt["exit_kind"] == "clean_stop"
    assert attempt["reason_code"] == expected_reason
    assert attempt["cleanup_verified"] is True


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
            **_subprocess_group_kwargs(),
        )
        readiness = _wait_until_ready(
            process=first_process,
            port=port,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label="first hosted process",
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


def test_hosted_foreground_process_graceful_stop_releases_instance_lock(
    tmp_path: Path,
) -> None:
    port = _reserve_free_port()
    env = _build_process_env(tmp_path, port)
    command = [sys.executable, "-m", "local_workspace_application.hosting"]
    group_kwargs = _subprocess_group_kwargs()

    first_stdout = tmp_path / "stop-first-stdout.log"
    first_stderr = tmp_path / "stop-first-stderr.log"
    second_stdout = tmp_path / "stop-second-stdout.log"
    second_stderr = tmp_path / "stop-second-stderr.log"

    first_out = first_stdout.open("w", encoding="utf-8")
    first_err = first_stderr.open("w", encoding="utf-8")
    second_out = second_stdout.open("w", encoding="utf-8")
    second_err = second_stderr.open("w", encoding="utf-8")

    first_process: subprocess.Popen[str] | None = None
    second_process: subprocess.Popen[str] | None = None
    try:
        first_process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=first_out,
            stderr=first_err,
            text=True,
            **group_kwargs,
        )
        readiness = _wait_until_ready(
            process=first_process,
            port=port,
            stdout_path=first_stdout,
            stderr_path=first_stderr,
            label="first hosted process",
        )
        _assert_boundary_component(readiness)
        assert first_process.poll() is None

        expected_reason = _send_graceful_stop_signal(first_process)
        exit_code = _wait_for_process_exit(
            first_process,
            stdout_path=first_stdout,
            stderr_path=first_stderr,
            label="first hosted process",
        )
        first_out.flush()
        first_err.flush()
        assert exit_code == 0, (
            f"stdout:\n{_bounded_text(first_stdout)}\n"
            f"stderr:\n{_bounded_text(first_stderr)}"
        )
        first_payload = _last_json_line(_bounded_text(first_stdout))
        _assert_clean_stop_payload(first_payload, expected_reason=expected_reason)

        second_process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=second_out,
            stderr=second_err,
            text=True,
            **group_kwargs,
        )
        second_ready = _wait_until_ready(
            process=second_process,
            port=port,
            stdout_path=second_stdout,
            stderr_path=second_stderr,
            label="replacement hosted process",
        )
        _assert_boundary_component(second_ready)
        assert second_process.poll() is None

        second_reason = _send_graceful_stop_signal(second_process)
        second_exit = _wait_for_process_exit(
            second_process,
            stdout_path=second_stdout,
            stderr_path=second_stderr,
            label="replacement hosted process",
        )
        second_out.flush()
        second_err.flush()
        assert second_exit == 0, (
            f"stdout:\n{_bounded_text(second_stdout)}\n"
            f"stderr:\n{_bounded_text(second_stderr)}"
        )
        second_payload = _last_json_line(_bounded_text(second_stdout))
        assert second_payload["final_exit"]["exit_kind"] == "clean_stop"
        assert second_payload["final_exit"]["reason_code"] == second_reason
        assert second_payload["attempts"][0]["cleanup_verified"] is True
    finally:
        _cleanup_process(first_process)
        _cleanup_process(second_process)
        first_out.close()
        first_err.close()
        second_out.close()
        second_err.close()
