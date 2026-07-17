# © Artur Czarnecki. All rights reserved.

"""LKW.6C — real Windows PowerShell interaction adapter live proof."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        os.name != "nt",
        reason="Windows PowerShell interaction proof requires Windows",
    ),
]


def _install_windows_console_ctrl_shield() -> None:
    """Keep the pytest process alive when CTRL_BREAK is delivered on a shared console."""
    if os.name != "nt":
        return
    import ctypes

    if getattr(_install_windows_console_ctrl_shield, "_installed", False):
        return

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    def _handler(ctrl_type: int) -> bool:
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
_ADAPTER_SCRIPT = (
    _REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "scripts"
    / "invoke-lkw-interaction.ps1"
)
_STARTUP_DEADLINE_SECONDS = 90.0
_SHUTDOWN_DEADLINE_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.1
_LOG_TAIL_CHARS = 8000
_BOUNDARY_NAME = "local_workspace_hosting_boundary"
_ADAPTER_SCHEMA = "local_workspace.windows_interaction_adapter_result.v1"
_ADAPTER_ID = "lkw.windows_powershell"
_COLLECTION_ID = "lkw-windows-interaction-proof"
_TENANT_ID = "tenant-windows-interaction-proof"
_USER_ID = "user-windows-interaction-proof"


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
    env["LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_INTERACTIONS"] = "true"
    env["LOCAL_WORKSPACE_INTERACTION_SURFACE"] = "lab_json"
    env["LOCAL_WORKSPACE_INTERACTION_EXECUTE_DEFAULT"] = "true"
    env["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "false"
    return env


def _subprocess_group_kwargs() -> dict[str, Any]:
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def _send_graceful_stop_signal(process: subprocess.Popen[str]) -> str:
    sigbreak = getattr(signal, "SIGBREAK", None)
    previous = signal.signal(sigbreak, signal.SIG_IGN) if sigbreak is not None else None
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    finally:
        if sigbreak is not None and previous is not None:
            signal.signal(sigbreak, previous)
    return "signal.sigbreak"


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
) -> dict[str, Any]:
    deadline = time.monotonic() + _STARTUP_DEADLINE_SECONDS
    url = f"http://127.0.0.1:{port}/v1/local_workspace/readiness"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "hosted process exited before READY\n"
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
        "hosted process did not reach READY before deadline\n"
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


def _assert_interaction_component_when_present(readiness: dict[str, Any]) -> None:
    """Hosted readiness projects platform hosting components only.

    Direct-mode LKW registers ``interaction_intake`` on LocalWorkspaceHostLifecycle.
    Hosted mode exposes the hosting boundary component instead; successful intake
    execution is the interaction-surface proof when the optional component is absent.
    """
    components = readiness.get("components")
    assert isinstance(components, list)
    matches = [item for item in components if item.get("name") == "interaction_intake"]
    if not matches:
        return
    component = matches[0]
    assert component.get("enabled") is True
    assert component.get("healthy") is True


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
) -> int:
    deadline = time.monotonic() + _SHUTDOWN_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            return int(code)
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(
        "hosted process did not exit after graceful signal before deadline\n"
        f"stdout:\n{_bounded_text(stdout_path)}\n"
        f"stderr:\n{_bounded_text(stderr_path)}"
    )


def _cleanup_process(process: subprocess.Popen[str] | None) -> None:
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


def _assert_clean_stop_payload(payload: dict[str, Any]) -> None:
    assert payload["schema_version"] == "local_workspace.hosted_process_result.v1"
    assert payload["application_id"] == "local_workspace"
    final_exit = payload["final_exit"]
    assert final_exit["exit_kind"] == "clean_stop"
    assert final_exit["reason_code"] == "signal.sigbreak"
    assert final_exit["terminal_lifecycle_state"] == "stopped"
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["cleanup_verified"] is True


def _invoke_powershell_adapter(
    *,
    port: int,
    message: str,
    capability: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    powershell = _require_powershell()
    command = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(_ADAPTER_SCRIPT),
        "-BaseUrl",
        f"http://127.0.0.1:{port}",
        "-Message",
        message,
        "-Capability",
        capability,
        "-TenantId",
        _TENANT_ID,
        "-UserId",
        _USER_ID,
        "-MetadataJson",
        json.dumps(metadata, ensure_ascii=False),
    ]
    completed = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, (
        f"adapter exit={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    payload = _last_json_line(completed.stdout)
    assert payload.get("schema_version") == _ADAPTER_SCHEMA
    assert payload.get("adapter_id") == _ADAPTER_ID
    assert payload.get("endpoint") == "/v1/interactions/intake"
    assert payload.get("execute") is True
    response = payload.get("response")
    assert isinstance(response, dict)
    return response


def _require_powershell() -> str:
    resolved = shutil.which("powershell.exe")
    if not resolved:
        raise AssertionError("powershell.exe is required for Windows interaction proof")
    return resolved


def test_windows_powershell_adapter_executes_real_lkw_interactions(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    _require_powershell()
    assert _ADAPTER_SCRIPT.is_file(), f"missing adapter script: {_ADAPTER_SCRIPT}"

    port = _reserve_free_port()
    env = _build_process_env(tmp_path, port)
    workspace = Path(env["INTERGRAX_ALLOWED_READ_ROOTS"])
    marker_suffix = uuid.uuid4().hex[:12]
    marker = f"LKW_WINDOWS_INTERACTION_PROOF_{marker_suffix}"
    fixture_path = workspace / f"windows-interaction-proof-{marker_suffix}.txt"
    fixture_path.write_text(
        f"Windows PowerShell interaction proof fixture\n{marker}\n",
        encoding="utf-8",
    )

    stdout_path = tmp_path / "hosted-stdout.log"
    stderr_path = tmp_path / "hosted-stderr.log"
    command = [sys.executable, "-m", "local_workspace_application.hosting"]
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            **_subprocess_group_kwargs(),
        )
        readiness = _wait_until_ready(
            process=process,
            port=port,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        _assert_boundary_component(readiness)
        _assert_interaction_component_when_present(readiness)
        assert process.poll() is None

        index_response = _invoke_powershell_adapter(
            port=port,
            message="index Windows PowerShell interaction proof fixture",
            capability="local.workspace.index",
            metadata={
                "source_paths": [str(fixture_path.resolve())],
                "collection_id": _COLLECTION_ID,
            },
        )
        assert index_response.get("tenant_id") == _TENANT_ID
        assert index_response.get("user_id") == _USER_ID
        assert index_response.get("capability") == "local.workspace.index"
        assert index_response.get("interaction_channel") == "lab"
        assert index_response.get("executed") is True
        assert index_response.get("state") == "completed"
        index_task_id = str(index_response.get("task_id") or "").strip()
        index_run_id = str(index_response.get("run_id") or "").strip()
        assert index_task_id
        assert index_run_id

        search_response = _invoke_powershell_adapter(
            port=port,
            message=marker,
            capability="local.workspace.search",
            metadata={
                "query": marker,
                "collection_id": _COLLECTION_ID,
                "top_k": 5,
            },
        )
        assert search_response.get("tenant_id") == _TENANT_ID
        assert search_response.get("user_id") == _USER_ID
        assert search_response.get("capability") == "local.workspace.search"
        assert search_response.get("interaction_channel") == "lab"
        assert search_response.get("executed") is True
        assert search_response.get("state") == "completed"
        search_task_id = str(search_response.get("task_id") or "").strip()
        search_run_id = str(search_response.get("run_id") or "").strip()
        assert search_task_id
        assert search_run_id
        assert search_task_id != index_task_id
        assert search_run_id != index_run_id

        _send_graceful_stop_signal(process)
        exit_code = _wait_for_process_exit(
            process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stdout_handle.flush()
        stderr_handle.flush()
        assert exit_code == 0, (
            f"stdout:\n{_bounded_text(stdout_path)}\n"
            f"stderr:\n{_bounded_text(stderr_path)}"
        )
        stop_payload = _last_json_line(_bounded_text(stdout_path))
        _assert_clean_stop_payload(stop_payload)

        record_property("windows_interaction.hosted_ready", "true")
        record_property("windows_interaction.adapter_invoked", "true")
        record_property("windows_interaction.adapter_id", _ADAPTER_ID)
        record_property("windows_interaction.powershell_runtime", "Windows PowerShell")
        record_property("windows_interaction.transport", "http")
        record_property(
            "windows_interaction.intake_endpoint", "/v1/interactions/intake"
        )
        record_property("windows_interaction.interaction_surface", "lab_json")
        record_property("windows_interaction.interaction_channel", "lab")
        record_property("windows_interaction.index_executed", "true")
        record_property("windows_interaction.index_state", "completed")
        record_property("windows_interaction.index_task_id", index_task_id)
        record_property("windows_interaction.index_run_id", index_run_id)
        record_property("windows_interaction.search_executed", "true")
        record_property("windows_interaction.search_state", "completed")
        record_property("windows_interaction.search_task_id", search_task_id)
        record_property("windows_interaction.search_run_id", search_run_id)
        record_property("windows_interaction.task_ids_distinct", "true")
        record_property("windows_interaction.run_ids_distinct", "true")
        record_property("windows_interaction.graceful_stop", "true")
        record_property("windows_interaction.cleanup_verified", "true")
    finally:
        _cleanup_process(process)
        stdout_handle.close()
        stderr_handle.close()
