# © Artur Czarnecki. All rights reserved.

"""Shared helpers for LKW OS interaction live proofs."""

from __future__ import annotations

import json
import os
import platform
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

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _REPO_ROOT / "applications" / "local_workspace_application" / "scripts"
_STARTUP_DEADLINE_SECONDS = 90.0
_SHUTDOWN_DEADLINE_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.1
_LOG_TAIL_CHARS = 8000
_BOUNDARY_NAME = "local_workspace_hosting_boundary"
_ADAPTER_SCHEMA = "local_workspace.os_interaction_adapter_result.v1"
_INTAKE_ENDPOINT = "/v1/interactions/intake"
_CLIENT_RUNTIME = "python"

ADAPTER_SCRIPTS = {
    "windows": _SCRIPTS_DIR / "invoke-lkw-interaction.ps1",
    "linux": _SCRIPTS_DIR / "invoke-lkw-interaction-linux.sh",
    "macos": _SCRIPTS_DIR / "invoke-lkw-interaction-macos.sh",
}

OS_CONTRACTS = {
    "windows": {
        "os_family": "windows",
        "adapter_id": "lkw.windows_powershell",
        "source": "windows_powershell",
        "wrapper_runtime": "windows_powershell",
        "collection_id": "lkw-windows-interaction-proof",
        "tenant_id": "tenant-windows-interaction-proof",
        "user_id": "user-windows-interaction-proof",
        "marker_prefix": "LKW_WINDOWS_INTERACTION_PROOF",
        "fixture_prefix": "windows-interaction-proof",
    },
    "linux": {
        "os_family": "linux",
        "adapter_id": "lkw.linux_shell",
        "source": "linux_shell",
        "wrapper_runtime": "posix_sh",
        "collection_id": "lkw-linux-interaction-proof",
        "tenant_id": "tenant-linux-interaction-proof",
        "user_id": "user-linux-interaction-proof",
        "marker_prefix": "LKW_LINUX_INTERACTION_PROOF",
        "fixture_prefix": "linux-interaction-proof",
    },
    "macos": {
        "os_family": "macos",
        "adapter_id": "lkw.macos_shell",
        "source": "macos_shell",
        "wrapper_runtime": "posix_sh",
        "collection_id": "lkw-macos-interaction-proof",
        "tenant_id": "tenant-macos-interaction-proof",
        "user_id": "user-macos-interaction-proof",
        "marker_prefix": "LKW_MACOS_INTERACTION_PROOF",
        "fixture_prefix": "macos-interaction-proof",
    },
}


def detect_runtime_os_family() -> str:
    mapping = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}
    detected = mapping.get(platform.system())
    if detected is None:
        raise AssertionError(f"unsupported_runtime_os:{platform.system()}")
    return detected


def install_windows_console_ctrl_shield() -> None:
    """Keep the pytest process alive when CTRL_BREAK is delivered on a shared console."""
    if os.name != "nt":
        return
    import ctypes

    if getattr(install_windows_console_ctrl_shield, "_installed", False):
        return

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    def _handler(ctrl_type: int) -> bool:
        return ctrl_type in (0, 1)

    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True):
        return
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signal.signal(sigbreak, signal.SIG_IGN)
    setattr(install_windows_console_ctrl_shield, "_installed", True)
    setattr(install_windows_console_ctrl_shield, "_handler", _handler)


def reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def bounded_text(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= _LOG_TAIL_CHARS:
        return text
    return text[-_LOG_TAIL_CHARS:]


def build_process_env(tmp_path: Path, port: int) -> dict[str, str]:
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


def subprocess_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def send_graceful_stop_signal(process: subprocess.Popen[str]) -> str:
    if os.name == "nt":
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


def http_json(
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


def wait_until_ready(
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
                f"stdout:\n{bounded_text(stdout_path)}\n"
                f"stderr:\n{bounded_text(stderr_path)}"
            )
        try:
            status, body = http_json("GET", url)
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
        f"stdout:\n{bounded_text(stdout_path)}\n"
        f"stderr:\n{bounded_text(stderr_path)}"
    )


def assert_boundary_component(readiness: dict[str, Any]) -> None:
    components = readiness.get("components")
    assert isinstance(components, list)
    matches = [item for item in components if item.get("name") == _BOUNDARY_NAME]
    assert len(matches) == 1, components
    component = matches[0]
    assert component.get("enabled") is True
    assert component.get("required") is True
    assert component.get("healthy") is True


def assert_interaction_component_when_present(readiness: dict[str, Any]) -> None:
    components = readiness.get("components")
    assert isinstance(components, list)
    matches = [item for item in components if item.get("name") == "interaction_intake"]
    if not matches:
        return
    component = matches[0]
    assert component.get("enabled") is True
    assert component.get("healthy") is True


def last_json_line(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines, f"expected JSON stdout, got empty:\n{text}"
    payload = json.loads(lines[-1])
    assert isinstance(payload, dict)
    return payload


def wait_for_process_exit(
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
        f"stdout:\n{bounded_text(stdout_path)}\n"
        f"stderr:\n{bounded_text(stderr_path)}"
    )


def cleanup_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        send_graceful_stop_signal(process)
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


def assert_clean_stop_payload(payload: dict[str, Any], *, expected_reason: str) -> None:
    assert payload["schema_version"] == "local_workspace.hosted_process_result.v1"
    assert payload["application_id"] == "local_workspace"
    final_exit = payload["final_exit"]
    assert final_exit["exit_kind"] == "clean_stop"
    assert final_exit["reason_code"] == expected_reason
    assert final_exit["terminal_lifecycle_state"] == "stopped"
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["cleanup_verified"] is True


def _build_adapter_command(
    *,
    os_family: str,
    adapter_script: Path,
    port: int,
    message: str,
    capability: str,
    metadata: dict[str, Any],
    tenant_id: str,
    user_id: str,
) -> list[str]:
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    base_url = f"http://127.0.0.1:{port}"
    if os_family == "windows":
        import shutil

        powershell = shutil.which("powershell.exe")
        if not powershell:
            raise AssertionError(
                "powershell.exe is required for Windows interaction proof"
            )
        return [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(adapter_script),
            "-BaseUrl",
            base_url,
            "-Message",
            message,
            "-Capability",
            capability,
            "-TenantId",
            tenant_id,
            "-UserId",
            user_id,
            "-MetadataJson",
            metadata_json,
        ]
    return [
        "sh",
        str(adapter_script),
        "--base-url",
        base_url,
        "--message",
        message,
        "--capability",
        capability,
        "--tenant-id",
        tenant_id,
        "--user-id",
        user_id,
        "--metadata-json",
        metadata_json,
    ]


def invoke_os_interaction_adapter(
    *,
    os_family: str,
    port: int,
    message: str,
    capability: str,
    metadata: dict[str, Any],
    tenant_id: str,
    user_id: str,
) -> dict[str, Any]:
    contract = OS_CONTRACTS[os_family]
    adapter_script = ADAPTER_SCRIPTS[os_family]
    assert adapter_script.is_file(), f"missing adapter script: {adapter_script}"
    command = _build_adapter_command(
        os_family=os_family,
        adapter_script=adapter_script,
        port=port,
        message=message,
        capability=capability,
        metadata=metadata,
        tenant_id=tenant_id,
        user_id=user_id,
    )
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
    payload = last_json_line(completed.stdout)
    assert payload.get("schema_version") == _ADAPTER_SCHEMA
    assert payload.get("result") == "PASS"
    assert payload.get("adapter_id") == contract["adapter_id"]
    assert payload.get("os_family") == os_family
    assert payload.get("source") == contract["source"]
    assert payload.get("wrapper_runtime") == contract["wrapper_runtime"]
    assert payload.get("client_runtime") == _CLIENT_RUNTIME
    assert payload.get("endpoint") == _INTAKE_ENDPOINT
    assert payload.get("execute") is True
    response = payload.get("response")
    assert isinstance(response, dict)
    return response


def record_os_interaction_evidence(
    record_property: Callable[[str, object], None],
    *,
    os_family: str,
    index_task_id: str,
    index_run_id: str,
    search_task_id: str,
    search_run_id: str,
) -> None:
    contract = OS_CONTRACTS[os_family]
    record_property("os_interaction.hosted_ready", "true")
    record_property("os_interaction.adapter_invoked", "true")
    record_property("os_interaction.os_family", os_family)
    record_property("os_interaction.os_version", platform.version())
    record_property(
        "os_interaction.architecture",
        platform.machine() or platform.architecture()[0],
    )
    record_property("os_interaction.client_runtime", _CLIENT_RUNTIME)
    record_property("os_interaction.wrapper_runtime", contract["wrapper_runtime"])
    record_property("os_interaction.adapter_id", contract["adapter_id"])
    record_property("os_interaction.source", contract["source"])
    record_property("os_interaction.transport", "http")
    record_property("os_interaction.intake_endpoint", _INTAKE_ENDPOINT)
    record_property("os_interaction.interaction_surface", "lab_json")
    record_property("os_interaction.interaction_channel", "lab")
    record_property("os_interaction.index_executed", "true")
    record_property("os_interaction.index_state", "completed")
    record_property("os_interaction.index_task_id", index_task_id)
    record_property("os_interaction.index_run_id", index_run_id)
    record_property("os_interaction.search_executed", "true")
    record_property("os_interaction.search_state", "completed")
    record_property("os_interaction.search_task_id", search_task_id)
    record_property("os_interaction.search_run_id", search_run_id)
    record_property("os_interaction.task_ids_distinct", "true")
    record_property("os_interaction.run_ids_distinct", "true")
    record_property("os_interaction.graceful_stop", "true")
    record_property("os_interaction.cleanup_verified", "true")


def run_os_interaction_live_proof(
    *,
    os_family: str,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Execute the shared hosted index/search interaction proof for one OS family."""
    install_windows_console_ctrl_shield()
    actual = detect_runtime_os_family()
    assert actual == os_family, f"runtime_os_mismatch:{actual}!={os_family}"

    contract = OS_CONTRACTS[os_family]
    adapter_script = ADAPTER_SCRIPTS[os_family]
    assert adapter_script.is_file(), f"missing adapter script: {adapter_script}"

    port = reserve_free_port()
    env = build_process_env(tmp_path, port)
    workspace = Path(env["INTERGRAX_ALLOWED_READ_ROOTS"])
    marker_suffix = uuid.uuid4().hex[:12]
    marker = f"{contract['marker_prefix']}_{marker_suffix}"
    fixture_path = workspace / f"{contract['fixture_prefix']}-{marker_suffix}.txt"
    fixture_path.write_text(
        f"{os_family} OS interaction proof fixture\n{marker}\n",
        encoding="utf-8",
    )

    stdout_path = tmp_path / "hosted-stdout.log"
    stderr_path = tmp_path / "hosted-stderr.log"
    command = [sys.executable, "-m", "local_workspace_application.hosting"]
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    expected_reason = "signal.sigbreak" if os_family == "windows" else "signal.sigterm"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            **subprocess_group_kwargs(),
        )
        readiness = wait_until_ready(
            process=process,
            port=port,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        assert_boundary_component(readiness)
        assert_interaction_component_when_present(readiness)
        assert process.poll() is None

        index_response = invoke_os_interaction_adapter(
            os_family=os_family,
            port=port,
            message=f"index {os_family} interaction proof fixture",
            capability="local.workspace.index",
            metadata={
                "source_paths": [str(fixture_path.resolve())],
                "collection_id": contract["collection_id"],
            },
            tenant_id=str(contract["tenant_id"]),
            user_id=str(contract["user_id"]),
        )
        assert index_response.get("tenant_id") == contract["tenant_id"]
        assert index_response.get("user_id") == contract["user_id"]
        assert index_response.get("capability") == "local.workspace.index"
        assert index_response.get("interaction_channel") == "lab"
        assert index_response.get("executed") is True
        assert index_response.get("state") == "completed"
        index_task_id = str(index_response.get("task_id") or "").strip()
        index_run_id = str(index_response.get("run_id") or "").strip()
        assert index_task_id
        assert index_run_id

        search_response = invoke_os_interaction_adapter(
            os_family=os_family,
            port=port,
            message=marker,
            capability="local.workspace.search",
            metadata={
                "query": marker,
                "collection_id": contract["collection_id"],
                "top_k": 5,
            },
            tenant_id=str(contract["tenant_id"]),
            user_id=str(contract["user_id"]),
        )
        assert search_response.get("tenant_id") == contract["tenant_id"]
        assert search_response.get("user_id") == contract["user_id"]
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

        reason = send_graceful_stop_signal(process)
        assert reason == expected_reason
        exit_code = wait_for_process_exit(
            process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stdout_handle.flush()
        stderr_handle.flush()
        assert exit_code == 0, (
            f"stdout:\n{bounded_text(stdout_path)}\n"
            f"stderr:\n{bounded_text(stderr_path)}"
        )
        stop_payload = last_json_line(bounded_text(stdout_path))
        assert_clean_stop_payload(stop_payload, expected_reason=expected_reason)

        record_os_interaction_evidence(
            record_property,
            os_family=os_family,
            index_task_id=index_task_id,
            index_run_id=index_run_id,
            search_task_id=search_task_id,
            search_run_id=search_run_id,
        )
    finally:
        cleanup_process(process)
        stdout_handle.close()
        stderr_handle.close()
