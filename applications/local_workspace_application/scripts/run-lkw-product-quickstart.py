#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Shared cross-platform LKW product quickstart runner.

OS launchers are transport-only. Product orchestration and acceptance live here.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import io
import json
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_RUN_LOGS_DIR = _APP_DIR / ".run_logs"
_SAMPLE_FILE = _APP_DIR / "sample_docs" / "lkw_product_quickstart.txt"
_ENV_FILE = _APP_DIR / ".env"
_ENV_EXAMPLE = _APP_DIR / ".env.example"
_BOOTSTRAP_BAT = _SCRIPT_DIR / "build-local-docker.bat"
_BOOTSTRAP_SH = _SCRIPT_DIR / "build-local-docker.sh"
_COMPOSE_FILE = _APP_DIR / "docker" / "docker-compose.yml"
_COMPOSE_PROJECT = "intergrax_lkw"
_DEFAULT_GENERATION_MODEL = "llama3.1:latest"
_MIN_FREE_SPACE_BYTES = 20 * 1024**3
_PRODUCT_HOST_PORT = 8020
_OTEL_HOST_PORT = 4318
_DEFAULT_MONGODB_HOST_PORT = 27018
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_ENV_KEYS = frozenset(
    {
        "INTERGRAX_LLM_MODEL",
        "INTERGRAX_LLM_PROVIDER",
        "LKW_MONGODB_HOST_PORT",
    }
)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from lkw_ollama_embedding_bootstrap import (
    MAX_EMBEDDING_MODEL_LENGTH,
    OllamaEmbeddingBootstrapError,
    ensure_ollama_embedding_model_if_configured as _ensure_ollama_embedding_model_if_configured,
    ensure_ollama_embedding_model as _ensure_ollama_embedding_model,
    resolve_ollama_embedding_model as _resolve_ollama_embedding_model,
    validate_resolved_embedding_model,
)

_MAX_EMBEDDING_MODEL_LENGTH = MAX_EMBEDDING_MODEL_LENGTH

_DEFAULT_BASE_URL = "http://127.0.0.1:8020"
_DEFAULT_TIMEOUT = 600
_API_PREFIX = "/v1/local_workspace"
_TENANT_ID = "lkw-product-quickstart"
_QUESTION = "What is the project codename?"
_ANSWER_MARKER = "AURORA-17"
_CITATION_FILE = "lkw_product_quickstart.txt"
_SAFE_REASON = re.compile(r"^[A-Za-z0-9_.-]+$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_FAILURE_ACTIONS = {
    "docker_cli_missing": "Install Docker Desktop/Engine with Compose and rerun.",
    "docker_daemon_unavailable": "Start Docker and rerun.",
    "compose_unavailable": "Install or enable Docker Compose and rerun.",
    "unsupported_operating_system": "Run the documented launcher on Windows, Linux, or macOS.",
    "invalid_os_wrapper_pair": "Run the launcher matching the supported operating system.",
    "operating_system_mismatch": "Run the launcher directly on its matching supported operating system.",
    "port_unavailable": "Free the required LKW host port or correct supported product configuration, then rerun.",
    "insufficient_disk_space": "Free disk space for the first Docker/model bootstrap, then rerun.",
    "invalid_mandatory_configuration": "Correct the supported LKW model/provider or host-port configuration, then rerun.",
    "env_example_missing": "Restore .env.example from the repository and rerun.",
    "env_materialization_failed": "Make the application configuration directory writable, then rerun.",
    "bootstrap_script_missing": "Restore the supported LKW bootstrap scripts and rerun.",
    "sample_file_missing": "Restore the bundled LKW quickstart sample and rerun.",
    "stack_start_failed": "Retry the quickstart; if it persists, inspect the documented advanced Docker status commands.",
    "mongodb_not_ready": "Retry after Docker can start the local dependency stack.",
    "qdrant_not_ready": "Retry after Docker can start the local dependency stack.",
    "ollama_not_ready": "Retry after Docker can start the local model service.",
    "lkw_host_not_ready": "Retry after the local LKW host can start.",
    "health_timeout": "Retry the quickstart after Docker services finish starting.",
    "generation_model_pull_failed": "Check local model-service connectivity and rerun.",
    "embedding_model_resolution_failed": "Retry after the local LKW host is ready.",
    "embedding_model_pull_failed": "Check local model-service connectivity and rerun.",
    "invalid_timeout": "Run the launcher with a positive timeout.",
}

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
    log_file: Path | None = None


class _TeeStdout(io.TextIOBase):
    def __init__(self, stream: Any, log_file: Any) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        self._stream.flush()
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()


class ProgressReporter:
    def __init__(
        self,
        *,
        total_stages: int,
        output: Callable[[str], None] | None = None,
        clock: Callable[[], float] | None = None,
        heartbeat_interval: float = 15.0,
    ) -> None:
        self._total_stages = total_stages
        self._output = output or (lambda message: print(message, flush=True))
        self._clock = clock or time.monotonic
        self.heartbeat_interval = max(0.1, heartbeat_interval)
        self._description = ""
        self._started_at: float | None = None
        self._next_heartbeat_at: float | None = None

    def start(self, stage_number: int, description: str) -> None:
        self._description = description
        self._started_at = self._clock()
        self._next_heartbeat_at = self._started_at + self.heartbeat_interval
        self._output(f"[{stage_number}/{self._total_stages}] {description}...")

    def heartbeat(self) -> None:
        if self._started_at is None or self._next_heartbeat_at is None:
            return
        now = self._clock()
        if now < self._next_heartbeat_at:
            return
        elapsed = max(0, int(now - self._started_at))
        operation = self._description[:1].lower() + self._description[1:]
        self._output(f"Still {operation}... {elapsed}s")
        self._next_heartbeat_at = now + self.heartbeat_interval

    def complete(self, description: str) -> None:
        started_at = self._started_at
        elapsed = 0 if started_at is None else max(0, int(self._clock() - started_at))
        self._output(f"{description} ({elapsed}s).")
        self._description = ""
        self._started_at = None
        self._next_heartbeat_at = None


def resolve_run_log_path(name: str) -> Path:
    log_name = Path(name).name
    if not log_name or log_name != name:
        raise QuickstartError("invalid_log_file_name", stage="preflight")
    return _RUN_LOGS_DIR / log_name


@contextlib.contextmanager
def _maybe_tee_stdout(log_file: Path | None):
    if log_file is None:
        yield
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_file.open("w", encoding="utf-8") as handle:
        sys.stdout = _TeeStdout(original_stdout, handle)
        try:
            yield
        finally:
            sys.stdout = original_stdout


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
    safe_stage = stage if _SAFE_REASON.fullmatch(stage) else "unknown"
    safe_reason = reason if _SAFE_REASON.fullmatch(reason) else "unsafe_failure_reason"
    _print_kv("lkw_quickstart_result", "FAIL")
    _print_kv("failed_stage", safe_stage)
    _print_kv("failure_reason", safe_reason)
    _print_kv(
        "recommended_action",
        _FAILURE_ACTIONS.get(
            safe_reason,
            "Retry the documented LKW product quickstart.",
        ),
    )


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
    progress: ProgressReporter | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=dict(env) if env is not None else None,
        )
        started_at = time.monotonic()
        deadline = None if timeout is None else started_at + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(list(args), timeout)
            wait_timeout = remaining
            if progress is not None:
                heartbeat_wait = progress.heartbeat_interval
                wait_timeout = (
                    heartbeat_wait
                    if wait_timeout is None
                    else min(wait_timeout, heartbeat_wait)
                )
            try:
                stdout, stderr = process.communicate(timeout=wait_timeout)
                return subprocess.CompletedProcess(
                    list(args),
                    process.returncode,
                    stdout,
                    stderr,
                )
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
                if progress is not None:
                    progress.heartbeat()
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.communicate()
            except OSError:
                pass
        raise QuickstartError("command_timeout", stage=stage) from None
    except OSError:
        raise QuickstartError("command_start_failed", stage=stage) from None


def _call_with_progress(
    operation: Callable[[], Any],
    *,
    timeout: float,
    progress: ProgressReporter | None,
    stage: str,
) -> Any:
    if progress is None:
        return operation()

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put((True, operation()))
        except Exception as exc:  # noqa: BLE001 - return safe failure to caller
            result_queue.put((False, exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QuickstartError("http_timeout", stage=stage)
        try:
            succeeded, value = result_queue.get(
                timeout=min(remaining, progress.heartbeat_interval)
            )
        except queue.Empty:
            progress.heartbeat()
            continue
        if succeeded:
            return value
        raise value


def ensure_env_file() -> bool:
    if _ENV_FILE.is_file():
        return False
    if not _ENV_EXAMPLE.is_file():
        raise QuickstartError("env_example_missing", stage="preflight")
    try:
        shutil.copyfile(_ENV_EXAMPLE, _ENV_FILE)
    except OSError:
        raise QuickstartError("env_materialization_failed", stage="preflight") from None
    return True


def _read_supported_env_values() -> dict[str, str]:
    if not _ENV_FILE.is_file():
        raise QuickstartError("env_materialization_failed", stage="preflight")
    try:
        lines = _ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise QuickstartError(
            "invalid_mandatory_configuration",
            stage="preflight",
        ) from None
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in _ENV_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value.strip()
    return values


def _configured_value(values: Mapping[str, str], key: str) -> str | None:
    environment_value = os.environ.get(key)
    if environment_value is not None:
        return environment_value.strip()
    return values.get(key)


def resolve_generation_model() -> str:
    values = _read_supported_env_values()
    value = _configured_value(values, "INTERGRAX_LLM_MODEL")
    if value is not None:
        if not _MODEL_PATTERN.fullmatch(value):
            raise QuickstartError(
                "invalid_mandatory_configuration",
                stage="preflight",
            )
        return value
    return _DEFAULT_GENERATION_MODEL


def _resolve_mongodb_host_port(values: Mapping[str, str]) -> int:
    raw_value = _configured_value(values, "LKW_MONGODB_HOST_PORT")
    if raw_value is None or not raw_value:
        return _DEFAULT_MONGODB_HOST_PORT
    if not raw_value.isdecimal():
        raise QuickstartError(
            "invalid_mandatory_configuration",
            stage="preflight",
        )
    port = int(raw_value)
    if not 1 <= port <= 65535:
        raise QuickstartError(
            "invalid_mandatory_configuration",
            stage="preflight",
        )
    return port


def _validate_mandatory_configuration() -> str:
    values = _read_supported_env_values()
    provider = _configured_value(values, "INTERGRAX_LLM_PROVIDER")
    if provider is not None and provider.strip().lower() not in {"", "ollama"}:
        raise QuickstartError(
            "invalid_mandatory_configuration",
            stage="preflight",
        )
    model = resolve_generation_model()
    _resolve_mongodb_host_port(values)
    return model


def _check_docker_capabilities() -> None:
    if shutil.which("docker") is None:
        raise QuickstartError("docker_cli_missing", stage="preflight")
    daemon = run_command(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        timeout=30,
        stage="preflight",
    )
    if daemon.returncode != 0:
        raise QuickstartError("docker_daemon_unavailable", stage="preflight")
    compose = run_command(
        ["docker", "compose", "version"],
        timeout=30,
        stage="preflight",
    )
    if compose.returncode != 0:
        raise QuickstartError("compose_unavailable", stage="preflight")


def _parse_compose_ps_services(stdout: str) -> list[dict[str, Any]] | None:
    raw = stdout[:65536].strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        services: list[dict[str, Any]] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if isinstance(item, dict):
                services.append(item)
        return services
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        services: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                services.append(item)
        return services
    return None


def _host_ports_from_compose_ports_field(ports_field: str) -> set[int]:
    ports: set[int] = set()
    for fragment in str(ports_field).split(","):
        fragment = fragment.strip()
        if "->" not in fragment:
            continue
        host_mapping = fragment.split("->", 1)[0]
        host_port_text = host_mapping.rsplit(":", 1)[-1].strip()
        if not host_port_text.isdecimal():
            continue
        port = int(host_port_text)
        if 1 <= port <= 65535:
            ports.add(port)
    return ports


def _host_ports_from_compose_service(service: Mapping[str, Any]) -> set[int]:
    ports = _host_ports_from_compose_publishers(service)
    ports_field = service.get("Ports", service.get("ports"))
    if isinstance(ports_field, str):
        ports.update(_host_ports_from_compose_ports_field(ports_field))
    return ports


def _host_ports_from_compose_publishers(service: Mapping[str, Any]) -> set[int]:
    ports: set[int] = set()
    publishers = service.get("Publishers", service.get("publishers"))
    if not isinstance(publishers, list):
        return ports
    for entry in publishers:
        if not isinstance(entry, dict):
            continue
        published = entry.get("PublishedPort", entry.get("publishedPort"))
        if isinstance(published, bool):
            continue
        try:
            port = int(published)
        except (TypeError, ValueError, OverflowError):
            continue
        if 1 <= port <= 65535:
            ports.add(port)
    return ports


def _canonical_product_owned_host_ports() -> frozenset[int] | None:
    completed = run_command(
        compose_exec_args("ps", "-a", "--format", "json"),
        timeout=30,
        stage="preflight",
    )
    if completed.returncode != 0:
        return None
    services = _parse_compose_ps_services(completed.stdout)
    if services is None:
        return None
    owned: set[int] = set()
    for service in services:
        owned.update(_host_ports_from_compose_service(service))
    return frozenset(owned)


def _running_product_stack() -> bool:
    completed = run_command(
        compose_exec_args("ps", "--format", "json"),
        timeout=30,
        stage="preflight",
    )
    if completed.returncode != 0:
        return False
    services = _parse_compose_ps_services(completed.stdout)
    if services is None:
        return False
    for service in services:
        name = str(service.get("Service", service.get("service", ""))).strip()
        state = str(service.get("State", service.get("state", ""))).strip().lower()
        if name == "local_workspace" and state in {"running", "up"}:
            return True
    return False


_IPV6_UNSUPPORTED_ERRNOS = frozenset(
    error
    for error in (
        getattr(errno, "EAFNOSUPPORT", None),
        getattr(errno, "EPROTONOSUPPORT", None),
        getattr(errno, "ENOPROTOOPT", None),
        getattr(errno, "EADDRNOTAVAIL", None),
        10043,  # WSAEPROTONOSUPPORT
        10047,  # WSAEAFNOSUPPORT
        10049,  # WSAEADDRNOTAVAIL
    )
    if error is not None
)


def _is_unsupported_ipv6_error(error: OSError) -> bool:
    return error.errno in _IPV6_UNSUPPORTED_ERRNOS


def _is_loopback_tcp_port_reachable(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex(("127.0.0.1", port)) == 0
    finally:
        probe.close()


def _probe_host_port(port: int) -> None:
    probes: list[tuple[int, tuple[object, ...]]] = [
        (socket.AF_INET, ("0.0.0.0", port)),
    ]
    ipv6_family = getattr(socket, "AF_INET6", None)
    if ipv6_family is not None:
        probes.append((ipv6_family, ("::", port, 0, 0)))

    for family, address in probes:
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError as error:
            if family == ipv6_family and _is_unsupported_ipv6_error(error):
                continue
            raise QuickstartError("port_unavailable", stage="preflight") from None
        try:
            exclusive_address_use = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive_address_use is not None and hasattr(
                probe, "setsockopt"
            ):
                try:
                    probe.setsockopt(
                        socket.SOL_SOCKET,
                        exclusive_address_use,
                        1,
                    )
                except OSError:
                    pass
            probe.bind(address)
        except OSError as error:
            if family == ipv6_family and _is_unsupported_ipv6_error(error):
                continue
            raise QuickstartError("port_unavailable", stage="preflight") from None
        finally:
            probe.close()


def _check_required_ports(
    *,
    mongodb_host_port: int,
    allow_running_stack: bool,
) -> None:
    canonical_owned: frozenset[int] | None = None
    if allow_running_stack:
        canonical_owned = _canonical_product_owned_host_ports()
    required_ports = {_PRODUCT_HOST_PORT, _OTEL_HOST_PORT, mongodb_host_port}
    for port in sorted(required_ports):
        if canonical_owned is not None and port in canonical_owned:
            continue
        _probe_host_port(port)
        if _is_loopback_tcp_port_reachable(port):
            raise QuickstartError("port_unavailable", stage="preflight")


def run_product_preflight(config: QuickstartConfig) -> str:
    validate_os_wrapper_pair(config.os_family, config.wrapper_id)
    validate_loopback_base_url(config.base_url)
    if not _SAMPLE_FILE.is_file():
        raise QuickstartError("sample_file_missing", stage="preflight")
    if not _ENV_FILE.is_file() and not _ENV_EXAMPLE.is_file():
        raise QuickstartError("env_example_missing", stage="preflight")
    ensure_env_file()
    model = _validate_mandatory_configuration()
    _check_docker_capabilities()
    if shutil.disk_usage(_APP_DIR).free < _MIN_FREE_SPACE_BYTES:
        raise QuickstartError("insufficient_disk_space", stage="preflight")
    if not config.skip_stack_start:
        values = _read_supported_env_values()
        _check_required_ports(
            mongodb_host_port=_resolve_mongodb_host_port(values),
            allow_running_stack=True,
        )
    return model


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


def _stack_failure_reason() -> str:
    completed = run_command(
        compose_exec_args("ps", "-a", "--format", "json"),
        timeout=30,
        stage="stack_start",
    )
    if completed.returncode != 0:
        return "stack_start_failed"
    services = _parse_compose_ps_services(completed.stdout)
    if services is None:
        return "stack_start_failed"
    service_reasons = {
        "lkw-mongodb": "mongodb_not_ready",
        "qdrant": "qdrant_not_ready",
        "ollama": "ollama_not_ready",
        "local_workspace": "lkw_host_not_ready",
    }
    for service in services:
        if not isinstance(service, dict):
            continue
        name = str(service.get("Service", service.get("service", ""))).strip()
        state = str(service.get("State", service.get("state", ""))).strip().lower()
        health = str(service.get("Health", service.get("health", ""))).strip().lower()
        if name in service_reasons and (
            state in {"exited", "dead", "created"} or health == "unhealthy"
        ):
            return service_reasons[name]
    return "stack_start_failed"


def resolve_ollama_embedding_model(
    *,
    timeout_seconds: int,
    progress: ProgressReporter | None = None,
) -> str:
    try:
        return _resolve_ollama_embedding_model(
            compose_exec_args=compose_exec_args,
            run_command=run_command,
            cwd=_APP_DIR,
            timeout_seconds=timeout_seconds,
            run_command_kwargs={
                "stage": "stack_start",
                "progress": progress,
            },
        )
    except OllamaEmbeddingBootstrapError as exc:
        raise QuickstartError(exc.reason, stage="stack_start") from exc


def ensure_ollama_embedding_model(
    model_name: str,
    *,
    timeout_seconds: int,
    progress: ProgressReporter | None = None,
) -> None:
    try:
        _ensure_ollama_embedding_model(
            model_name,
            compose_exec_args=compose_exec_args,
            run_command=run_command,
            cwd=_APP_DIR,
            timeout_seconds=timeout_seconds,
            run_command_kwargs={
                "stage": "stack_start",
                "progress": progress,
            },
        )
    except OllamaEmbeddingBootstrapError as exc:
        raise QuickstartError(exc.reason, stage="stack_start") from exc


def ensure_embedding_model_if_ollama(
    *,
    timeout_seconds: int,
    progress: ProgressReporter | None = None,
) -> str | None:
    try:
        return _ensure_ollama_embedding_model_if_configured(
            compose_exec_args=compose_exec_args,
            run_command=run_command,
            cwd=_APP_DIR,
            timeout_seconds=timeout_seconds,
            run_command_kwargs={
                "stage": "stack_start",
                "progress": progress,
            },
        )
    except OllamaEmbeddingBootstrapError as exc:
        raise QuickstartError(exc.reason, stage="stack_start") from exc


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
    progress: ProgressReporter | None = None,
) -> tuple[int, dict[str, Any]]:
    def _request() -> tuple[int, dict[str, Any]]:
        data = json.dumps(dict(body)).encode("utf-8")
        merged = dict(headers)
        merged["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=data,
            headers=merged,
            method="POST",
        )
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

    return _call_with_progress(
        _request,
        timeout=timeout,
        progress=progress,
        stage=stage,
    )


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


def wait_for_health(
    base_url: str,
    *,
    timeout_seconds: int,
    progress: ProgressReporter | None = None,
) -> None:
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
        if progress is not None:
            progress.heartbeat()
        time.sleep(2)
    raise QuickstartError("health_timeout", stage="health")


def wait_for_operation(
    base_url: str,
    operation_id: str,
    headers: Mapping[str, str],
    *,
    timeout_seconds: int,
    progress: ProgressReporter | None = None,
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
        if progress is not None:
            progress.heartbeat()
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


def upload_sample_file(
    base_url: str,
    workspace_id: str,
) -> str:
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


def ask_workspace(
    base_url: str,
    workspace_id: str,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    status, payload = http_post_json(
        f"{base_url}{_API_PREFIX}/workspaces/{workspace_id}/ask",
        {"question": _QUESTION, "limit": 5},
        _tenant_headers(),
        timeout=180.0,
        stage="ask",
        progress=progress,
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
        "Stack remains running for inspection. See applications/local_workspace_application/docs/product/QUICKSTART.md "
        "for stop and troubleshooting commands.",
        flush=True,
    )


def run_quickstart(config: QuickstartConfig) -> int:
    with _maybe_tee_stdout(config.log_file):
        return _run_quickstart(config)


def _run_quickstart(config: QuickstartConfig) -> int:
    current_stage = "preflight"
    progress = ProgressReporter(total_stages=10)
    try:
        progress.start(1, "Checking prerequisites")
        base_url = validate_loopback_base_url(config.base_url)
        created_env = not _ENV_FILE.is_file()
        generation_model = run_product_preflight(config)
        if created_env:
            print(
                "Created applications/local_workspace_application/.env from .env.example "
                "for local evaluation.",
                flush=True,
            )
        progress.complete("Prerequisites ready")
        if not config.skip_stack_start:
            current_stage = "stack_start"
            progress.start(2, "Starting local LKW stack")
            if not _BOOTSTRAP_BAT.is_file() or not _BOOTSTRAP_SH.is_file():
                raise QuickstartError("bootstrap_script_missing", stage="preflight")
            completed = run_command(
                bootstrap_args(config.os_family),
                cwd=_REPO_ROOT,
                timeout=config.timeout_seconds,
                stage="stack_start",
                progress=progress,
                env={
                    **os.environ,
                    "INTERGRAX_LLM_MODEL": generation_model,
                },
            )
            if completed.returncode != 0:
                raise QuickstartError(
                    _stack_failure_reason(),
                    stage="stack_start",
                )
            progress.complete("Local LKW stack started")
        else:
            progress.start(2, "Reusing local LKW stack")
            progress.complete("Using existing local LKW stack")
        progress.start(3, "Waiting for LKW services")
        current_stage = "health"
        wait_for_health(
            base_url,
            timeout_seconds=config.timeout_seconds,
            progress=progress,
        )
        progress.complete("LKW services are ready")
        progress.start(4, "Preparing embedding model")
        current_stage = "stack_start"
        embedding_model = ensure_embedding_model_if_ollama(
            timeout_seconds=config.timeout_seconds,
            progress=progress,
        )
        if embedding_model is None:
            progress.complete("Non-Ollama embedding provider; skipped Ollama embedding pull")
        else:
            progress.complete("Embedding model is ready")
        progress.start(5, "Creating evaluation workspace")
        current_stage = "workspace"
        workspace_id = create_workspace(base_url)
        progress.complete("Evaluation workspace is ready")
        progress.start(6, "Uploading sample knowledge")
        current_stage = "upload"
        operation_id = upload_sample_file(
            base_url,
            workspace_id,
        )
        progress.complete("Sample knowledge upload accepted")
        progress.start(7, "Indexing sample knowledge")
        current_stage = "ingestion"
        wait_for_operation(
            base_url,
            operation_id,
            _tenant_headers(),
            timeout_seconds=config.timeout_seconds,
            progress=progress,
        )
        progress.complete("Sample knowledge is indexed")
        progress.start(8, "Asking a grounded question")
        current_stage = "ask"
        ask_payload = ask_workspace(base_url, workspace_id, progress=progress)
        progress.complete("Grounded answer is ready")
        progress.start(9, "Verifying saved Ask result")
        run_id = str(ask_payload.get("run_id", "")).strip()
        answer = str(ask_payload.get("answer", "")).strip()
        current_stage = "persisted_read"
        verify_persisted_ask(base_url, run_id, workspace_id)
        progress.complete("Saved Ask result is verified")
        progress.start(10, "Finalizing Quick Start")
        _assert_safe_user_text(answer)
        progress.complete("Quick Start completed successfully")
        emit_success(answer, workspace_id, run_id)
        return 0
    except QuickstartError as exc:
        _emit_failure(exc.stage, exc.reason)
        return 1
    except Exception:  # noqa: BLE001 - preserve safe failure contract
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
    parser.add_argument(
        "--log-file",
        metavar="NAME",
        help=(
            "Write runner output to applications/local_workspace_application/"
            ".run_logs/NAME (local only)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        _emit_failure("preflight", "invalid_timeout")
        return 1
    log_file = resolve_run_log_path(args.log_file) if args.log_file else None
    config = QuickstartConfig(
        os_family=OsFamily(args.os_family),
        wrapper_id=WrapperId(args.wrapper_id),
        base_url=str(args.base_url),
        timeout_seconds=int(args.timeout_seconds),
        skip_stack_start=bool(args.skip_stack_start),
        log_file=log_file,
    )
    return run_quickstart(config)


if __name__ == "__main__":
    sys.exit(main())
