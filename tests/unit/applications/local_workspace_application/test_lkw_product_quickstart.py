# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import errno
import importlib.util
import io
import subprocess
import sys
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/run-lkw-product-quickstart.py"
)
_WINDOWS_BAT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-product-quickstart-windows.bat"
)
_LINUX_SH = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-product-quickstart-linux.sh"
)
_MACOS_SH = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-product-quickstart-macos.sh"
)
_WINDOWS_BOOTSTRAP = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/build-local-docker.bat"
)
_SHELL_BOOTSTRAP = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/build-local-docker.sh"
)


def _load_module() -> ModuleType:
    module_name = "run_lkw_product_quickstart"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quick() -> ModuleType:
    return _load_module()


@pytest.fixture(autouse=True)
def _stub_product_preflight(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "_real_run_product_preflight",
        quick.run_product_preflight,
        raising=False,
    )
    monkeypatch.setattr(
        quick,
        "run_product_preflight",
        lambda *_args, **_kwargs: "llama3.1:latest",
    )
    monkeypatch.setattr(quick, "_is_loopback_tcp_port_reachable", lambda _port: False)


def _config(quick: ModuleType, **overrides: Any) -> Any:
    values = {
        "os_family": quick.OsFamily.WINDOWS,
        "wrapper_id": quick.WrapperId.WINDOWS_BAT,
        "base_url": "http://127.0.0.1:8020",
        "timeout_seconds": 30,
        "skip_stack_start": True,
    }
    values.update(overrides)
    return quick.QuickstartConfig(**values)


def test_valid_os_wrapper_pairs(quick: ModuleType) -> None:
    assert quick.VALID_OS_WRAPPER_PAIRS == frozenset(
        {
            (quick.OsFamily.WINDOWS, quick.WrapperId.WINDOWS_BAT),
            (quick.OsFamily.LINUX, quick.WrapperId.LINUX_SH),
            (quick.OsFamily.MACOS, quick.WrapperId.MACOS_SH),
        }
    )


def test_resolve_run_log_path_uses_application_run_logs_dir(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quick, "_APP_DIR", tmp_path)
    monkeypatch.setattr(quick, "_RUN_LOGS_DIR", tmp_path / ".run_logs")

    path = quick.resolve_run_log_path("lkw-live-regression.log")

    assert path == tmp_path / ".run_logs" / "lkw-live-regression.log"


def test_resolve_run_log_path_rejects_path_traversal(quick: ModuleType) -> None:
    with pytest.raises(quick.QuickstartError) as exc:
        quick.resolve_run_log_path("../escape.log")
    assert exc.value.reason == "invalid_log_file_name"


def test_run_quickstart_can_write_log_file_under_run_logs_dir(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quick, "_APP_DIR", tmp_path)
    monkeypatch.setattr(quick, "_RUN_LOGS_DIR", tmp_path / ".run_logs")
    log_file = quick.resolve_run_log_path("lkw-live-regression.log")

    with quick._maybe_tee_stdout(log_file):
        print("LKW quickstart: PASS")

    assert log_file.read_text(encoding="utf-8") == "LKW quickstart: PASS\n"


def test_invalid_os_wrapper_pair_rejected(quick: ModuleType) -> None:
    with pytest.raises(quick.QuickstartError) as exc:
        quick.validate_os_wrapper_pair(
            quick.OsFamily.WINDOWS,
            quick.WrapperId.LINUX_SH,
        )
    assert exc.value.reason == "invalid_os_wrapper_pair"


def test_operating_system_mismatch_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quick, "detect_os_family", lambda: quick.OsFamily.LINUX)
    with pytest.raises(quick.QuickstartError) as exc:
        quick.validate_os_wrapper_pair(
            quick.OsFamily.WINDOWS,
            quick.WrapperId.WINDOWS_BAT,
        )
    assert exc.value.reason == "operating_system_mismatch"


def test_non_loopback_base_url_rejected(quick: ModuleType) -> None:
    with pytest.raises(quick.QuickstartError) as exc:
        quick.validate_loopback_base_url("http://example.com:8020")
    assert exc.value.reason == "non_loopback_base_url"


def test_env_example_copied_only_when_absent(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    example = app_dir / ".env.example"
    example.write_text(
        "SAFE_KEY=safe\nINTERGRAX_EMBEDDING_PROVIDER=ollama\n"
        "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text\n",
        encoding="utf-8",
    )
    env_file = app_dir / ".env"
    monkeypatch.setattr(quick, "_APP_DIR", app_dir)
    monkeypatch.setattr(quick, "_ENV_FILE", env_file)
    monkeypatch.setattr(quick, "_ENV_EXAMPLE", example)
    created = quick.ensure_env_file()
    assert created is True
    text = env_file.read_text(encoding="utf-8")
    assert "SAFE_KEY=safe" in text
    assert "INTERGRAX_EMBEDDING_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text" in text
    assert "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL" not in text


def test_existing_env_never_overwritten(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    example = app_dir / ".env.example"
    example.write_text("SAFE_KEY=from_example\n", encoding="utf-8")
    env_file = app_dir / ".env"
    original = b"SAFE_KEY=existing\r\n\xff"
    env_file.write_bytes(original)
    monkeypatch.setattr(quick, "_APP_DIR", app_dir)
    monkeypatch.setattr(quick, "_ENV_FILE", env_file)
    monkeypatch.setattr(quick, "_ENV_EXAMPLE", example)
    created = quick.ensure_env_file()
    assert created is False
    assert env_file.read_bytes() == original


def _prepare_preflight_files(
    quick: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_text: str,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    env_file = app_dir / ".env"
    env_file.write_text(env_text, encoding="utf-8")
    example = app_dir / ".env.example"
    example.write_text("INTERGRAX_LLM_PROVIDER=ollama\n", encoding="utf-8")
    sample = app_dir / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_APP_DIR", app_dir)
    monkeypatch.setattr(quick, "_ENV_FILE", env_file)
    monkeypatch.setattr(quick, "_ENV_EXAMPLE", example)
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)


def test_docker_cli_missing_has_safe_preflight_reason(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quick.shutil, "which", lambda _name: None)
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_docker_capabilities()
    assert (exc.value.stage, exc.value.reason) == ("preflight", "docker_cli_missing")


def test_daemon_unavailable_has_safe_preflight_reason(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quick.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_args, **_kwargs: type("CP", (), {"returncode": 1})(),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_docker_capabilities()
    assert exc.value.reason == "docker_daemon_unavailable"


def test_compose_unavailable_has_safe_preflight_reason(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quick.shutil, "which", lambda _name: "docker")
    results = iter(
        [
            type("CP", (), {"returncode": 0})(),
            type("CP", (), {"returncode": 1})(),
        ]
    )
    monkeypatch.setattr(quick, "run_command", lambda *_a, **_k: next(results))
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_docker_capabilities()
    assert exc.value.reason == "compose_unavailable"


def test_occupied_required_port_has_safe_preflight_reason(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BusySocket:
        def setsockopt(self, *_args: Any) -> None:
            return None

        def bind(self, *_args: Any) -> None:
            raise OSError("private socket detail")

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", lambda *_a, **_k: _BusySocket())
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_required_ports(
            mongodb_host_port=27018,
            allow_running_stack=False,
        )
    assert exc.value.reason == "port_unavailable"


def test_ipv4_wildcard_conflict_is_rejected_and_socket_is_closed(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets: list[Any] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family
            self.closed = False
            sockets.append(self)

        def bind(self, address: tuple[object, ...]) -> None:
            if (
                self.family == quick.socket.AF_INET
                and address == ("0.0.0.0", 27018)
            ):
                raise OSError(errno.EADDRINUSE, "port is busy")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_required_ports(
            mongodb_host_port=27018,
            allow_running_stack=False,
        )

    assert exc.value.reason == "port_unavailable"
    assert all(probe.closed for probe in sockets)


def test_ipv6_wildcard_conflict_is_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets: list[Any] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family
            self.closed = False
            sockets.append(self)

        def bind(self, address: tuple[object, ...]) -> None:
            if (
                self.family == quick.socket.AF_INET6
                and address == ("::", 27018, 0, 0)
            ):
                raise OSError(errno.EADDRINUSE, "forwarded port is busy")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    with pytest.raises(quick.QuickstartError) as exc:
        quick._probe_host_port(27018)

    assert exc.value.reason == "port_unavailable"
    assert len(sockets) == 2
    assert all(probe.closed for probe in sockets)


def test_free_port_is_accepted_and_all_probe_sockets_are_closed(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets: list[Any] = []

    class _Socket:
        def __init__(self, *_args: Any) -> None:
            self.closed = False
            sockets.append(self)

        def bind(self, _address: tuple[object, ...]) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._probe_host_port(27018)

    assert len(sockets) == 2
    assert all(probe.closed for probe in sockets)


def test_unsupported_ipv6_does_not_reject_free_ipv4_port(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets: list[Any] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            if family == quick.socket.AF_INET6:
                raise OSError(errno.EAFNOSUPPORT, "IPv6 unavailable")
            self.closed = False
            sockets.append(self)

        def bind(self, _address: tuple[object, ...]) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._probe_host_port(27018)

    assert len(sockets) == 1
    assert sockets[0].closed


def test_disk_space_failure_has_safe_preflight_reason(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_preflight_files(
        quick,
        tmp_path,
        monkeypatch,
        "INTERGRAX_LLM_PROVIDER=ollama\n",
    )
    monkeypatch.setattr(quick.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type("CP", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(
        quick.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": quick._MIN_FREE_SPACE_BYTES - 1})(),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick._real_run_product_preflight(_config(quick))
    assert exc.value.reason == "insufficient_disk_space"


def test_invalid_generation_configuration_is_not_echoed(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_preflight_files(
        quick,
        tmp_path,
        monkeypatch,
        "INTERGRAX_LLM_MODEL=bad model secret\n",
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.resolve_generation_model()
    assert exc.value.reason == "invalid_mandatory_configuration"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        quick._emit_failure(exc.value.stage, exc.value.reason)
    text = buffer.getvalue()
    assert "bad model secret" not in text
    assert "recommended_action=" in text
    assert "Traceback" not in text


def test_generation_model_resolution_honors_configured_and_default_values(
    quick: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_preflight_files(
        quick,
        tmp_path,
        monkeypatch,
        "INTERGRAX_LLM_MODEL=custom/generation:latest\n",
    )
    assert quick.resolve_generation_model() == "custom/generation:latest"
    env_file = Path(quick._ENV_FILE)
    env_file.write_text("INTERGRAX_LLM_PROVIDER=ollama\n", encoding="utf-8")
    assert quick.resolve_generation_model() == "llama3.1:latest"


def _compose_ps_stdout(*services: dict[str, Any]) -> str:
    import json

    return json.dumps(list(services))


def _canonical_service(
    name: str,
    *,
    state: str = "running",
    published_ports: list[int] | None = None,
) -> dict[str, Any]:
    publishers = [
        {
            "URL": "0.0.0.0",
            "TargetPort": port,
            "PublishedPort": port,
            "Protocol": "tcp",
        }
        for port in (published_ports or [])
    ]
    return {"Service": name, "State": state, "Publishers": publishers}


def test_canonical_running_local_workspace_owns_8020(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "local_workspace",
                        state="running",
                        published_ports=[8020],
                    ),
                    _canonical_service("lkw-mongodb", published_ports=[27018]),
                    _canonical_service("otel-collector", published_ports=[4318]),
                ),
            },
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", lambda *_a, **_k: _Socket())
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert probed == []


def test_canonical_restarting_local_workspace_owns_8020(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "local_workspace",
                        state="restarting",
                        published_ports=[8020],
                    ),
                    _canonical_service("lkw-mongodb", published_ports=[27018]),
                    _canonical_service("otel-collector", published_ports=[4318]),
                ),
            },
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", lambda *_a, **_k: _Socket())
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert probed == []


def test_canonical_partial_stack_mongo_owns_configured_port(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "lkw-mongodb",
                        state="running",
                        published_ports=[27019],
                    ),
                ),
            },
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family

        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._check_required_ports(mongodb_host_port=27019, allow_running_stack=True)
    assert sorted(probed) == [4318, 4318, 8020, 8020]


def test_canonical_otel_owns_4318(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "otel-collector",
                        state="running",
                        published_ports=[4318],
                    ),
                ),
            },
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family

        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert 4318 not in probed
    assert 8020 in probed
    assert 27018 in probed


def test_foreign_compose_project_ownership_is_not_canonical(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "_canonical_product_owned_host_ports",
        lambda: frozenset(),
    )
    monkeypatch.setattr(quick, "_is_loopback_tcp_port_reachable", lambda _port: False)

    class _BusySocket:
        def bind(self, address: tuple[object, ...]) -> None:
            if int(address[1]) == 8020:
                raise OSError("foreign docker project owns port")

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", lambda *_a, **_k: _BusySocket())
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert exc.value.reason == "port_unavailable"


def test_loopback_reachable_port_is_rejected_when_bind_probe_passes(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "_canonical_product_owned_host_ports",
        lambda: frozenset(),
    )
    monkeypatch.setattr(quick, "_is_loopback_tcp_port_reachable", lambda port: port == 27018)

    class _Socket:
        def bind(self, *_args: Any) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", lambda *_a, **_k: _Socket())
    with pytest.raises(quick.QuickstartError) as exc:
        quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert exc.value.reason == "port_unavailable"


def test_exited_mongodb_maps_to_mongodb_not_ready(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service("lkw-mongodb", state="exited"),
                ),
            },
        )(),
    )
    assert quick._stack_failure_reason() == "mongodb_not_ready"


def test_port_not_in_canonical_publishers_still_probes(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "local_workspace",
                        state="exited",
                        published_ports=[],
                    ),
                ),
            },
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family

        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert sorted(probed) == [4318, 4318, 8020, 8020, 27018, 27018]


def test_malformed_compose_response_falls_back_to_generic_probe(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {"returncode": 0, "stdout": "not-json"},
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family

        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert sorted(probed) == [4318, 4318, 8020, 8020, 27018, 27018]


def test_canonical_ownership_for_one_port_does_not_skip_other_ports(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "local_workspace",
                        state="running",
                        published_ports=[8020],
                    ),
                ),
            },
        )(),
    )
    probed: list[int] = []

    class _Socket:
        def __init__(self, family: int, *_args: Any) -> None:
            self.family = family

        def bind(self, address: tuple[object, ...]) -> None:
            probed.append(int(address[1]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(quick.socket, "socket", _Socket)
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)
    assert 8020 not in probed
    assert 4318 in probed
    assert 27018 in probed


def test_compose_ndjson_ps_output_is_parsed_for_canonical_ownership(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    ndjson = "\n".join(
        [
            '{"Service":"lkw-mongodb","State":"running","Ports":"0.0.0.0:27018->27017/tcp","Publishers":[{"PublishedPort":27018}]}',
            '{"Service":"local_workspace","State":"restarting","Ports":"0.0.0.0:8020->8020/tcp","Publishers":[{"PublishedPort":8020}]}',
            '{"Service":"otel-collector","State":"running","Ports":"0.0.0.0:4318->4318/tcp","Publishers":[{"PublishedPort":4318}]}',
        ]
    )
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type("CP", (), {"returncode": 0, "stdout": ndjson})(),
    )

    class _UnexpectedSocket:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("port probe must be skipped for canonical ownership")

    monkeypatch.setattr(quick.socket, "socket", _UnexpectedSocket)
    quick._check_required_ports(mongodb_host_port=27018, allow_running_stack=True)


def test_running_product_stack_allows_safe_port_reuse(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": _compose_ps_stdout(
                    _canonical_service(
                        "local_workspace",
                        state="running",
                        published_ports=[8020],
                    ),
                    _canonical_service("lkw-mongodb", published_ports=[27018]),
                    _canonical_service("otel-collector", published_ports=[4318]),
                ),
            },
        )(),
    )

    class _UnexpectedSocket:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("port probe must be skipped for canonical ownership")

    monkeypatch.setattr(quick.socket, "socket", _UnexpectedSocket)
    quick._check_required_ports(
        mongodb_host_port=27018,
        allow_running_stack=True,
    )


def test_dependency_state_maps_without_forwarding_raw_output(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP",
            (),
            {
                "returncode": 0,
                "stdout": '[{"Service":"qdrant","State":"exited","Health":""}]',
            },
        )(),
    )
    assert quick._stack_failure_reason() == "qdrant_not_ready"


def test_failure_output_includes_stable_action_without_raw_details(
    quick: ModuleType,
) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        quick._emit_failure("preflight", "docker_daemon_unavailable")
    text = buffer.getvalue()
    assert "failed_stage=preflight" in text
    assert "failure_reason=docker_daemon_unavailable" in text
    assert "recommended_action=Start Docker and rerun." in text
    assert "Traceback" not in text


def test_port_failure_output_has_stable_preflight_contract(
    quick: ModuleType,
) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        quick._emit_failure("preflight", "port_unavailable")
    text = buffer.getvalue()
    assert "lkw_quickstart_result=FAIL" in text
    assert "failed_stage=preflight" in text
    assert "failure_reason=port_unavailable" in text
    assert "recommended_action=Free the required LKW host port" in text
    assert "Traceback" not in text


def test_bootstrap_selected_per_os(quick: ModuleType) -> None:
    assert quick.bootstrap_args(quick.OsFamily.WINDOWS)[0] == "cmd.exe"
    assert quick.bootstrap_args(quick.OsFamily.LINUX)[0] == "sh"
    assert quick.bootstrap_args(quick.OsFamily.MACOS)[0] == "sh"
    assert quick.bootstrap_args(quick.OsFamily.LINUX)[1] == str(quick._BOOTSTRAP_SH)
    assert quick.bootstrap_args(quick.OsFamily.MACOS)[1] == str(quick._BOOTSTRAP_SH)


def test_bootstrap_compose_and_generation_model_contracts(quick: ModuleType) -> None:
    windows_source = _WINDOWS_BOOTSTRAP.read_text(encoding="utf-8")
    shell_source = _SHELL_BOOTSTRAP.read_text(encoding="utf-8")

    assert quick._COMPOSE_PROJECT == "intergrax_lkw"
    assert quick.compose_exec_args("config")[:6] == [
        "docker",
        "compose",
        "-p",
        "intergrax_lkw",
        "-f",
        str(quick._COMPOSE_FILE),
    ]

    assert 'set "COMPOSE_PROJECT_NAME=intergrax_lkw"' in windows_source
    assert 'docker compose -p "%COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%"' in (
        windows_source
    )
    assert 'COMPOSE_PROJECT_NAME="intergrax_lkw"' in shell_source
    assert (
        'docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" up --build -d'
        in shell_source
    )
    assert 'docker compose -f "$COMPOSE_FILE"' not in shell_source
    assert "INTERGRAX_LLM_MODEL" in windows_source
    assert "INTERGRAX_LLM_MODEL" in shell_source
    assert "ollama pull" in windows_source
    assert 'ollama pull "$INTERGRAX_LLM_MODEL"' in shell_source


def test_stack_bootstrap_invokes_embedding_pull(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    monkeypatch.setattr(quick, "ensure_env_file", lambda: False)
    calls: list[list[str]] = []

    def _run_command(args: list[str], **kwargs: Any) -> Any:
        calls.append(list(args))
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(quick, "run_command", _run_command)
    monkeypatch.setattr(
        quick,
        "ensure_embedding_model_if_ollama",
        lambda **_k: "configured-embed-model",
    )
    monkeypatch.setattr(quick, "wait_for_health", lambda *_a, **_k: None)
    monkeypatch.setattr(quick, "create_workspace", lambda *_a, **_k: "ws-1")
    monkeypatch.setattr(quick, "upload_sample_file", lambda *_a, **_k: "op-1")
    monkeypatch.setattr(quick, "wait_for_operation", lambda *_a, **_k: {})
    monkeypatch.setattr(
        quick,
        "ask_workspace",
        lambda *_a, **_k: {
            "run_id": "run-1",
            "answer": "AURORA-17",
            "citations": [{"file_name": quick._CITATION_FILE}],
        },
    )
    monkeypatch.setattr(quick, "verify_persisted_ask", lambda *_a, **_k: None)
    code = quick.run_quickstart(_config(quick, skip_stack_start=False))
    assert code == 0
    assert any("build-local-docker" in " ".join(call) for call in calls)


def test_skip_stack_start_skips_bootstrap(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    monkeypatch.setattr(quick, "ensure_env_file", lambda: False)
    monkeypatch.setattr(
        quick,
        "ensure_embedding_model_if_ollama",
        lambda **_k: "configured-embed-model",
    )
    monkeypatch.setattr(quick, "wait_for_health", lambda *_a, **_k: None)
    monkeypatch.setattr(quick, "create_workspace", lambda *_a, **_k: "ws-1")
    monkeypatch.setattr(quick, "upload_sample_file", lambda *_a, **_k: "op-1")
    monkeypatch.setattr(quick, "wait_for_operation", lambda *_a, **_k: {})
    monkeypatch.setattr(
        quick,
        "ask_workspace",
        lambda *_a, **_k: {
            "run_id": "run-1",
            "answer": "codename AURORA-17",
            "citations": [{"file_name": quick._CITATION_FILE}],
        },
    )
    monkeypatch.setattr(quick, "verify_persisted_ask", lambda *_a, **_k: None)

    called = False

    def _run_command(*_a: Any, **_k: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("bootstrap must not run")

    monkeypatch.setattr(quick, "run_command", _run_command)
    code = quick.run_quickstart(_config(quick, skip_stack_start=True))
    assert code == 0
    assert called is False


def _patch_success_flow(
    quick: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ask_payload: dict[str, Any] | None = None,
    persisted_side_effect: Any = None,
) -> dict[str, Any]:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    monkeypatch.setattr(quick, "ensure_env_file", lambda: False)
    monkeypatch.setattr(
        quick,
        "ensure_embedding_model_if_ollama",
        lambda **_k: "configured-embed-model",
    )
    monkeypatch.setattr(quick, "wait_for_health", lambda *_a, **_k: None)
    workspace_calls: list[dict[str, Any]] = []
    upload_calls: list[dict[str, Any]] = []

    def _create(base_url: str) -> str:
        workspace_calls.append({"base_url": base_url})
        return "ws-test"

    def _upload(base_url: str, workspace_id: str) -> str:
        upload_calls.append({"base_url": base_url, "workspace_id": workspace_id})
        return "op-test"

    monkeypatch.setattr(quick, "create_workspace", _create)
    monkeypatch.setattr(quick, "upload_sample_file", _upload)
    monkeypatch.setattr(
        quick,
        "wait_for_operation",
        lambda *_a, **_k: {"status": "completed", "documents_indexed": 1, "files_failed": 0},
    )
    payload = ask_payload or {
        "run_id": "run-test",
        "answer": "The project codename is AURORA-17.",
        "citations": [{"file_name": quick._CITATION_FILE}],
    }
    monkeypatch.setattr(quick, "ask_workspace", lambda *_a, **_k: payload)
    if persisted_side_effect is not None:
        monkeypatch.setattr(quick, "verify_persisted_ask", persisted_side_effect)
    else:
        monkeypatch.setattr(quick, "verify_persisted_ask", lambda *_a, **_k: None)
    return {"workspace_calls": workspace_calls, "upload_calls": upload_calls}


def test_workspace_creation_request(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], **kwargs: Any) -> tuple[int, dict[str, Any]]:
        calls.append((url, body, headers))
        return 201, {"workspace_id": "ws-created"}

    monkeypatch.setattr(quick, "http_post_json", _post_json)
    workspace_id = quick.create_workspace("http://127.0.0.1:8020")
    assert workspace_id == "ws-created"
    assert len(calls) == 1
    url, body, headers = calls[0]
    assert url.endswith("/workspaces")
    assert "LKW Product Quickstart" in body["name"]
    assert headers["X-Tenant-Id"] == quick._TENANT_ID


def test_managed_upload_includes_sample(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    captured: dict[str, Any] = {}

    def _post_bytes(url: str, body: bytes, headers: dict[str, str], **kwargs: Any) -> tuple[int, dict[str, Any]]:
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        return 202, {
            "status": "accepted",
            "accepted_count": 1,
            "failed_count": 0,
            "items": [
                {
                    "operation_id": "op-1",
                    "source_id": "src-1",
                }
            ],
        }

    monkeypatch.setattr(quick, "http_post_bytes", _post_bytes)
    operation_id = quick.upload_sample_file("http://127.0.0.1:8020", "ws-1")
    assert operation_id == "op-1"
    assert quick._CITATION_FILE in captured["body"].decode("utf-8", errors="ignore")
    assert "Idempotency-Key" in captured["headers"]
    assert captured["headers"]["X-Tenant-Id"] == quick._TENANT_ID


def test_operation_polling_completed(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = ["queued", "processing", "completed"]
    def _get(url: str, headers: dict[str, str], **kwargs: Any) -> dict[str, Any]:
        status = statuses.pop(0)
        return {
            "status": status,
            "documents_indexed": 1,
            "files_failed": 0,
            "error": None,
        }

    monkeypatch.setattr(quick, "http_get_json", _get)
    payload = quick.wait_for_operation(
        "http://127.0.0.1:8020",
        "op-1",
        {"X-Tenant-Id": quick._TENANT_ID},
        timeout_seconds=10,
    )
    assert payload["status"] == "completed"


def test_operation_failed_terminal(quick: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quick,
        "http_get_json",
        lambda *_a, **_k: {"status": "failed"},
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.wait_for_operation(
            "http://127.0.0.1:8020",
            "op-1",
            {},
            timeout_seconds=5,
        )
    assert exc.value.reason == "operation_failed"


def test_operation_timeout(quick: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quick,
        "http_get_json",
        lambda *_a, **_k: {"status": "processing"},
    )
    times = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(quick.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(quick.time, "sleep", lambda *_a, **_k: None)
    with pytest.raises(quick.QuickstartError) as exc:
        quick.wait_for_operation(
            "http://127.0.0.1:8020",
            "op-1",
            {},
            timeout_seconds=1,
        )
    assert exc.value.reason == "operation_timeout"


def test_progress_reporter_emits_stage_start_heartbeat_and_completion(
    quick: ModuleType,
) -> None:
    now = 0.0
    output: list[str] = []

    def _clock() -> float:
        return now

    progress = quick.ProgressReporter(
        total_stages=1,
        output=output.append,
        clock=_clock,
        heartbeat_interval=10,
    )
    progress.start(1, "Indexing sample knowledge")
    now = 11.0
    progress.heartbeat()
    now = 12.0
    progress.complete("Sample knowledge is indexed")

    assert output == [
        "[1/1] Indexing sample knowledge...",
        "Still indexing sample knowledge... 11s",
        "Sample knowledge is indexed (12s).",
    ]


def test_run_command_emits_heartbeat_without_exposing_captured_output(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 0.0
    output: list[str] = []

    def _clock() -> float:
        return now

    class _Process:
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            nonlocal now
            if timeout is not None and now == 0.0:
                now = 11.0
                raise subprocess.TimeoutExpired("long-command", timeout)
            return "captured secret stdout", "captured secret stderr"

        def kill(self) -> None:
            return None

    monkeypatch.setattr(quick.subprocess, "Popen", lambda *_a, **_k: _Process())
    progress = quick.ProgressReporter(
        total_stages=1,
        output=output.append,
        clock=_clock,
        heartbeat_interval=10,
    )
    progress.start(1, "Preparing embedding model")
    completed = quick.run_command(
        ["safe-command"],
        timeout=30,
        progress=progress,
    )

    assert completed.stdout == "captured secret stdout"
    assert completed.stderr == "captured secret stderr"
    assert output == [
        "[1/1] Preparing embedding model...",
        "Still preparing embedding model... 11s",
    ]
    assert "captured secret" not in "\n".join(output)


def test_run_command_fast_completion_does_not_flood_heartbeats(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    output: list[str] = []

    class _Process:
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr(quick.subprocess, "Popen", lambda *_a, **_k: _Process())
    progress = quick.ProgressReporter(total_stages=1, output=output.append)
    progress.start(1, "Starting local LKW stack")
    quick.run_command(["safe-command"], progress=progress)

    assert output == ["[1/1] Starting local LKW stack..."]


def test_health_waiting_emits_bounded_heartbeat(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 0.0
    output: list[str] = []

    def _clock() -> float:
        return now

    def _sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(quick.time, "monotonic", _clock)
    monkeypatch.setattr(quick.time, "sleep", _sleep)
    monkeypatch.setattr(quick, "http_get_json", lambda *_a, **_k: {"status": "starting"})
    progress = quick.ProgressReporter(
        total_stages=1,
        output=output.append,
        clock=_clock,
        heartbeat_interval=4,
    )
    progress.start(1, "Waiting for LKW services")

    with pytest.raises(quick.QuickstartError) as exc:
        quick.wait_for_health(
            "http://127.0.0.1:8020",
            timeout_seconds=10,
            progress=progress,
        )

    assert exc.value.reason == "health_timeout"
    assert output.count("Still waiting for LKW services... 4s") == 1


def test_indexing_waiting_emits_bounded_heartbeat(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 0.0
    output: list[str] = []

    def _clock() -> float:
        return now

    def _sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(quick.time, "monotonic", _clock)
    monkeypatch.setattr(quick.time, "sleep", _sleep)
    monkeypatch.setattr(
        quick,
        "http_get_json",
        lambda *_a, **_k: {"status": "processing"},
    )
    progress = quick.ProgressReporter(
        total_stages=1,
        output=output.append,
        clock=_clock,
        heartbeat_interval=4,
    )
    progress.start(1, "Indexing sample knowledge")

    with pytest.raises(quick.QuickstartError) as exc:
        quick.wait_for_operation(
            "http://127.0.0.1:8020",
            "op-1",
            {},
            timeout_seconds=10,
            progress=progress,
        )

    assert exc.value.reason == "operation_timeout"
    assert output.count("Still indexing sample knowledge... 4s") == 1


def test_ask_passes_progress_to_blocking_request_helper(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    progress = quick.ProgressReporter(total_stages=1, output=lambda _text: None)

    def _post_json(*_args: Any, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        captured["progress"] = kwargs["progress"]
        return (
            200,
            {
                "status": "completed",
                "answer": "AURORA-17",
                "citations": [{"file_name": quick._CITATION_FILE}],
                "run_id": "run-1",
            },
        )

    monkeypatch.setattr(quick, "http_post_json", _post_json)
    quick.ask_workspace("http://127.0.0.1:8020", "ws-1", progress=progress)

    assert captured["progress"] is progress


def test_ask_completed_with_marker_and_citation(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "http_post_json",
        lambda *_a, **_k: (
            200,
            {
                "status": "completed",
                "answer": "codename AURORA-17",
                "citations": [{"file_name": quick._CITATION_FILE}],
                "run_id": "run-1",
            },
        ),
    )
    payload = quick.ask_workspace("http://127.0.0.1:8020", "ws-1")
    assert "AURORA-17" in payload["answer"]


def test_ask_completed_happy_path_via_http_mock(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _post(url: str, body: dict[str, Any], headers: dict[str, str], **kwargs: Any) -> tuple[int, dict[str, Any]]:
        return 200, {
            "status": "completed",
            "answer": "AURORA-17",
            "citations": [{"file_name": quick._CITATION_FILE}],
            "run_id": "run-1",
        }

    monkeypatch.setattr(quick, "http_post_json", _post)
    payload = quick.ask_workspace("http://127.0.0.1:8020", "ws-1")
    assert payload["run_id"] == "run-1"


def test_ask_insufficient_evidence_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "http_post_json",
        lambda *_a, **_k: (200, {"status": "insufficient_evidence", "answer": "", "citations": []}),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.ask_workspace("http://127.0.0.1:8020", "ws-1")
    assert exc.value.reason == "insufficient_evidence"


def test_answer_missing_marker_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "http_post_json",
        lambda *_a, **_k: (
            200,
            {
                "status": "completed",
                "answer": "no marker here",
                "citations": [{"file_name": quick._CITATION_FILE}],
                "run_id": "run-1",
            },
        ),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.ask_workspace("http://127.0.0.1:8020", "ws-1")
    assert exc.value.reason == "answer_marker_missing"


def test_citation_wrong_file_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "http_post_json",
        lambda *_a, **_k: (
            200,
            {
                "status": "completed",
                "answer": "AURORA-17",
                "citations": [{"file_name": "other.txt"}],
                "run_id": "run-1",
            },
        ),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.ask_workspace("http://127.0.0.1:8020", "ws-1")
    assert exc.value.reason == "citation_file_missing"


def test_persisted_read_mismatch_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick,
        "http_get_json",
        lambda *_a, **_k: {
            "run_id": "other",
            "workspace_id": "ws-1",
            "status": "completed",
            "answer": "AURORA-17",
            "citations": [{"file_name": quick._CITATION_FILE}],
        },
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.verify_persisted_ask("http://127.0.0.1:8020", "run-1", "ws-1")
    assert exc.value.reason == "persisted_run_id_mismatch"


def test_success_output_summary(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_flow(quick, monkeypatch, tmp_path)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = quick.run_quickstart(_config(quick))
    text = buffer.getvalue()
    assert code == 0
    assert "LKW quickstart: PASS" in text
    assert "lkw_quickstart_result=PASS" in text
    assert "answer_marker=AURORA-17" in text
    assert "citation_file=lkw_product_quickstart.txt" in text
    assert "persisted_run_verified=true" in text
    assert "stack_left_running=true" in text


def test_secrets_and_raw_responses_not_printed(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_flow(quick, monkeypatch, tmp_path)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        quick.run_quickstart(_config(quick))
    text = buffer.getvalue().lower()
    for forbidden in ("source_path", "storage_key", "mongodb", "qdrant", "intergrax_allowed"):
        assert forbidden not in text


def test_failure_output_contract(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    monkeypatch.setattr(quick, "ensure_env_file", lambda: False)
    monkeypatch.setattr(
        quick,
        "ensure_embedding_model_if_ollama",
        lambda **_kwargs: "configured-embed-model",
    )
    monkeypatch.setattr(quick, "wait_for_health", lambda *_a, **_k: None)
    monkeypatch.setattr(quick, "create_workspace", lambda *_a, **_k: "ws-1")
    monkeypatch.setattr(quick, "upload_sample_file", lambda *_a, **_k: "op-1")
    monkeypatch.setattr(quick, "wait_for_operation", lambda *_a, **_k: {})
    monkeypatch.setattr(
        quick,
        "ask_workspace",
        lambda *_a, **_k: (_ for _ in ()).throw(
            quick.QuickstartError("answer_marker_missing", stage="ask")
        ),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = quick.run_quickstart(_config(quick))
    text = buffer.getvalue()
    assert code == 1
    assert "lkw_quickstart_result=FAIL" in text
    assert "failed_stage=ask" in text
    assert "failure_reason=answer_marker_missing" in text


def test_workspace_urlerror_has_safe_failure_contract(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("raw transport secret")

    monkeypatch.setattr(quick.urllib.request, "urlopen", _urlopen)
    with pytest.raises(quick.QuickstartError) as exc:
        quick.http_post_json(
            "http://127.0.0.1:8020/workspaces",
            {},
            {},
            stage="workspace",
        )
    assert exc.value.reason == "http_transport_failed"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        quick._emit_failure(exc.value.stage, exc.value.reason)
    text = buffer.getvalue()
    assert "lkw_quickstart_result=FAIL" in text
    assert "raw transport secret" not in text
    assert "Traceback" not in text


def test_http_timeout_has_safe_reason(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError()),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.http_get_json("http://127.0.0.1:8020/health", {}, stage="health")
    assert exc.value.reason == "http_transport_failed"


@pytest.mark.parametrize("body", [b"\xff", b"not-json"])
def test_malformed_http_payload_has_invalid_json_reason(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    class _Response:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    monkeypatch.setattr(quick.urllib.request, "urlopen", lambda *_a, **_k: _Response())
    with pytest.raises(quick.QuickstartError) as exc:
        quick.http_get_json("http://127.0.0.1:8020/health", {}, stage="health")
    assert exc.value.reason == "invalid_json_response"


@pytest.mark.parametrize(
    ("payload", "field", "stage"),
    [
        ({"status": "completed", "documents_indexed": "secret", "files_failed": 0}, "documents_indexed", "ingestion"),
        ({"status": "accepted", "accepted_count": "bad", "failed_count": 0, "items": []}, "accepted_count", "upload"),
    ],
)
def test_malformed_numeric_fields_have_invalid_shape_reason(
    quick: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    field: str,
    stage: str,
) -> None:
    if stage == "ingestion":
        monkeypatch.setattr(quick, "http_get_json", lambda *_a, **_k: payload)
        call = lambda: quick.wait_for_operation(
            "http://127.0.0.1:8020", "op-1", {}, timeout_seconds=1
        )
    else:
        sample = Path(quick._SAMPLE_FILE)
        monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
        monkeypatch.setattr(quick, "http_post_bytes", lambda *_a, **_k: (202, payload))
        call = lambda: quick.upload_sample_file("http://127.0.0.1:8020", "ws-1")
    with pytest.raises(quick.QuickstartError) as exc:
        call()
    assert exc.value.reason == "invalid_response_shape"
    assert exc.value.stage == stage
    assert field in payload


def test_bootstrap_timeout_has_command_timeout_contract(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    monkeypatch.setattr(quick, "ensure_env_file", lambda: False)

    class _TimeoutProcess:
        returncode = -9

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            nonlocal now
            if timeout is None:
                return "", ""
            now = 2.0
            raise subprocess.TimeoutExpired("bootstrap", timeout)

        def kill(self) -> None:
            return None

    now = 0.0

    def _clock() -> float:
        return now

    monkeypatch.setattr(quick.time, "monotonic", _clock)
    monkeypatch.setattr(
        quick.subprocess,
        "Popen",
        lambda *_a, **_k: _TimeoutProcess(),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = quick.run_quickstart(
            _config(quick, skip_stack_start=False, timeout_seconds=1)
        )
    assert code == 1
    assert "failure_reason=command_timeout" in buffer.getvalue()


def test_subprocess_launch_failure_has_safe_contract(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        quick.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("secret command path")),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.run_command(["missing-command"], stage="stack_start")
    assert exc.value.reason == "command_start_failed"


def test_subprocess_output_is_not_printed_on_failure(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = tmp_path / "lkw_product_quickstart.txt"
    sample.write_text("AURORA-17", encoding="utf-8")
    monkeypatch.setattr(quick, "_SAMPLE_FILE", sample)
    monkeypatch.setattr(quick, "ensure_env_file", lambda: False)
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP", (), {
                "returncode": 1,
                "stdout": "stdout FAKE_SECRET",
                "stderr": "stderr FAKE_SECRET",
            }
        )(),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = quick.run_quickstart(_config(quick, skip_stack_start=False))
    text = buffer.getvalue()
    assert code == 1
    assert "failure_reason=stack_start_failed" in text
    assert "FAKE_SECRET" not in text


def test_unexpected_exception_has_safe_failure_contract(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_flow(quick, monkeypatch, tmp_path)
    monkeypatch.setattr(
        quick,
        "create_workspace",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("private traceback")),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = quick.run_quickstart(_config(quick))
    text = buffer.getvalue()
    assert code == 1
    assert "failure_reason=unexpected_internal_error" in text
    assert "private traceback" not in text
    assert "Traceback" not in text


def test_model_resolution_reads_container_configured_value(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def _run(args: list[str], **_kwargs: Any) -> Any:
        calls.append(args)
        return type(
            "CP",
            (),
            {"returncode": 0, "stdout": "ollama\ncustom/embed:latest\n", "stderr": ""},
        )()

    monkeypatch.setattr(quick, "run_command", _run)
    model_name = quick.resolve_ollama_embedding_model(timeout_seconds=10)
    assert model_name == "custom/embed:latest"
    command = " ".join(calls[0])
    assert "local_workspace" in command
    assert "embedding_profile_from_env" in command


def test_embedding_profile_resolution_code_uses_canonical_contract() -> None:
    text = (
        Path("applications/local_workspace_application/scripts/lkw_ollama_embedding_bootstrap.py")
        .read_text(encoding="utf-8")
    )
    assert "embedding_profile_from_env" in text
    assert "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL" not in text


def test_resolved_model_is_passed_to_ollama_pull(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def _run(args: list[str], **_kwargs: Any) -> Any:
        calls.append(args)
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(quick, "run_command", _run)
    quick.ensure_ollama_embedding_model("custom/embed:latest", timeout_seconds=10)
    assert calls[0][-1] == "custom/embed:latest"


@pytest.mark.parametrize("output", ["one\ntwo\n", "\n", "x" * 257, "bad\x01model\n", "openai\nmodel\n"])
def test_malformed_embedding_model_output_is_rejected(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(
        quick,
        "run_command",
        lambda *_a, **_k: type(
            "CP", (), {"returncode": 0, "stdout": output, "stderr": ""}
        )(),
    )
    with pytest.raises(quick.QuickstartError) as exc:
        quick.resolve_ollama_embedding_model(timeout_seconds=10)
    assert exc.value.reason in {
        "embedding_model_resolution_failed",
        "embedding_provider_not_ollama",
        "embedding_profile_resolution_failed",
    }


def test_skip_stack_start_still_provisions_ollama_embedding_when_configured(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    _patch_success_flow(quick, monkeypatch, tmp_path)
    monkeypatch.setattr(
        quick,
        "ensure_embedding_model_if_ollama",
        lambda **_k: calls.append("embed") or "custom/embed:latest",
    )
    code = quick.run_quickstart(_config(quick, skip_stack_start=True))
    assert code == 0
    assert calls == ["embed"]


def test_no_shell_true_in_runner_source() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "shell=False" in source


@pytest.mark.parametrize(
    ("path", "os_family", "wrapper_id"),
    [
        (_WINDOWS_BAT, "windows", "windows_bat"),
        (_LINUX_SH, "linux", "linux_sh"),
        (_MACOS_SH, "macos", "macos_sh"),
    ],
)
def test_wrapper_references_runner(
    path: Path, os_family: str, wrapper_id: str
) -> None:
    text = path.read_text(encoding="utf-8")
    assert "run-lkw-product-quickstart.py" in text
    assert os_family in text
    assert wrapper_id in text
