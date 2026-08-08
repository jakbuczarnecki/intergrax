# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

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
    example.write_text("SAFE_KEY=safe\n", encoding="utf-8")
    env_file = app_dir / ".env"
    monkeypatch.setattr(quick, "_APP_DIR", app_dir)
    monkeypatch.setattr(quick, "_ENV_FILE", env_file)
    monkeypatch.setattr(quick, "_ENV_EXAMPLE", example)
    created = quick.ensure_env_file()
    assert created is True
    text = env_file.read_text(encoding="utf-8")
    assert "SAFE_KEY=safe" in text
    assert "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL=" in text


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


def test_bootstrap_selected_per_os(quick: ModuleType) -> None:
    assert quick.bootstrap_args(quick.OsFamily.WINDOWS)[0] == "cmd.exe"
    assert quick.bootstrap_args(quick.OsFamily.LINUX)[0] == "sh"
    assert quick.bootstrap_args(quick.OsFamily.MACOS)[0] == "sh"


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
        "resolve_ollama_embedding_model",
        lambda **_k: "configured-embed-model",
    )
    monkeypatch.setattr(
        quick,
        "ensure_ollama_embedding_model",
        lambda *_a, **_k: None,
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
        "resolve_ollama_embedding_model",
        lambda **_k: "configured-embed-model",
    )
    monkeypatch.setattr(quick, "ensure_ollama_embedding_model", lambda *_a, **_k: None)
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
        "resolve_ollama_embedding_model",
        lambda **_k: "configured-embed-model",
    )
    monkeypatch.setattr(quick, "ensure_ollama_embedding_model", lambda *_a, **_k: None)
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
        return type("CP", (), {"returncode": 0, "stdout": "custom/embed:latest\n", "stderr": ""})()

    monkeypatch.setattr(quick, "run_command", _run)
    model_name = quick.resolve_ollama_embedding_model(timeout_seconds=10)
    assert model_name == "custom/embed:latest"
    command = " ".join(calls[0])
    assert "local_workspace" in command
    assert "OllamaEmbeddingProvider.ENV_MODEL" in command
    assert "OllamaEmbeddingProvider.DEFAULT_MODEL" in command


def test_model_resolution_expression_uses_runtime_default_when_env_missing(
    quick: ModuleType,
) -> None:
    assert "os.getenv(OllamaEmbeddingProvider.ENV_MODEL)" in quick._MODEL_RESOLUTION_CODE
    assert "OllamaEmbeddingProvider.DEFAULT_MODEL" in quick._MODEL_RESOLUTION_CODE


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


@pytest.mark.parametrize("output", ["one\ntwo\n", "\n", "x" * 257, "bad\x01model\n"])
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
    assert exc.value.reason == "embedding_model_resolution_failed"


def test_skip_stack_start_still_resolves_and_pulls_model(
    quick: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    _patch_success_flow(quick, monkeypatch, tmp_path)
    monkeypatch.setattr(
        quick,
        "resolve_ollama_embedding_model",
        lambda **_k: calls.append("resolve") or "custom/embed:latest",
    )
    monkeypatch.setattr(
        quick,
        "ensure_ollama_embedding_model",
        lambda model_name, **_k: calls.append(f"pull:{model_name}"),
    )
    code = quick.run_quickstart(_config(quick, skip_stack_start=True))
    assert code == 0
    assert calls == ["resolve", "pull:custom/embed:latest"]


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
