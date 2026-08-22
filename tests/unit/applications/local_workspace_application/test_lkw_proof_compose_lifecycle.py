# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _REPO_ROOT / "applications" / "local_workspace_application" / "scripts"
_LIFECYCLE = _SCRIPTS_DIR / "lkw_proof_compose_lifecycle.py"
_PREFLIGHT = _SCRIPTS_DIR / "lkw_host_port_preflight.py"
_TRUSTED_ASK = _SCRIPTS_DIR / "run-lkw-ask-workspace-live-proof.py"
_OS_INTERACTION = _SCRIPTS_DIR / "run-lkw-os-interaction-proof.py"
_BACKGROUND_BAT = _SCRIPTS_DIR / "run-lkw-background-task-proof.bat"
_HOSTING_BAT = _SCRIPTS_DIR / "run-lkw-hosting-proof.bat"
_WATCHER_BAT = _SCRIPTS_DIR / "run-lkw-file-watcher-e2e-proof.bat"
_WATCHER_PY = _SCRIPTS_DIR / "run-lkw-file-watcher-e2e-proof.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lifecycle() -> ModuleType:
    return _load(_LIFECYCLE, "lkw_proof_compose_lifecycle")


@pytest.fixture(scope="module")
def preflight() -> ModuleType:
    return _load(_PREFLIGHT, "lkw_host_port_preflight_lifecycle")


@pytest.fixture(scope="module")
def trusted_ask() -> ModuleType:
    return _load(_TRUSTED_ASK, "run_lkw_ask_workspace_live_proof_lifecycle")


def test_non_destructive_teardown_command_has_no_volume_flag(
    lifecycle: ModuleType,
    preflight: ModuleType,
) -> None:
    stack = lifecycle.resolve_known_stack("lkw-background-task-proof")
    command = preflight.non_destructive_compose_down_args(stack)
    assert command[:4] == ["docker", "compose", "-p", "lkw-background-task-proof"]
    assert "down" in command
    assert "--remove-orphans" in command
    assert "-v" not in command
    assert "--volumes" not in command


def test_teardown_targets_exact_project(
    lifecycle: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: str,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = 300,
    ) -> subprocess.CompletedProcess[str]:
        captured.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    lifecycle.teardown_known_stack(
        "lkw-hosting-proof",
        run_command=fake_run,
    )
    assert len(captured) == 1
    command = captured[0]
    assert command[3] == "lkw-hosting-proof"
    assert any("docker-compose.mongodb.yml" in part for part in command)


def test_teardown_failure_raises_runtime_error(
    lifecycle: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "failed")

    with pytest.raises(RuntimeError, match="compose_down_failed"):
        lifecycle.teardown_known_stack(
            "lkw-os-interaction-proof",
            run_command=fake_run,
        )


def test_terminal_teardown_skipped_without_ownership(lifecycle: ModuleType) -> None:
    outcome = lifecycle.run_terminal_compose_teardown(
        compose_ownership_entered=False,
        teardown_fn=lambda: (_ for _ in ()).throw(AssertionError("must_not_run")),
    )
    assert outcome.attempted is False
    assert outcome.result == "SKIPPED"


def test_functional_pass_cleanup_failure_overall_fail(lifecycle: ModuleType) -> None:
    exit_code = lifecycle.finalize_exit_code_with_teardown(
        functional_pass=True,
        functional_exit_code=0,
        teardown_outcome=lifecycle.TerminalTeardownOutcome(
            attempted=True,
            result="FAIL",
            error_type="RuntimeError",
        ),
    )
    assert exit_code == 1


def test_functional_fail_cleanup_failure_preserves_fail(lifecycle: ModuleType) -> None:
    exit_code = lifecycle.finalize_exit_code_with_teardown(
        functional_pass=False,
        functional_exit_code=1,
        teardown_outcome=lifecycle.TerminalTeardownOutcome(
            attempted=True,
            result="FAIL",
            error_type="RuntimeError",
        ),
    )
    assert exit_code == 1


def test_trusted_ask_teardown_after_functional_failure(
    trusted_ask: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teardown_calls = {"n": 0}

    def fake_teardown() -> None:
        teardown_calls["n"] += 1

    monkeypatch.setattr(sys, "argv", [str(trusted_ask._SCRIPT_PATH)])
    monkeypatch.setattr(trusted_ask, "check_startup_host_port_preflight", lambda: None)
    monkeypatch.setattr(trusted_ask, "materialize_runtime_context", lambda: None)
    monkeypatch.setattr(trusted_ask, "start_canonical_stack", lambda: None)
    monkeypatch.setattr(trusted_ask, "ensure_ollama_model", lambda: None)
    monkeypatch.setattr(trusted_ask, "wait_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(
        trusted_ask,
        "verify_running_vector_store_is_qdrant",
        lambda: "qdrant",
    )
    def fake_request_json(
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> tuple[int, dict[str, object]]:
        if url.endswith("/search"):
            return 500, {"detail": "search_failed"}
        if url.endswith("/workspaces") and method == "POST":
            return 201, {"workspace_id": "ws-proof"}
        if url.endswith("/sync") and method == "POST":
            return 202, {"operation_id": "op-proof"}
        if url.endswith("/sources") and method == "POST":
            return 201, {"source_id": "src-proof"}
        if "/operations/" in url:
            return 200, {"operation_id": "op-proof", "status": "completed"}
        return 500, {"detail": "unexpected"}

    monkeypatch.setattr(trusted_ask, "_request_json", fake_request_json)
    monkeypatch.setattr(trusted_ask, "teardown_owned_compose_stack", fake_teardown)

    exit_code = trusted_ask.main()
    text = capsys.readouterr().out

    assert exit_code == 1
    assert teardown_calls["n"] == 1
    assert "failing_phase=search" in text
    assert "proof_teardown_attempted=true" in text
    assert "proof_teardown_result=PASS" in text
    assert "proof_teardown_failed" not in text


def test_trusted_ask_skip_docker_does_not_teardown(
    trusted_ask: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teardown_calls = {"n": 0}
    monkeypatch.setattr(
        sys,
        "argv",
        [str(trusted_ask._SCRIPT_PATH), "--skip-docker"],
    )
    monkeypatch.setattr(trusted_ask, "wait_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(
        trusted_ask,
        "verify_running_vector_store_is_qdrant",
        lambda: (_ for _ in ()).throw(RuntimeError("stop_early")),
    )
    monkeypatch.setattr(
        trusted_ask,
        "teardown_owned_compose_stack",
        lambda: teardown_calls.__setitem__("n", teardown_calls["n"] + 1),
    )

    trusted_ask.main()
    text = capsys.readouterr().out

    assert teardown_calls["n"] == 0
    assert "proof_teardown_attempted=true" not in text


def test_trusted_ask_success_teardown_before_pass_is_authoritative(
    trusted_ask: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    def fake_teardown() -> None:
        events.append("teardown")

    monkeypatch.setattr(sys, "argv", [str(trusted_ask._SCRIPT_PATH), "--skip-docker"])
    monkeypatch.setattr(trusted_ask, "wait_ready", lambda *_a, **_k: events.append("ready"))
    monkeypatch.setattr(
        trusted_ask,
        "verify_running_vector_store_is_qdrant",
        lambda: (_ for _ in ()).throw(RuntimeError("stop_after_ready")),
    )
    monkeypatch.setattr(trusted_ask, "teardown_owned_compose_stack", fake_teardown)

    trusted_ask.main()
    assert "teardown" not in events


def test_background_task_bat_uses_dedicated_compose_project() -> None:
    text = _BACKGROUND_BAT.read_text(encoding="utf-8")
    assert "LKW_COMPOSE_PROJECT=lkw-background-task-proof" in text
    assert '-p "%LKW_COMPOSE_PROJECT%"' in text
    assert "LKW_COMPOSE_OWNERSHIP_ENTERED=true" in text
    assert "teardown --stack-id lkw-background-task-proof" in text
    assert "--skip-docker" in text


def test_hosting_bat_uses_dedicated_compose_project_and_teardown() -> None:
    text = _HOSTING_BAT.read_text(encoding="utf-8")
    assert "LKW_COMPOSE_PROJECT=lkw-hosting-proof" in text
    assert "teardown --stack-id lkw-hosting-proof" in text


def test_file_watcher_bat_uses_dedicated_compose_project_and_teardown() -> None:
    text = _WATCHER_BAT.read_text(encoding="utf-8")
    assert "LKW_COMPOSE_PROJECT=lkw-file-watcher-e2e-proof" in text
    assert "teardown --stack-id lkw-file-watcher-e2e-proof" in text
    assert "Stack left running for inspection" not in text


def test_file_watcher_python_compose_command_uses_dedicated_project() -> None:
    watcher = _load(_WATCHER_PY, "run_lkw_file_watcher_e2e_proof_lifecycle")
    command = watcher.build_compose_command(
        "ps",
        base_compose=watcher._DEFAULT_BASE_COMPOSE,
        kafka_compose=watcher._DEFAULT_KAFKA_COMPOSE,
        watcher_compose=watcher._DEFAULT_WATCHER_COMPOSE,
        mongodb_compose=watcher._DEFAULT_MONGODB_COMPOSE,
    )
    assert command[3] == "lkw-file-watcher-e2e-proof"


def test_os_interaction_compose_args_use_dedicated_project() -> None:
    interaction = _load(_OS_INTERACTION, "run_lkw_os_interaction_proof_lifecycle")
    command = interaction._compose_args([interaction._BASE_COMPOSE, interaction._MONGODB_COMPOSE])
    assert command[:4] == ["docker", "compose", "-p", "lkw-os-interaction-proof"]
