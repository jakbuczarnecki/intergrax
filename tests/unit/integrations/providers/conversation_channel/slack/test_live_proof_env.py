# © Artur Czarnecki. All rights reserved.

"""Proof-script loading of LKW application Slack credentials from ``.env``."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_PROOF_SCRIPT = (
    _PROJECT_ROOT / "scripts" / "proof" / "slack_conversation_channel_live_proof.py"
)

_TOKEN_KEYS = (
    "INTERGRAX_SLACK_APP_TOKEN",
    "INTERGRAX_SLACK_BOT_TOKEN",
    "INTERGRAX_SLACK_PROOF_TIMEOUT_SECONDS",
)


def _load_proof_module(monkeypatch: pytest.MonkeyPatch, *, allow_dotenv: bool = False) -> ModuleType:
    """Import the proof script without permanently loading the developer ``.env``."""
    from dotenv import load_dotenv as real_load_dotenv

    if not allow_dotenv:
        # Block only the module-level developer .env load; restore for helper tests.
        monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: False)
    module_name = "slack_conversation_channel_live_proof_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _PROOF_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.load_dotenv = real_load_dotenv
    return module


@pytest.fixture
def clear_slack_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _TOKEN_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_proof_resolves_lkw_env_relative_to_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_proof_module(monkeypatch)
    expected = (
        _PROJECT_ROOT / "applications" / "local_workspace_application" / ".env"
    ).resolve()
    assert module._LKW_ENV_FILE.resolve() == expected
    assert module._REPO_ROOT.resolve() == _PROJECT_ROOT.resolve()


def test_proof_env_path_independent_of_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_proof_module(monkeypatch)
    expected = module._LKW_ENV_FILE.resolve()
    monkeypatch.chdir(tmp_path)
    assert module._LKW_ENV_FILE.resolve() == expected
    assert (Path.cwd() / "applications" / "local_workspace_application" / ".env") != expected


def test_lkw_env_values_become_visible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clear_slack_token_env: None,
) -> None:
    module = _load_proof_module(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "INTERGRAX_SLACK_APP_TOKEN=xapp-test-from-file\n"
        "INTERGRAX_SLACK_BOT_TOKEN=xoxb-test-from-file\n"
        "INTERGRAX_SLACK_PROOF_TIMEOUT_SECONDS=42\n",
        encoding="utf-8",
    )
    assert module._load_lkw_environment(env_file) is True
    assert os.environ["INTERGRAX_SLACK_APP_TOKEN"] == "xapp-test-from-file"
    assert os.environ["INTERGRAX_SLACK_BOT_TOKEN"] == "xoxb-test-from-file"
    assert os.environ["INTERGRAX_SLACK_PROOF_TIMEOUT_SECONDS"] == "42"


def test_process_environment_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clear_slack_token_env: None,
) -> None:
    module = _load_proof_module(monkeypatch)
    monkeypatch.setenv("INTERGRAX_SLACK_APP_TOKEN", "xapp-process-wins")
    monkeypatch.setenv("INTERGRAX_SLACK_BOT_TOKEN", "xoxb-process-wins")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "INTERGRAX_SLACK_APP_TOKEN=xapp-from-file\n"
        "INTERGRAX_SLACK_BOT_TOKEN=xoxb-from-file\n",
        encoding="utf-8",
    )
    assert module._load_lkw_environment(env_file) is True
    assert os.environ["INTERGRAX_SLACK_APP_TOKEN"] == "xapp-process-wins"
    assert os.environ["INTERGRAX_SLACK_BOT_TOKEN"] == "xoxb-process-wins"


def test_missing_env_file_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clear_slack_token_env: None,
) -> None:
    module = _load_proof_module(monkeypatch)
    missing = tmp_path / "does-not-exist.env"
    assert module._load_lkw_environment(missing) is False


@pytest.mark.asyncio
async def test_missing_tokens_return_live_proof_blocked(
    monkeypatch: pytest.MonkeyPatch,
    clear_slack_token_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_proof_module(monkeypatch)
    code = await module._run()
    captured = capsys.readouterr()
    assert code == 2
    assert "LIVE_PROOF_BLOCKED: missing INTERGRAX_SLACK_APP_TOKEN and/or INTERGRAX_SLACK_BOT_TOKEN" in (
        captured.out + captured.err
    )


@pytest.mark.asyncio
async def test_tokens_never_appear_in_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clear_slack_token_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_proof_module(monkeypatch)
    secret_app = "xapp-secret-must-not-leak"
    secret_bot = "xoxb-secret-must-not-leak"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"INTERGRAX_SLACK_APP_TOKEN={secret_app}\n"
        f"INTERGRAX_SLACK_BOT_TOKEN={secret_bot}\n",
        encoding="utf-8",
    )
    assert module._load_lkw_environment(env_file) is True
    # Force blocked path after load so we only exercise diagnostics, not live Slack.
    monkeypatch.delenv("INTERGRAX_SLACK_APP_TOKEN", raising=False)
    monkeypatch.delenv("INTERGRAX_SLACK_BOT_TOKEN", raising=False)
    code = await module._run()
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert code == 2
    assert secret_app not in combined
    assert secret_bot not in combined
    assert "configuration source:" in combined


def test_configuration_source_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_proof_module(monkeypatch)
    assert "LKW .env available" in module._configuration_source_message(env_available=True)
    assert "no LKW .env" in module._configuration_source_message(env_available=False)
