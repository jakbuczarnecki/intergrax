# © Artur Czarnecki. All rights reserved.

"""Tests for python -m local_workspace_application.file_watcher entrypoint."""

from __future__ import annotations

import json

import pytest

from local_workspace_application.file_watcher import FileWatcherSidecarResult
from local_workspace_application.file_watcher.__main__ import main

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    ("result", "expected_code"),
    [
        (
            FileWatcherSidecarResult(
                exit_kind="clean_stop",
                exit_code=0,
                final_checkpoint_saved=True,
            ),
            0,
        ),
        (
            FileWatcherSidecarResult(
                exit_kind="configuration_error",
                exit_code=2,
                error_id="file_watcher_identity_not_configured",
            ),
            2,
        ),
        (
            FileWatcherSidecarResult(
                exit_kind="checkpoint_failed",
                exit_code=1,
                error_id="checkpoint_write_failed",
            ),
            1,
        ),
        (
            FileWatcherSidecarResult(
                exit_kind="runtime_failed",
                exit_code=1,
                error_id="file_watcher_runtime_failed",
            ),
            1,
        ),
    ],
)
def test_main_prints_safe_sorted_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    result: FileWatcherSidecarResult,
    expected_code: int,
) -> None:
    monkeypatch.setattr(
        "local_workspace_application.file_watcher.__main__.run_local_workspace_file_watcher_sidecar",
        lambda: result,
    )

    code = main()
    captured = capsys.readouterr()

    assert code == expected_code
    assert captured.out.count("{") == 1
    parsed = json.loads(captured.out.strip())
    assert parsed == result.model_dump(mode="json")
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert "Traceback" not in captured.out
    assert "Exception" not in captured.out
    assert "/tmp/" not in captured.out
    assert "C:\\" not in captured.out
