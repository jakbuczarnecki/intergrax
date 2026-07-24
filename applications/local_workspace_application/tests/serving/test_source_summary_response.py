# © Artur Czarnecki. All rights reserved.

"""Serving-layer source summary projection (Windows/POSIX locators)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from local_workspace_application.serving.workspace_routes import (
    _source_summary_response,
)
from local_workspace_application.workspaces.models import (
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)

pytestmark = pytest.mark.unit


def _source(*, path: str) -> WorkspaceSource:
    return WorkspaceSource(
        source_id="src-1",
        workspace_id="ws-1",
        tenant_id="tenant-a",
        source_type=WorkspaceSourceType.LOCAL_FOLDER,
        path=path,
        recursive=True,
        status=WorkspaceSourceStatus.REGISTERED,
        created_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
        last_sync_at=None,
    )


def test_windows_path_projected_to_folder_name_only() -> None:
    summary = _source_summary_response(
        _source(path=r"C:\Users\Artur\Private\Client-X\Contracts")
    )
    payload = summary.model_dump(mode="json")
    assert "path" not in payload
    assert payload["label"] == "Contracts"
    for fragment in ("C:\\", "Users", "Artur", "Private", "Client-X"):
        assert fragment not in payload["label"]


def test_posix_path_projected_to_folder_name_only() -> None:
    summary = _source_summary_response(
        _source(path="/home/user/projects/specifications")
    )
    payload = summary.model_dump(mode="json")
    assert "path" not in payload
    assert payload["label"] == "specifications"
    assert "home" not in payload["label"]
    assert "user" not in payload["label"]


def test_root_only_path_uses_fallback() -> None:
    summary = _source_summary_response(_source(path="/"))
    assert summary.label == "Local folder"
