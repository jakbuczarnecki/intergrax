# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.tools.providers.rag.index_lifecycle_service import (
    RAG_CHECK_INDEX_STATUS_TOOL_ID,
    RAG_GET_DOCUMENT_TOOL_ID,
    RAG_LIST_DOCUMENTS_TOOL_ID,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.tool_wiring import wire_local_workspace_tools
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.host.environment_profile import build_local_workspace_environment_profile
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime

pytestmark = pytest.mark.unit


def test_lkw_base_tool_profile_includes_t7_rag_and_document_tools() -> None:
    wiring = wire_local_workspace_tools()
    enabled = set(wiring.profile.enabled)
    for tool_id in (
        "document.parse_preview",
        RAG_LIST_DOCUMENTS_TOOL_ID,
        RAG_GET_DOCUMENT_TOOL_ID,
        RAG_CHECK_INDEX_STATUS_TOOL_ID,
    ):
        assert tool_id in enabled


def test_factory_runtime_receives_web_url_staging_read_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "factory-data"
    data_home.mkdir()
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))

    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
    )
    roots = runtime.env_wiring.tool_wiring.wiring_context.read_allowlist_roots
    assert settings.web_url_staging_dir in roots
    assert settings.managed_upload_staging_dir in roots
