# © Artur Czarnecki. All rights reserved.

"""Live/API smoke: typed LKW evidence visibility through POST /v1/local_workspace/run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_workspace_application.host.factory import create_local_workspace_backend_app

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/local_workspace"
_APP_SUMMARY_KEY = "application_run_summary.v1"
_EVIDENCE_KEY = "lkw_evidence.v1"

_FIXTURE_TEXT = "Intergrax LKW evidence smoke fixture — searchable paragraph."
_QUERY = "Intergrax LKW evidence smoke"
_COLLECTION_ID = "lkw-evidence-smoke-ws"

_UNSAFE_DIAGNOSTIC_KEYS = frozenset(
    {
        "query_text",
        "text",
        "content",
        "raw_chunks",
        "chunks",
        "document",
        "documents",
    }
)


@pytest.fixture
def lkw_smoke_workspace(tmp_path: Path) -> tuple[Path, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture_doc = workspace / "fixture.txt"
    fixture_doc.write_text(_FIXTURE_TEXT, encoding="utf-8")
    return fixture_doc, str(fixture_doc.resolve())


@pytest.fixture
def lkw_smoke_client(
    lkw_smoke_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    fixture_doc, fixture_path = lkw_smoke_workspace
    _ = fixture_doc
    workspace_root = fixture_doc.parent

    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(workspace_root.resolve()))
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", "true")

    return TestClient(create_local_workspace_backend_app())


def _post_run(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"{_PREFIX}/run", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("state") == "completed", body
    return body


def _metadata(body: Mapping[str, Any]) -> dict[str, Any]:
    metadata = body.get("metadata")
    assert isinstance(metadata, dict), body
    return metadata


def _assert_trace_not_exposed(metadata: dict[str, Any]) -> None:
    assert "full_trace" not in metadata
    assert "agent_run_trace" not in metadata


def _assert_app_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    assert _APP_SUMMARY_KEY in metadata
    summary = metadata[_APP_SUMMARY_KEY]
    assert isinstance(summary, dict)
    assert summary.get("schema_version") == "application_run_summary.v1"
    assert summary.get("terminal_status") is not None
    invocations = summary.get("agent_invocations")
    assert isinstance(invocations, list)
    assert invocations, "expected at least one agent invocation"
    return summary


def _assert_evidence_shell(
    metadata: dict[str, Any],
    *,
    capability: str,
) -> dict[str, Any]:
    assert _EVIDENCE_KEY in metadata
    evidence = metadata[_EVIDENCE_KEY]
    assert isinstance(evidence, dict)
    assert evidence.get("schema_version") == _EVIDENCE_KEY
    assert evidence.get("capability") == capability
    diagnostics = evidence.get("diagnostics")
    assert isinstance(diagnostics, dict)
    return evidence


def _assert_no_unsafe_diagnostic_keys(diagnostics: dict[str, Any]) -> None:
    for schema_id, payload in diagnostics.items():
        assert isinstance(payload, dict), schema_id
        leaked = _UNSAFE_DIAGNOSTIC_KEYS.intersection(payload.keys())
        assert not leaked, f"{schema_id} leaked unsafe keys: {sorted(leaked)}"


def _assert_raw_text_not_in_evidence(evidence: dict[str, Any], raw_text: str) -> None:
    serialized = json.dumps(evidence)
    assert raw_text not in serialized


def _tool_calls_for_capability(summary: dict[str, Any]) -> int | None:
    invocations = summary.get("agent_invocations") or []
    if not invocations:
        return None
    first = invocations[0]
    if not isinstance(first, dict):
        return None
    raw = first.get("total_tool_calls")
    return int(raw) if raw is not None else None


def test_lkw_evidence_live_smoke_index(
    lkw_smoke_client: TestClient,
    lkw_smoke_workspace: tuple[Path, str],
) -> None:
    _, fixture_path = lkw_smoke_workspace
    body = _post_run(
        lkw_smoke_client,
        {
            "message": "index fixture",
            "capability": "local.workspace.index",
            "metadata": {
                "source_paths": [fixture_path],
                "collection_id": _COLLECTION_ID,
            },
        },
    )
    metadata = _metadata(body)
    _assert_trace_not_exposed(metadata)
    summary = _assert_app_summary(metadata)
    evidence = _assert_evidence_shell(metadata, capability="local.workspace.index")

    index_diag = evidence["diagnostics"].get("lkw.index_summary.v1")
    assert isinstance(index_diag, dict), evidence["diagnostics"]
    for field in (
        "accepted_count",
        "rejected_count",
        "ingested_count",
        "chunk_count",
        "source_count",
    ):
        assert field in index_diag, index_diag

    _assert_no_unsafe_diagnostic_keys(evidence["diagnostics"])
    _assert_raw_text_not_in_evidence(evidence, _FIXTURE_TEXT)

    tool_calls = _tool_calls_for_capability(summary)
    assert tool_calls is not None and tool_calls > 0


def test_lkw_evidence_live_smoke_search(
    lkw_smoke_client: TestClient,
    lkw_smoke_workspace: tuple[Path, str],
) -> None:
    _, fixture_path = lkw_smoke_workspace
    _post_run(
        lkw_smoke_client,
        {
            "message": "index fixture",
            "capability": "local.workspace.index",
            "metadata": {
                "source_paths": [fixture_path],
                "collection_id": _COLLECTION_ID,
            },
        },
    )

    body = _post_run(
        lkw_smoke_client,
        {
            "message": _QUERY,
            "capability": "local.workspace.search",
            "metadata": {
                "query": _QUERY,
                "collection_id": _COLLECTION_ID,
                "top_k": 5,
            },
        },
    )
    metadata = _metadata(body)
    _assert_trace_not_exposed(metadata)
    _assert_app_summary(metadata)
    evidence = _assert_evidence_shell(metadata, capability="local.workspace.search")

    search_diag = evidence["diagnostics"].get("lkw.search_summary.v1")
    assert isinstance(search_diag, dict), evidence["diagnostics"]
    assert "num_results" in search_diag
    assert "evidence_count" in search_diag
    if search_diag.get("source_refs") is not None:
        assert isinstance(search_diag["source_refs"], list)
    if search_diag.get("query_digest") is not None:
        assert isinstance(search_diag["query_digest"], str)

    _assert_no_unsafe_diagnostic_keys(evidence["diagnostics"])
    _assert_raw_text_not_in_evidence(evidence, _FIXTURE_TEXT)


def test_lkw_evidence_live_smoke_synthesize(
    lkw_smoke_client: TestClient,
    lkw_smoke_workspace: tuple[Path, str],
) -> None:
    _, fixture_path = lkw_smoke_workspace
    _post_run(
        lkw_smoke_client,
        {
            "message": "index fixture",
            "capability": "local.workspace.index",
            "metadata": {
                "source_paths": [fixture_path],
                "collection_id": _COLLECTION_ID,
            },
        },
    )
    search_body = _post_run(
        lkw_smoke_client,
        {
            "message": _QUERY,
            "capability": "local.workspace.search",
            "metadata": {
                "query": _QUERY,
                "collection_id": _COLLECTION_ID,
                "top_k": 5,
            },
        },
    )
    search_metadata = _metadata(search_body)
    search_evidence = search_metadata[_EVIDENCE_KEY]["diagnostics"].get("lkw.search_summary.v1")
    assert isinstance(search_evidence, dict)

    body = _post_run(
        lkw_smoke_client,
        {
            "message": "synthesize draft",
            "capability": "local.workspace.synthesize",
            "metadata": {
                "shadow_workspace": True,
                "output_name": "synthesis-draft.md",
                "search_summary": {
                    "query": _QUERY,
                    "num_results": search_evidence.get("num_results", 1),
                    "evidence_count": search_evidence.get("evidence_count", 1),
                },
                "evidence": [
                    {
                        "text": _FIXTURE_TEXT,
                        "source_path": fixture_path,
                        "chunk_id": "chunk-smoke-1",
                    }
                ],
            },
        },
    )
    metadata = _metadata(body)
    _assert_trace_not_exposed(metadata)
    _assert_app_summary(metadata)
    evidence = _assert_evidence_shell(metadata, capability="local.workspace.synthesize")

    synth_diag = evidence["diagnostics"].get("lkw.synthesize_summary.v1")
    assert isinstance(synth_diag, dict), evidence["diagnostics"]
    for field in ("write_status", "shadow_write", "source_evidence_count"):
        assert field in synth_diag, synth_diag
    assert synth_diag["shadow_write"] is True
    assert isinstance(synth_diag["write_status"], str) and synth_diag["write_status"]
    assert synth_diag.get("artifact_path") or synth_diag.get("artifact_ref"), synth_diag
    if synth_diag.get("artifact_path") is not None:
        assert isinstance(synth_diag["artifact_path"], str)
    if synth_diag.get("artifact_ref") is not None:
        assert isinstance(synth_diag["artifact_ref"], str)
    if synth_diag.get("reason") is not None:
        assert isinstance(synth_diag["reason"], str)
    if synth_diag.get("content_missing") is not None:
        assert isinstance(synth_diag["content_missing"], bool)

    _assert_no_unsafe_diagnostic_keys(evidence["diagnostics"])
    _assert_raw_text_not_in_evidence(evidence, _FIXTURE_TEXT)
