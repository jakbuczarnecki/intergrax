# © Artur Czarnecki. All rights reserved.

"""Unit tests for redacted LKW pipeline proof summary projection (LKW-3C)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_workspace_application.serving.proof_summary import (
    LKW_PIPELINE_CAPABILITY,
    LKW_PROOF_SUMMARY_KEY,
    attach_lkw_proof_summary_metadata,
    build_lkw_proof_summary,
)

_PIPELINE_AGENTS = ("local_indexer", "local_search", "local_synthesizer")

_UNSAFE_KEYS = frozenset(
    {
        "query_text",
        "text",
        "content",
        "raw_chunks",
        "chunks",
        "document",
        "documents",
        "full_trace",
        "agent_run_trace",
    }
)


def _pipeline_metadata() -> dict[str, Any]:
    return {
        "application_run_summary.v1": {
            "schema_version": "application_run_summary.v1",
            "terminal_status": "succeeded",
            "agent_invocations": [
                {"agent_id": "local_indexer", "total_tool_calls": 1},
                {"agent_id": "local_search", "total_tool_calls": 1},
                {"agent_id": "local_synthesizer", "total_tool_calls": 1},
            ],
        },
        "lkw_evidence.v1": {
            "schema_version": "lkw_evidence.v1",
            "capability": LKW_PIPELINE_CAPABILITY,
            "diagnostics": {
                "lkw.search_summary.v1": {
                    "num_results": 1,
                    "evidence_count": 1,
                    "source_refs": ["fixture.txt"],
                },
                "lkw.synthesize_summary.v1": {
                    "shadow_write": True,
                    "content_missing": False,
                    "artifact_path": "pipeline-synthesis-draft.md",
                    "write_status": "written",
                },
            },
        },
        "runtime_event_summary.v1": {
            "schema_version": "runtime_event_summary.v1",
            "tool_events": {"total": 3},
        },
        "run_artifact_bundle.v1": {
            "schema_version": "run_artifact_bundle.v1",
            "workspace": [{"artifact_id": "art-1", "workspace_id": "ws-1"}],
        },
    }


@pytest.mark.unit
def test_build_lkw_proof_summary_returns_none_for_non_pipeline_capability() -> None:
    metadata = _pipeline_metadata()
    assert build_lkw_proof_summary(metadata, capability="local.workspace.search") is None


@pytest.mark.unit
def test_build_lkw_proof_summary_pipeline_shape() -> None:
    summary = build_lkw_proof_summary(_pipeline_metadata(), capability=LKW_PIPELINE_CAPABILITY)
    assert summary is not None
    assert summary["schema_version"] == LKW_PROOF_SUMMARY_KEY
    assert summary["capability"] == LKW_PIPELINE_CAPABILITY
    assert summary["status"] == "passed"
    assert summary["agent_order"] == list(_PIPELINE_AGENTS)
    assert summary["tool_calls_by_agent"] == {
        "local_indexer": 1,
        "local_search": 1,
        "local_synthesizer": 1,
    }
    assert summary["evidence"]["present"] is True
    assert summary["evidence"]["count"] >= 1
    assert summary["evidence"]["source_refs_present"] is True
    assert summary["synthesis"]["shadow_write"] is True
    assert summary["synthesis"]["content_missing"] is False
    assert summary["synthesis"]["artifact_present"] is True
    assert summary["artifact"]["bundle_present"] is True
    assert summary["artifact"]["workspace_refs_count"] >= 1
    assert summary["safety"]["raw_trace_exposed"] is False
    assert summary["safety"]["raw_content_exposed"] is False


@pytest.mark.unit
def test_build_lkw_proof_summary_does_not_include_unsafe_keys() -> None:
    summary = build_lkw_proof_summary(_pipeline_metadata(), capability=LKW_PIPELINE_CAPABILITY)
    assert summary is not None
    serialized = json.dumps(summary)
    for key in _UNSAFE_KEYS:
        assert f'"{key}"' not in serialized


@pytest.mark.unit
def test_attach_lkw_proof_summary_metadata_only_for_pipeline() -> None:
    metadata = _pipeline_metadata()
    attach_lkw_proof_summary_metadata(metadata, capability="local.workspace.index")
    assert LKW_PROOF_SUMMARY_KEY not in metadata

    attach_lkw_proof_summary_metadata(metadata, capability=LKW_PIPELINE_CAPABILITY)
    assert LKW_PROOF_SUMMARY_KEY in metadata


@pytest.mark.unit
def test_build_lkw_proof_summary_failed_when_agent_order_wrong() -> None:
    metadata = _pipeline_metadata()
    metadata["application_run_summary.v1"]["agent_invocations"] = [
        {"agent_id": "local_indexer", "total_tool_calls": 1},
    ]
    summary = build_lkw_proof_summary(metadata, capability=LKW_PIPELINE_CAPABILITY)
    assert summary is not None
    assert summary["status"] == "failed"
