# © Artur Czarnecki. All rights reserved.

"""PLUG-04 — RAG public EP typed load evidence."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_RAG_CHUNKERS,
    EP_RAG_RETRIEVERS,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.rag.bootstrap.entry_point_load import (
    collect_rag_plugin_load_evidence,
    register_rag_chunker_entry_points,
)
from intergrax.rag.document_splitters.registry.strategy_registry import (
    ChunkingStrategyRegistry,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_ep_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _BrokenChunker:
    pass


def test_rag_chunker_report_rejects_invalid_sibling_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "broken",
                f"{__name__}:_BrokenChunker",
                EP_RAG_CHUNKERS,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    report = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert report.group == EP_RAG_CHUNKERS
    assert report.registered_count == 0
    assert len(report.rejected) == 1
    assert report.rejected[0].reason_code == PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert report.failed == ()


def test_rag_chunker_report_ordering_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("z_bad", f"{__name__}:_BrokenChunker", EP_RAG_CHUNKERS),
            _EntryPoint("a_bad", f"{__name__}:_BrokenChunker", EP_RAG_CHUNKERS),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    first = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )
    reset_entry_point_spec_cache_for_tests()
    second = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert first.rejected == second.rejected


def test_collect_rag_evidence_discovery_disabled_returns_empty_reports() -> None:
    evidence = collect_rag_plugin_load_evidence(discover_entry_points=False)

    assert evidence.chunker_report.accepted == ()
    assert evidence.retriever_report.accepted == ()
    assert evidence.reranker_report.accepted == ()
    assert evidence.chunker_report.group == EP_RAG_CHUNKERS
    assert evidence.retriever_report.group == EP_RAG_RETRIEVERS


def test_rag_load_failure_isolated_in_failed_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("broken_load", "not-a-valid-target", EP_RAG_CHUNKERS)]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    report = register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )

    assert report.rejected == ()
    assert len(report.failed) == 1
    assert report.failed[0].error is not None
