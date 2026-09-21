# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W3 — Tools/WebSearch must not resolve Nexus (static/dynamic/lazy)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.persisted_run_trace import PersistedRun, RunMetadata, RunStats
from intergrax.tools.registry.runtime_bindings import RunTraceReaderBinding
from intergrax.websearch.contracts.routing_snapshot_sync import WebSearchLlmRoutingSnapshotSync
from tests.qualification.harness_01.nexus_boundary_detector import (
    file_has_dynamic_nexus_import,
    file_has_lazy_nexus_module_resolution,
    file_imports_nexus_module,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS_ROOT = _REPO_ROOT / "intergrax" / "tools"
_WEBSEARCH_ROOT = _REPO_ROOT / "intergrax" / "websearch"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _collect_nexus_offenders(root: Path) -> list[str]:
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
        if file_imports_nexus_module(source):
            offenders.append(rel)
        if file_has_dynamic_nexus_import(source) and rel not in offenders:
            offenders.append(rel)
        if file_has_lazy_nexus_module_resolution(source) and rel not in offenders:
            offenders.append(rel)
    return offenders


def test_harness_01_w3_intergrax_tools_has_zero_nexus_resolution() -> None:
    offenders = _collect_nexus_offenders(_TOOLS_ROOT)
    assert offenders == [], "intergrax/tools must not import or resolve intergrax.runtime.nexus:\n" + "\n".join(
        offenders
    )


def test_harness_01_w3_intergrax_websearch_has_zero_nexus_resolution() -> None:
    offenders = _collect_nexus_offenders(_WEBSEARCH_ROOT)
    assert offenders == [], (
        "intergrax/websearch must not import or resolve intergrax.runtime.nexus:\n" + "\n".join(offenders)
    )


class _StubTraceReader:
    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        return PersistedRun(
            metadata=RunMetadata(
                run_id=run_id,
                session_id="s",
                user_id="u",
                tenant_id=tenant_id,
                started_at_utc="2026-01-01T00:00:00Z",
                stats=RunStats(duration_ms=1, llm_usage={}),
            ),
            events=[],
        )

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> list:
        return []


def test_harness_01_w3_run_trace_reader_binding_pluginable_without_nexus() -> None:
    reader: RunTraceReaderBinding = _StubTraceReader()
    persisted = reader.read_run("r1", "t1")
    assert persisted.metadata.run_id == "r1"


class _NoOpRoutingSync:
    def sync_before_llm_call(self, *, run_id: str | None) -> None:
        _ = run_id


def test_harness_01_w3_websearch_routing_sync_port_pluginable_without_nexus() -> None:
    port: WebSearchLlmRoutingSnapshotSync = _NoOpRoutingSync()
    port.sync_before_llm_call(run_id="run-x")
