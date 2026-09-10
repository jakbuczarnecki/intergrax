# © Artur Czarnecki. All rights reserved.

"""P0 — platform-wide execution bypass inventory static gates."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
_ARCH_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md"
)
_INVENTORY_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)

_FROZEN_CHILD_RUNNER_IMPORTS = frozenset(
    {
        "intergrax/runtime/execution/delegated_subtask_child_port.py",
        "intergrax/runtime/execution/execution_work_port.py",
        "intergrax/runtime/nexus/execution/graph_executor.py",
        "intergrax/applications/_shared/production_agent_capability_runtime.py",
    },
)

_PRODUCTION_AGENT_CAPABILITY_RUNTIME = (
    _INTERGRAX_ROOT / "applications" / "_shared" / "production_agent_capability_runtime.py"
)
_COMPENSATION_WORKER = _INTERGRAX_ROOT / "agents" / "persistence" / "compensation_queue_worker.py"
_SCENARIO_BASELINE = _INTERGRAX_ROOT / "applications" / "_shared" / "scenario_runtime_baseline.py"
_HOST_TASK = _INTERGRAX_ROOT / "runtime" / "execution" / "host_task.py"

_EXPECTED_ENTRYPOINT_COUNT = 22
_VERDICT_COLUMN_INDEX = 10
_SEVERITY_COLUMN_INDEX = 11
_METRICS_VERDICT_LABELS = (
    "CANONICAL",
    "CANONICAL WITH GAP",
    "LEGACY BUT NON-PRODUCTION",
    "UNSUPPORTED / DEAD",
    "BYPASS",
    "AMBIGUOUS",
)
_BYPASS_ROW_TO_PROVEN_ID = {
    "EP-15": "BY-01",
    "EP-16": "BY-02",
}


def _inventory_doc_text() -> str:
    return _INVENTORY_DOC.read_text(encoding="utf-8")


def _section_slice(text: str, heading: str, *, until_heading_prefix: str = "## ") -> str:
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = len(rest)
    for match in re.finditer(rf"^{until_heading_prefix}", rest, flags=re.MULTILINE):
        if match.start() == 0:
            continue
        end = match.start()
        break
    return rest[:end]


def _normalize_verdict_cell(raw: str) -> str:
    cleaned = raw.strip().strip("*").strip()
    if cleaned.startswith("AMBIGUOUS"):
        return "AMBIGUOUS"
    return cleaned


def _parse_central_inventory_rows(text: str) -> list[dict[str, str]]:
    body = _section_slice(text, "## Central inventory")
    rows: list[dict[str, str]] = []
    for line in body.splitlines():
        if not re.match(r"^\| EP-\d{2} \|", line):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) <= _SEVERITY_COLUMN_INDEX:
            raise AssertionError(f"malformed central inventory row: {line[:120]!r}")
        ep_id = parts[0]
        verdict = _normalize_verdict_cell(parts[_VERDICT_COLUMN_INDEX])
        severity = parts[_SEVERITY_COLUMN_INDEX].strip().strip("*").strip()
        rows.append({"id": ep_id, "verdict": verdict, "severity": severity})
    return rows


def _parse_metrics_verdict_counts(text: str) -> dict[str, int]:
    body = _section_slice(text, "## Metrics (summary)", until_heading_prefix="## Central")
    counts: dict[str, int] = {}
    for label in _METRICS_VERDICT_LABELS:
        pattern = rf"^\| {re.escape(label)} \| (\d+) \|"
        match = re.search(pattern, body, flags=re.MULTILINE)
        assert match, f"Metrics (summary) missing row for {label!r}"
        counts[label] = int(match.group(1))
    return counts


def _central_inventory_verdict_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row["verdict"] for row in rows)


def _rel_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _modules_importing_child_execution_runner() -> set[str]:
    found: set[str] = set()
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "ChildExecutionRunner" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.execution.child":
                for alias in node.names:
                    if alias.name == "ChildExecutionRunner":
                        found.add(_rel_posix(path))
                        break
    return found


def test_p0_qualification_documents_present() -> None:
    assert _ARCH_DOC.is_file(), "architecture qualification doc missing"
    assert _INVENTORY_DOC.is_file(), "P0 inventory qualification doc missing"
    inventory = _INVENTORY_DOC.read_text(encoding="utf-8")
    assert "## Central inventory" in inventory
    assert "BY-01" in inventory
    assert "BY-02" in inventory


def test_p0_frozen_child_execution_runner_import_surface() -> None:
    """Only proven canonical adapters + one documented composition bypass may import ChildExecutionRunner."""
    found = _modules_importing_child_execution_runner()
    assert found == _FROZEN_CHILD_RUNNER_IMPORTS, (
        "ChildExecutionRunner import surface changed — update P0 inventory and this gate together. "
        f"found={sorted(found)} expected={sorted(_FROZEN_CHILD_RUNNER_IMPORTS)}"
    )


def test_p0_documented_direct_child_bypass_still_at_composition_default() -> None:
    """BY-01 evidence anchor — default factory must remain visible until U4 closes it."""
    source = _PRODUCTION_AGENT_CAPABILITY_RUNTIME.read_text(encoding="utf-8")
    assert "DelegatedSubtaskServiceFactory" in source
    assert "as_child_execution_port(ChildExecutionRunner" in source


def test_p0_compensation_worker_not_execution_runtime_entry() -> None:
    """BY-02 evidence anchor — worker drains tools without ExecutionRuntime admission."""
    source = _COMPENSATION_WORKER.read_text(encoding="utf-8")
    assert "drain_pending_compensation_jobs" in source
    assert "DeclarativeToolInvoker" in source
    assert "ExecutionRuntime" not in source
    assert "HostTaskExecutionPort" not in source


def test_p0_scenario_entry_uses_host_task_execution() -> None:
    source = _SCENARIO_BASELINE.read_text(encoding="utf-8")
    assert "async def execute_scenario_task" in source
    assert "build_environment_host_task_execution" in source
    assert "host_execution.execute" in source


def test_p0_host_task_routes_through_execution_facade() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8")
    assert "ExecutionRuntime" in source
    assert "Execution(" in source
    assert "UnifiedTaskRunner" not in source


def test_p0_central_inventory_metrics_and_bypass_consistency() -> None:
    text = _inventory_doc_text()
    rows = _parse_central_inventory_rows(text)
    assert len(rows) == _EXPECTED_ENTRYPOINT_COUNT
    ep_ids = [row["id"] for row in rows]
    assert len(ep_ids) == len(set(ep_ids)), f"duplicate EP IDs: {ep_ids}"

    inventory_counts = _central_inventory_verdict_counts(rows)
    assert sum(inventory_counts.values()) == _EXPECTED_ENTRYPOINT_COUNT

    metrics_counts = _parse_metrics_verdict_counts(text)
    for label in _METRICS_VERDICT_LABELS:
        assert metrics_counts[label] == inventory_counts[label], (
            f"Metrics {label}={metrics_counts[label]} != central inventory {inventory_counts[label]}"
        )

    bypass_rows = [row for row in rows if row["verdict"] == "BYPASS"]
    assert metrics_counts["BYPASS"] == len(bypass_rows)
    assert metrics_counts["AMBIGUOUS"] == inventory_counts["AMBIGUOUS"]

    proven_section = _section_slice(
        text,
        "## Bypass graph (proven)",
        until_heading_prefix="## Ambiguous",
    )
    assert "BY-03" not in proven_section
    assert "BY-01" in proven_section
    assert "BY-02" in proven_section

    for row in bypass_rows:
        proven_id = _BYPASS_ROW_TO_PROVEN_ID.get(row["id"])
        assert proven_id, f"missing proven bypass mapping for {row['id']}"
        assert proven_id in proven_section

    ambiguous_rows = [row for row in rows if row["verdict"] == "AMBIGUOUS"]
    assert len(ambiguous_rows) == 1
    assert ambiguous_rows[0]["id"] == "EP-17"
    ambiguous_section = _section_slice(
        text,
        "## Ambiguous execution paths requiring owner decision",
        until_heading_prefix="## Direct",
    )
    assert "EP-17" in ambiguous_section
    assert "NOT COUNTED AS PROVEN BYPASS" in ambiguous_section

    p0_bypasses = sum(1 for row in bypass_rows if row["severity"] == "P0")
    p1_bypasses = sum(1 for row in bypass_rows if row["severity"] == "P1")
    p2_gaps = sum(1 for row in rows if row["verdict"] == "CANONICAL WITH GAP")
    p3_legacy = sum(1 for row in rows if row["verdict"] == "LEGACY BUT NON-PRODUCTION")
    metrics_body = _section_slice(text, "## Metrics (summary)", until_heading_prefix="## Central")
    assert re.search(r"^\| P0 bypasses \| (\d+) \|", metrics_body, re.MULTILINE).group(1) == str(
        p0_bypasses
    )
    assert re.search(r"^\| P1 bypasses \| (\d+) \|", metrics_body, re.MULTILINE).group(1) == str(
        p1_bypasses
    )
    assert re.search(r"^\| P2 gaps \| (\d+) \|", metrics_body, re.MULTILINE).group(1) == str(p2_gaps)
    assert re.search(r"^\| P3 legacy cleanups \| (\d+) \|", metrics_body, re.MULTILINE).group(1) == str(
        p3_legacy
    )
    supported_match = re.search(
        r"^\| Supported execution bypasses \(production\) \| (\d+) \|",
        metrics_body,
        re.MULTILINE,
    )
    assert supported_match is not None
    assert int(supported_match.group(1)) == len(bypass_rows)
