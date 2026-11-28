"""Read-only pytest plugin: records collected node IDs and phase outcomes (EQO qualification)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PLUGIN: _EqoNodeCapture | None = None


def pytest_addoption(parser) -> None:  # noqa: ANN001
    parser.addoption(
        "--eqo-node-report",
        action="store",
        default=None,
        help="Path for eqo pytest node capture JSON (read-only sidecar).",
    )
    parser.addoption(
        "--eqo-source-root",
        action="store",
        default=None,
        help="Repository root for repository-relative test paths in node capture.",
    )


def pytest_configure(config) -> None:  # noqa: ANN001
    global _PLUGIN
    path = config.option.eqo_node_report
    if not path:
        return
    source_root = config.option.eqo_source_root
    src = Path(source_root).resolve() if source_root else Path.cwd().resolve()
    _PLUGIN = _EqoNodeCapture(Path(path), source_root=src)
    config.pluginmanager.register(_PLUGIN, "eqo_nodecapture")


def _rel_test_path(item_path: Path, source_root: Path) -> str:
    try:
        return item_path.resolve().relative_to(source_root).as_posix()
    except ValueError:
        return item_path.as_posix()


class _EqoNodeCapture:
    def __init__(self, report_path: Path, *, source_root: Path) -> None:
        self.report_path = report_path
        self.source_root = source_root
        self.collected: list[dict[str, Any]] = []
        self.phase_reports: list[dict[str, Any]] = []
        self.collection_errors: list[dict[str, Any]] = []
        self._nodes: dict[str, dict[str, Any]] = {}

    def pytest_collection_modifyitems(self, session, config, items) -> None:  # noqa: ANN001
        self.collected = []
        for item in items:
            file_path = _rel_test_path(item.path, self.source_root)
            entry = {
                "node_id": item.nodeid,
                "file_path": file_path,
                "outcome": "COLLECTED",
            }
            self.collected.append(entry)
            self._nodes[item.nodeid] = entry

    def pytest_runtest_makereport(self, item, call) -> None:  # noqa: ANN001
        if call.when == "call":
            item.user_properties.append(("eqo_node_id", item.nodeid))
            item.user_properties.append(("eqo_file_path", _rel_test_path(item.path, self.source_root)))

    def pytest_runtest_logreport(self, report) -> None:  # noqa: ANN001
        nodeid = report.nodeid
        when = report.when
        outcome = report.outcome.upper() if report.outcome else "UNKNOWN"
        meta: dict[str, Any] = {"when": when, "outcome": outcome}
        if getattr(report, "wasxfail", None):
            meta["wasxfail"] = str(report.wasxfail)
        if report.longrepr and outcome in ("FAILED", "ERROR"):
            meta["longrepr_snippet"] = str(report.longrepr)[:200]
        self.phase_reports.append({"node_id": nodeid, **meta})
        node = self._nodes.get(nodeid)
        if node is None:
            node = {"node_id": nodeid, "file_path": "", "outcome": outcome}
            self._nodes[nodeid] = node
        phases = node.setdefault("phases", {})
        phases[when] = meta
        if when == "call":
            node["outcome"] = outcome
            if meta.get("wasxfail"):
                node["wasxfail"] = meta["wasxfail"]

    def pytest_collection_finish(self, session) -> None:  # noqa: ANN001
        if session.testsfailed and not self.collected:
            self.collection_errors.append({"message": "collection_failed", "failed": session.testsfailed})

    def pytest_sessionfinish(self, session, exitstatus) -> None:  # noqa: ANN001
        payload = {
            "schema": "eqo.pytest_node_capture.v1",
            "exitstatus": exitstatus,
            "node_records": list(self._nodes.values()) if self._nodes else self.collected,
            "phase_reports": self.phase_reports,
            "collection_errors": self.collection_errors,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
