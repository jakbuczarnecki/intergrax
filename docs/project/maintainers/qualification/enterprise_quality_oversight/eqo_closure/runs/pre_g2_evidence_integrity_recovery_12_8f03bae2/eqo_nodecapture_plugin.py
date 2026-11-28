"""Read-only pytest plugin: records collected node IDs and phase outcomes (EQO qualification)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PLUGIN = None
_REPORT_PATH: Path | None = None


def pytest_configure(config) -> None:  # noqa: ANN001
    global _PLUGIN, _REPORT_PATH
    path = config.option.eqo_node_report
    if not path:
        return
    _REPORT_PATH = Path(path)
    _PLUGIN = _EqoNodeCapture(_REPORT_PATH)
    config.pluginmanager.register(_PLUGIN, "eqo_nodecapture")


def pytest_addoption(parser) -> None:  # noqa: ANN001
    parser.addoption(
        "--eqo-node-report",
        action="store",
        default=None,
        help="Path for eqo pytest node capture JSON (read-only sidecar).",
    )


class _EqoNodeCapture:
    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path
        self.records: list[dict[str, Any]] = []
        self.collection_errors: list[dict[str, Any]] = []

    def pytest_collection_modifyitems(self, session, config, items) -> None:  # noqa: ANN001
        for item in items:
            self.records.append(
                {
                    "node_id": item.nodeid,
                    "file_path": str(item.path).replace("\\", "/").split("tests/", 1)[-1],
                    "file_path_resolved": item.path.as_posix(),
                    "outcome": "COLLECTED",
                }
            )

    def pytest_runtest_logreport(self, report) -> None:  # noqa: ANN001
        if report.when != "call":
            return
        nodeid = report.nodeid
        outcome = report.outcome.upper()
        skip = getattr(report, "wasxfail", None)
        meta: dict[str, Any] = {}
        if skip:
            meta["wasxfail"] = str(skip)
        if report.longrepr and outcome in ("FAILED", "ERROR"):
            meta["longrepr_snippet"] = str(report.longrepr)[:200]
        for rec in self.records:
            if rec["node_id"] == nodeid:
                rec["outcome"] = outcome
                rec.update(meta)
                return
        self.records.append({"node_id": nodeid, "outcome": outcome, **meta})

    def pytest_collection_finish(self, session) -> None:  # noqa: ANN001
        if session.testsfailed and not self.records:
            self.collection_errors.append({"message": "collection_failed", "failed": session.testsfailed})

    def pytest_sessionfinish(self, session, exitstatus) -> None:  # noqa: ANN001
        payload = {
            "schema": "eqo.pytest_node_capture.v1",
            "exitstatus": exitstatus,
            "node_records": self.records,
            "collection_errors": self.collection_errors,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
