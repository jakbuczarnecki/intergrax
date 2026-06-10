# © Artur Czarnecki. All rights reserved.

"""Scorecard auto-sync from AUDIT-IDEAL plan rows (AUDIT-IDEAL-32.2)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_MASTER_REGISTER_START = "## Master register"
_MASTER_REGISTER_END = "## Domain routing"
_ROW_PATTERN = re.compile(r"^\|\s*AUDIT-IDEAL-[\w.]+\s*\|", re.MULTILINE)
_STATUS_LINE = re.compile(
    r"\*\*(\d+)/(\d+) Done\*\*\s*·\s*\*\*(\d+) Deferred §6\.3\*\*\s*·\s*\*\*(\d+) Planned\*\*"
)


def _master_register_section(register_text: str) -> str:
    start = register_text.find(_MASTER_REGISTER_START)
    end = register_text.find(_MASTER_REGISTER_END)
    if start == -1 or end == -1 or end <= start:
        return register_text
    return register_text[start:end]


def _row_status(line: str) -> str | None:
    if "| **Deferred" in line:
        return "deferred"
    if "| **Done**" in line or "| **Done** " in line:
        return "done"
    if "| Planned |" in line:
        return "planned"
    return None


@dataclass(frozen=True, slots=True)
class AuditIdealScorecardSync:
    done_count: int
    deferred_count: int
    planned_count: int
    total_tasks: int
    harness_l3_layers: int
    total_layers: int
    in_sync: bool


def parse_audit_ideal_register(register_text: str) -> tuple[int, int, int, int]:
    section = _master_register_section(register_text)
    done = deferred = planned = 0
    for line in section.splitlines():
        if not _ROW_PATTERN.match(line):
            continue
        status = _row_status(line)
        if status == "done":
            done += 1
        elif status == "deferred":
            deferred += 1
        elif status == "planned":
            planned += 1
    return done, deferred, planned, done + deferred + planned


def parse_register_status_line(register_text: str) -> tuple[int, int, int, int] | None:
    match = _STATUS_LINE.search(register_text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def build_audit_ideal_scorecard_sync(
    *,
    register_text: str,
    harness_l3_layers: int = 32,
    total_layers: int = 32,
) -> AuditIdealScorecardSync:
    done, deferred, planned, total_tasks = parse_audit_ideal_register(register_text)
    status_line = parse_register_status_line(register_text)
    in_sync = status_line == (done, total_tasks, deferred, planned) if status_line is not None else True
    return AuditIdealScorecardSync(
        done_count=done,
        deferred_count=deferred,
        planned_count=planned,
        total_tasks=total_tasks,
        harness_l3_layers=harness_l3_layers,
        total_layers=total_layers,
        in_sync=in_sync,
    )


def write_scorecard_sync_artifact(repo_root: Path, sync: AuditIdealScorecardSync) -> Path:
    output = repo_root / "build" / "harness_baseline" / "audit_ideal_scorecard_sync.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "audit_ideal_done": sync.done_count,
        "audit_ideal_deferred": sync.deferred_count,
        "audit_ideal_planned": sync.planned_count,
        "audit_ideal_total": sync.total_tasks,
        "harness_l3_layers": sync.harness_l3_layers,
        "harness_total_layers": sync.total_layers,
        "in_sync": sync.in_sync,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_scorecard_sync(repo_root: Path) -> AuditIdealScorecardSync:
    register = repo_root / "docs" / "plan" / "AUDIT_IDEAL_2026.md"
    return build_audit_ideal_scorecard_sync(register_text=register.read_text(encoding="utf-8"))
