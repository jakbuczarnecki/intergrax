# © Artur Czarnecki. All rights reserved.

"""Operational release cycle counter for W-OPS L3 evidence."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ReleaseCycleRecord(BaseModel):
    cycle_id: str
    gate_green: bool = True
    notes: str = ""


class ReleaseCycleTracker(BaseModel):
    schema_version: str = "1.0.0"
    cycles: list[ReleaseCycleRecord] = Field(default_factory=list)

    @property
    def completed_count(self) -> int:
        return len(self.cycles)


def default_tracker_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "architecture_hardening" / "release_cycles.json"


def load_release_cycle_tracker(path: Path | None = None) -> ReleaseCycleTracker:
    target = path or default_tracker_path()
    if not target.is_file():
        return ReleaseCycleTracker()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return ReleaseCycleTracker.model_validate(payload)


def save_release_cycle_tracker(tracker: ReleaseCycleTracker, path: Path | None = None) -> Path:
    target = path or default_tracker_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tracker.model_dump_json(indent=2), encoding="utf-8")
    return target
