# © Artur Czarnecki. All rights reserved.

"""Harness shadow evaluation trend export (W-OPS.11, V-EVAL.4)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationReleaseSnapshot,
    EvaluationRegistryTrendReport,
    build_evaluation_registry_trend_report,
)
from intergrax.runtime.architecture.online_evaluation import (
    OnlineEvaluationBatch,
    append_online_evaluation_to_trend,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    OnlineEvaluationRegistry,
    default_online_evaluation_registry,
)


class EvaluationSnapshotArchive(BaseModel):
    schema_version: str = "1.0.0"
    snapshots: list[EvaluationReleaseSnapshot] = Field(default_factory=list)


def default_snapshots_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "architecture_hardening" / "evaluation_release_snapshots.json"


def load_evaluation_release_snapshots(path: Path | None = None) -> list[EvaluationReleaseSnapshot]:
    target = path or default_snapshots_path()
    if not target.is_file():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    archive = EvaluationSnapshotArchive.model_validate(payload)
    return archive.snapshots


def save_evaluation_release_snapshots(
    snapshots: list[EvaluationReleaseSnapshot],
    path: Path | None = None,
) -> Path:
    target = path or default_snapshots_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = EvaluationSnapshotArchive(snapshots=snapshots)
    target.write_text(archive.model_dump_json(indent=2), encoding="utf-8")
    return target


def export_shadow_evaluation_trend(
    release_id: str,
    *,
    registry: OnlineEvaluationRegistry | None = None,
    snapshots_path: Path | None = None,
    clear_registry_after_export: bool = True,
) -> EvaluationRegistryTrendReport:
    """
    Flush shadow observations from ``registry`` into a release snapshot and rebuild trends.

    Observations are grouped into one ``OnlineEvaluationBatch`` per ``release_id`` export call.
    """
    resolved_registry = registry or default_online_evaluation_registry()
    observations = resolved_registry.list_observations()
    batch = OnlineEvaluationBatch(release_id=release_id, observations=observations)
    existing = load_evaluation_release_snapshots(snapshots_path)
    snapshot, _comparisons = append_online_evaluation_to_trend(
        existing_snapshots=existing,
        batch=batch,
    )
    updated_snapshots = [*existing, snapshot]
    save_evaluation_release_snapshots(updated_snapshots, snapshots_path)
    if clear_registry_after_export:
        resolved_registry.clear()
    return build_evaluation_registry_trend_report(updated_snapshots)
