# © Artur Czarnecki. All rights reserved.

"""Platform run artifact bundle read model for LKW HTTP responses (LKW-PF2)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.run_artifact_bundle import RUN_ARTIFACT_BUNDLE_METADATA_KEY
from intergrax.contracts.task_artifacts import RunArtifactBundle, WorkspaceArtifactRef
from intergrax.runtime.task.task import TaskResult
from intergrax.runtime.task.task_metadata_keys import TaskResultMetadataKey

_UNSAFE_BUNDLE_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "text",
        "body",
        "raw_chunks",
        "chunks",
        "document",
        "documents",
    }
)


def extract_run_artifact_bundle(task_result: TaskResult) -> RunArtifactBundle | None:
    """Resolve typed ``RunArtifactBundle`` from task completion metadata."""
    raw = task_result.metadata.get(TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE)
    if raw is None:
        app_summary = task_result.metadata.get(TaskResultMetadataKey.APPLICATION_RUN_SUMMARY)
        if isinstance(app_summary, dict):
            nested = app_summary.get("metadata")
            if isinstance(nested, dict):
                raw = nested.get(RUN_ARTIFACT_BUNDLE_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    return RunArtifactBundle.model_validate(raw)


def find_synthesize_workspace_artifact(
    bundle: RunArtifactBundle,
    *,
    artifact_path: str | None,
    artifact_ref: str | None,
) -> WorkspaceArtifactRef | None:
    """Match LKW synthesize diagnostic refs to platform workspace artifact entries."""
    normalized_path = artifact_path.strip() if artifact_path else None
    normalized_ref = artifact_ref.strip() if artifact_ref else None
    if not normalized_path and not normalized_ref:
        return None

    for ref in bundle.workspace:
        if normalized_path and ref.relative_path == normalized_path:
            return ref
        if normalized_ref:
            composite = f"{ref.workspace_id}/{ref.artifact_id}"
            if normalized_ref in {composite, ref.artifact_id}:
                return ref
    return None


def run_artifact_bundle_payload_is_safe(payload: dict[str, Any]) -> bool:
    """Return False when bundle payload exposes raw content fields."""
    for section in ("application", "workspace", "sandbox"):
        items = payload.get(section)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if _UNSAFE_BUNDLE_KEYS.intersection(item.keys()):
                return False
    return True


def ensure_run_artifact_bundle_metadata(
    metadata: dict[str, Any],
    *,
    task_result: TaskResult,
) -> dict[str, Any]:
    """Ensure platform ``run_artifact_bundle.v1`` is present on curated HTTP metadata."""
    if TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE in metadata:
        return metadata
    bundle = extract_run_artifact_bundle(task_result)
    if bundle is None:
        return metadata
    metadata[TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE] = bundle.model_dump(mode="json")
    return metadata
