# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project artifact references from runtime spine events into metadata records."""

from __future__ import annotations

from intergrax.contracts.execution_artifact_read import (
    ExecutionArtifactLifecycleStatus,
    ExecutionArtifactMetadataReadResult,
    ExecutionArtifactMetadataRecord,
)
from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def project_artifact_metadata_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    *,
    limit: int,
) -> ExecutionArtifactMetadataReadResult:
    if limit < 1:
        raise ValueError("limit must be positive")

    records: list[ExecutionArtifactMetadataRecord] = []
    seen_refs: set[str] = set()
    for positioned in reconstruction.positioned_events:
        event = positioned.event
        candidates = _artifact_payload_candidates(event.payload)
        for artifact_ref, artifact_type, classification in candidates:
            if artifact_ref in seen_refs:
                continue
            seen_refs.add(artifact_ref)
            safe_summary = sanitize_inspection_text(f"{artifact_type}:{artifact_ref}")
            records.append(
                ExecutionArtifactMetadataRecord(
                    artifact_ref=artifact_ref,
                    artifact_type=artifact_type,
                    lifecycle_status=ExecutionArtifactLifecycleStatus.REGISTERED,
                    content_classification=classification,
                    tenant_id=event.tenant_id or reconstruction.tenant_id,
                    task_id=event.task_id,
                    run_id=event.run_id,
                    execution_id=event.execution_id,
                    attempt_id=event.attempt_id,
                    sequence_key=positioned.position.value,
                    evidence_refs=(str(event.event_id),),
                    safe_summary=safe_summary,
                ),
            )
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break

    sorted_records = tuple(
        sorted(records, key=lambda item: (item.sequence_key, item.artifact_ref)),
    )
    return ExecutionArtifactMetadataReadResult(
        records=sorted_records,
        is_truncated=len(seen_refs) > limit,
    )


def _artifact_payload_candidates(
    payload: dict[str, object],
) -> tuple[tuple[str, str, str], ...]:
    found: list[tuple[str, str, str]] = []
    refs = payload.get("artifact_refs")
    if isinstance(refs, list):
        for item in refs:
            if not isinstance(item, dict):
                continue
            parsed = _parse_artifact_dict(item)
            if parsed is not None:
                found.append(parsed)
    single = _parse_artifact_dict(payload)
    if single is not None:
        found.append(single)
    return tuple(found)


def _parse_artifact_dict(payload: dict[str, object]) -> tuple[str, str, str] | None:
    artifact_id = payload.get("artifact_id") or payload.get("id")
    if artifact_id is None:
        return None
    artifact_ref = str(artifact_id).strip()
    if not artifact_ref:
        return None
    artifact_type = str(
        payload.get("type") or payload.get("kind") or "structured",
    ).strip() or "structured"
    sensitivity = payload.get("sensitivity") or payload.get("security_class")
    classification = str(sensitivity).strip() if sensitivity else "internal"
    return artifact_ref, artifact_type, classification


__all__ = ["project_artifact_metadata_from_reconstruction"]
