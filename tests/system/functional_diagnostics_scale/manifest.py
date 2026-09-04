# © Artur Czarnecki. All rights reserved.

"""Independent expected manifest for scale qualification (oracle metadata only)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.knowledge.contracts.validation import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class ScaleExecutionManifestEntry:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    evidence_ids: tuple[str, ...]
    evidence_fingerprints: tuple[str, ...]
    is_heavy: bool
    analyzer_sample: bool

    def to_json_dict(self) -> JsonObject:
        return {
            "tenant_id": self.tenant_id,
            "task_id": str(self.task_id),
            "run_id": str(self.run_id),
            "attempt_id": str(self.attempt_id),
            "evidence_ids": self.evidence_ids,
            "evidence_fingerprints": self.evidence_fingerprints,
            "is_heavy": self.is_heavy,
            "analyzer_sample": self.analyzer_sample,
        }


@dataclass(frozen=True, slots=True)
class ScaleDatasetManifest:
    seed: int
    profile_name: str
    entries: tuple[ScaleExecutionManifestEntry, ...]
    total_evidence: int
    tenant_ids: tuple[str, ...]

    def to_json_dict(self) -> JsonObject:
        return {
            "seed": self.seed,
            "profile_name": self.profile_name,
            "entries": [entry.to_json_dict() for entry in self.entries],
            "total_evidence": self.total_evidence,
            "tenant_ids": self.tenant_ids,
        }

    def manifest_digest(self) -> str:
        canonical = json.dumps(
            self.to_json_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_to_json_value(manifest: ScaleDatasetManifest) -> JsonValue:
    return manifest.to_json_dict()


__all__ = [
    "ScaleDatasetManifest",
    "ScaleExecutionManifestEntry",
    "manifest_to_json_value",
]
