# © Artur Czarnecki. All rights reserved.

"""Artifact serialization for DIAG-FUNCTIONAL-SCALE-S1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from intergrax.knowledge.contracts.validation import JsonObject
from tests.system.functional_diagnostics_scale.backend import ScaleGateResult
from tests.system.functional_diagnostics_scale.manifest import ScaleDatasetManifest
from tests.system.functional_diagnostics_scale.metrics import (
    ExecutionReadScaleCurve,
    LatencyDistribution,
    ScaleCorrectnessMetrics,
    ScaleResourceMetrics,
    ThroughputMeasurement,
    latency_samples_to_json,
    throughput_to_json,
)
from tests.system.functional_diagnostics_scale.profile import FunctionalDiagnosticsScaleProfile


@dataclass(frozen=True, slots=True)
class ScaleQualificationReport:
    verdict: str
    blocker: str
    start_head: str
    final_head: str
    profile: FunctionalDiagnosticsScaleProfile
    manifest: ScaleDatasetManifest
    gates: tuple[ScaleGateResult, ...]
    correctness: ScaleCorrectnessMetrics
    latency: dict[str, LatencyDistribution]
    throughput: dict[str, ThroughputMeasurement]
    resources: ScaleResourceMetrics
    scale_curve: ExecutionReadScaleCurve | None
    backend_provider: str
    backend_document_store_type: str
    database_name: str
    collection_name: str
    production_provider_factory_used: bool
    backend_mocked: bool
    backend_in_memory: bool
    first_canonical_run: bool

    def to_json_dict(self) -> JsonObject:
        return {
            "verdict": self.verdict,
            "blocker": self.blocker,
            "start_head": self.start_head,
            "final_head": self.final_head,
            "profile": self.profile.to_json_dict(),
            "manifest_digest": self.manifest.manifest_digest(),
            "gates": [gate.to_json_dict() for gate in self.gates],
            "correctness": self.correctness.to_json_dict(),
            "latency": latency_samples_to_json(self.latency),
            "throughput": throughput_to_json(self.throughput),
            "resources": self.resources.to_json_dict(),
            "scale_curve": (
                self.scale_curve.to_json_dict() if self.scale_curve is not None else None
            ),
            "backend_provider": self.backend_provider,
            "backend_document_store_type": self.backend_document_store_type,
            "database_name": self.database_name,
            "collection_name": self.collection_name,
            "production_provider_factory_used": self.production_provider_factory_used,
            "backend_mocked": self.backend_mocked,
            "backend_in_memory": self.backend_in_memory,
            "first_canonical_run": self.first_canonical_run,
        }


def write_json_artifact(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_qualification_artifacts(
    *,
    artifact_dir: Path,
    profile: FunctionalDiagnosticsScaleProfile,
    report: ScaleQualificationReport,
) -> None:
    write_json_artifact(artifact_dir / "scale-profile.json", profile.to_json_dict())
    write_json_artifact(artifact_dir / "qualification-report.json", report.to_json_dict())
    write_json_artifact(
        artifact_dir / "latency-metrics.json",
        latency_samples_to_json(report.latency),
    )
    write_json_artifact(
        artifact_dir / "resource-metrics.json",
        report.resources.to_json_dict(),
    )
    write_json_artifact(
        artifact_dir / "scale-manifest.json",
        report.manifest.to_json_dict(),
    )


__all__ = [
    "ScaleQualificationReport",
    "write_json_artifact",
    "write_qualification_artifacts",
]
