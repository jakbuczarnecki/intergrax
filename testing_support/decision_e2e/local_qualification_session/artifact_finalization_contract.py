# © Artur Czarnecki. All rights reserved.

"""Enterprise qualification artifact finalization contract (DS-E2E-15J-L1.R4.R1.A)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ArtifactCompletenessStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ArtifactRequirement:
    name: str
    producer: str
    required: bool


@dataclass(frozen=True, slots=True)
class ArtifactValidationResult:
    name: str
    exists: bool
    valid: bool
    checksum: str | None


@dataclass(frozen=True, slots=True)
class ArtifactCompletenessReport:
    status: ArtifactCompletenessStatus
    artifacts: tuple[ArtifactValidationResult, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QualificationArtifactFinalizationContract:
    """Required artifacts, validation order, and completeness for session finalize."""

    RAW_REQUIREMENTS: tuple[ArtifactRequirement, ...] = (
        ArtifactRequirement("runs.json", "run_collector", True),
        ArtifactRequirement("run.log", "run_collector", True),
        ArtifactRequirement("summary.json", "session_summary", True),
        ArtifactRequirement("report.md", "final_report", True),
        ArtifactRequirement("analysis.json", "QualificationAnalysisBuilder", True),
        ArtifactRequirement("artifact-manifest.txt", "ArtifactManifest", True),
    )
    DERIVED_REQUIREMENTS: tuple[ArtifactRequirement, ...] = (
        ArtifactRequirement("failure_cohort.csv", "DerivedArtifactGenerator", True),
        ArtifactRequirement(
            "revision_effectiveness.csv",
            "DerivedArtifactGenerator",
            True,
        ),
        ArtifactRequirement("alignment_direction.csv", "DerivedArtifactGenerator", True),
        ArtifactRequirement("local_model_profile.json", "DerivedArtifactGenerator", True),
    )
    POST_REQUIREMENTS: tuple[ArtifactRequirement, ...] = (
        ArtifactRequirement("final-report.md", "final_report", True),
    )

    @classmethod
    def all_requirements(cls) -> tuple[ArtifactRequirement, ...]:
        return cls.RAW_REQUIREMENTS + cls.DERIVED_REQUIREMENTS + cls.POST_REQUIREMENTS

    @classmethod
    def required_artifact_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in cls.all_requirements() if item.required)

    @classmethod
    def raw_artifact_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in cls.RAW_REQUIREMENTS if item.required)

    @classmethod
    def derived_artifact_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in cls.DERIVED_REQUIREMENTS if item.required)

    @classmethod
    def validate_artifact(cls, session_dir: Path, name: str) -> ArtifactValidationResult:
        path = session_dir / name
        if not path.is_file():
            return ArtifactValidationResult(
                name=name,
                exists=False,
                valid=False,
                checksum=None,
            )
        checksum = _sha256_file(path)
        return ArtifactValidationResult(
            name=name,
            exists=True,
            valid=True,
            checksum=checksum,
        )

    @classmethod
    def completeness_report(
        cls,
        session_dir: Path,
        *,
        names: tuple[str, ...] | None = None,
    ) -> ArtifactCompletenessReport:
        required = names or cls.required_artifact_names()
        results = tuple(cls.validate_artifact(session_dir, name) for name in required)
        if any(not item.exists for item in results if item.name in required):
            status = ArtifactCompletenessStatus.INCOMPLETE
        elif any(not item.valid for item in results):
            status = ArtifactCompletenessStatus.INVALID
        else:
            status = ArtifactCompletenessStatus.COMPLETE
        return ArtifactCompletenessReport(status=status, artifacts=results)

    @classmethod
    def validate_manifest_checksums(cls, session_dir: Path) -> None:
        manifest_path = session_dir / "artifact-manifest.txt"
        if not manifest_path.is_file():
            raise ValueError("artifact-manifest.txt missing")
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        entries: dict[str, str] = {}
        self_line: str | None = None
        for line in lines:
            if not line.strip():
                continue
            if line.startswith("artifact-manifest.txt sha256:"):
                self_line = line
                continue
            if " sha256:" not in line:
                raise ValueError(f"invalid manifest line: {line}")
            name, digest = line.split(" sha256:", 1)
            entries[name] = digest.strip()

        body_lines = [line for line in lines if not line.startswith("artifact-manifest.txt")]
        body = "\n".join(body_lines) + ("\n" if body_lines else "")
        expected_self = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if self_line is None:
            raise ValueError("manifest missing self checksum line")
        actual_self = self_line.split(" sha256:", 1)[1].strip()
        if actual_self != expected_self:
            raise ValueError("artifact-manifest.txt self checksum mismatch")

        for name, expected in entries.items():
            path = session_dir / name
            if not path.is_file():
                raise ValueError(f"manifest artifact missing: {name}")
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(f"manifest checksum mismatch: {name}")
