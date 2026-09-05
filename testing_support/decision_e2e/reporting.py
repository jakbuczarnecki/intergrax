# © Artur Czarnecki. All rights reserved.

"""Machine-readable DS-E2E qualification reporting."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from testing_support.decision_e2e.contracts import (
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.requirements import validate_qualification_result


@dataclass(frozen=True, slots=True)
class QualificationReport:
    git_sha: str
    timestamp: str
    environment_profile: str
    results: tuple[DecisionE2EQualificationResult, ...]

    def enterprise_closed(self) -> bool:
        if not self.results:
            return False
        return all(
            row.disposition is QualificationDisposition.PASSED for row in self.results
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "environment_profile": self.environment_profile,
            "enterprise_closed": self.enterprise_closed(),
            "proofs": [row.to_report_row() for row in self.results],
        }


def resolve_git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def write_qualification_artifacts(
    report: QualificationReport,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    json_path = output_dir / "qualification.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = output_dir / "qualification.md"
    lines = [
        "# Decision System DS-E2E Qualification",
        "",
        f"- git_sha: `{report.git_sha}`",
        f"- timestamp: `{report.timestamp}`",
        f"- environment_profile: `{report.environment_profile}`",
        f"- enterprise_closed: `{report.enterprise_closed()}`",
        "",
        "| proof_id | disposition | reason |",
        "| --- | --- | --- |",
    ]
    for row in report.results:
        reason = (row.reason or "").replace("|", "/")
        lines.append(f"| {row.proof_id.value} | {row.disposition.value} | {reason} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


class QualificationReportCollector:
    """Session-scoped qualification result collector."""

    def __init__(self) -> None:
        self._results: list[DecisionE2EQualificationResult] = []

    def record(self, result: DecisionE2EQualificationResult) -> None:
        self._results.append(validate_qualification_result(result))

    def build_report(self, *, environment_profile: str) -> QualificationReport:
        seen: set[str] = set()
        duplicates: list[str] = []
        for row in self._results:
            proof_id = row.proof_id.value
            if proof_id in seen:
                duplicates.append(proof_id)
            seen.add(proof_id)
        if duplicates:
            joined = ", ".join(sorted(set(duplicates)))
            raise ValueError(
                "Authoritative qualification report requires exactly one result per proof; "
                f"duplicate proof_id entries: {joined}",
            )
        return QualificationReport(
            git_sha=resolve_git_sha(),
            timestamp=datetime.now(UTC).isoformat(),
            environment_profile=environment_profile,
            results=tuple(self._results),
        )
