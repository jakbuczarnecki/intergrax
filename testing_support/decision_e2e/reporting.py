# © Artur Czarnecki. All rights reserved.

"""Machine-readable DS-E2E qualification reporting."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    EXPECTED_DECISION_E2E_PROOFS,
    QualificationCompleteness,
    QualificationDisposition,
)
from testing_support.decision_e2e.requirements import validate_qualification_result

__all__ = (
    "QualificationReport",
    "QualificationReportCollector",
    "assess_qualification_completeness",
    "resolve_git_sha",
    "validate_qualification_result",
    "write_qualification_artifacts",
)


def assess_qualification_completeness(
    results: tuple[DecisionE2EQualificationResult, ...],
) -> QualificationCompleteness:
    actual = frozenset(row.proof_id for row in results)
    return QualificationCompleteness(
        expected=EXPECTED_DECISION_E2E_PROOFS,
        actual=actual,
    )


@dataclass(frozen=True, slots=True)
class QualificationReport:
    git_sha: str
    timestamp: str
    environment_profile: str
    results: tuple[DecisionE2EQualificationResult, ...]

    def completeness(self) -> QualificationCompleteness:
        return assess_qualification_completeness(self.results)

    def enterprise_closed(self) -> bool:
        completeness = self.completeness()
        if not completeness.complete:
            return False
        if len(self.results) != len(EXPECTED_DECISION_E2E_PROOFS):
            return False
        return all(
            row.disposition is QualificationDisposition.PASSED for row in self.results
        )

    def to_dict(self) -> dict[str, object]:
        completeness = self.completeness()
        passed = sum(
            1 for row in self.results if row.disposition is QualificationDisposition.PASSED
        )
        failed = sum(
            1 for row in self.results if row.disposition is QualificationDisposition.FAILED
        )
        blocked = sum(
            1 for row in self.results if row.disposition is QualificationDisposition.BLOCKED
        )
        return {
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "environment_profile": self.environment_profile,
            "enterprise_closed": self.enterprise_closed(),
            "proof_count": len(self.results),
            "unique_proof_count": len(completeness.actual),
            "expected_proof_set": sorted(item.value for item in completeness.expected),
            "actual_proof_set": sorted(item.value for item in completeness.actual),
            "missing_proof_set": sorted(item.value for item in completeness.missing),
            "unexpected_proof_set": sorted(item.value for item in completeness.unexpected),
            "passed_count": passed,
            "failed_count": failed,
            "blocked_count": blocked,
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
        actual_ids = frozenset(row.proof_id for row in self._results)
        unexpected = actual_ids - EXPECTED_DECISION_E2E_PROOFS
        if unexpected:
            joined = ", ".join(sorted(item.value for item in unexpected))
            raise ValueError(
                "Authoritative qualification report contains unexpected proof IDs: "
                f"{joined}",
            )
        return QualificationReport(
            git_sha=resolve_git_sha(),
            timestamp=datetime.now(UTC).isoformat(),
            environment_profile=environment_profile,
            results=tuple(self._results),
        )
