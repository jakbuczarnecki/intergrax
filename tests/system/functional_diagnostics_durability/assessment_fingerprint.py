# © Artur Czarnecki. All rights reserved.

"""Typed deterministic diagnostic assessment fingerprint for D1-R1 recovery proof."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
)


@dataclass(frozen=True, slots=True)
class DiagnosticAssessmentFingerprint:
  """Semantic fingerprint excluding PID, timestamps, and memory identity."""

  specification_id: str
  specification_version: int
  check_ids: tuple[str, ...]
  check_statuses: tuple[str, ...]
  first_proven_failure: str | None
  evidence_ref_counts: tuple[int, ...]

  def to_json_dict(self) -> dict[str, str | int | tuple[str, ...] | tuple[int, ...] | None]:
    return {
      "specification_id": self.specification_id,
      "specification_version": self.specification_version,
      "check_ids": self.check_ids,
      "check_statuses": self.check_statuses,
      "first_proven_failure": self.first_proven_failure,
      "evidence_ref_counts": self.evidence_ref_counts,
    }

  @classmethod
  def from_analysis(cls, analysis: FunctionalDiagnosticAnalysis) -> DiagnosticAssessmentFingerprint:
    return cls(
      specification_id=str(analysis.specification_id),
      specification_version=analysis.specification_version,
      check_ids=tuple(str(item.check_id) for item in analysis.check_results),
      check_statuses=tuple(item.status.value for item in analysis.check_results),
      first_proven_failure=(
        str(analysis.first_proven_failure)
        if analysis.first_proven_failure is not None
        else None
      ),
      evidence_ref_counts=tuple(len(item.supporting_evidence_refs) for item in analysis.check_results),
    )

  @classmethod
  def from_json_mapping(cls, payload: object) -> DiagnosticAssessmentFingerprint:
    if not isinstance(payload, dict):
      raise ValueError("assessment_fingerprint_invalid")
    specification_id = payload.get("specification_id")
    specification_version = payload.get("specification_version")
    check_ids = payload.get("check_ids")
    check_statuses = payload.get("check_statuses")
    first_proven_failure = payload.get("first_proven_failure")
    evidence_ref_counts = payload.get("evidence_ref_counts")
    if not isinstance(specification_id, str):
      raise ValueError("assessment_fingerprint_specification_id_invalid")
    if not isinstance(specification_version, int) or isinstance(specification_version, bool):
      raise ValueError("assessment_fingerprint_specification_version_invalid")
    if not isinstance(check_ids, list) or not all(isinstance(item, str) for item in check_ids):
      raise ValueError("assessment_fingerprint_check_ids_invalid")
    if not isinstance(check_statuses, list) or not all(isinstance(item, str) for item in check_statuses):
      raise ValueError("assessment_fingerprint_check_statuses_invalid")
    if first_proven_failure is not None and not isinstance(first_proven_failure, str):
      raise ValueError("assessment_fingerprint_first_proven_failure_invalid")
    if not isinstance(evidence_ref_counts, list) or not all(
      isinstance(item, int) and not isinstance(item, bool) for item in evidence_ref_counts
    ):
      raise ValueError("assessment_fingerprint_evidence_ref_counts_invalid")
    return cls(
      specification_id=specification_id,
      specification_version=specification_version,
      check_ids=tuple(check_ids),
      check_statuses=tuple(check_statuses),
      first_proven_failure=first_proven_failure,
      evidence_ref_counts=tuple(evidence_ref_counts),
    )


__all__ = ["DiagnosticAssessmentFingerprint"]
