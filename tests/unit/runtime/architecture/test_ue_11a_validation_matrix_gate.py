# © Artur Czarnecki. All rights reserved.

"""UE-11A — validation matrix completeness and proof integrity gate."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from testing_support.unified_execution_validation import (
    REQUIRED_CAPABILITY_IDS,
    REQUIRED_DOMAINS,
    UNIFIED_EXECUTION_VALIDATION_MATRIX,
    GapTarget,
    ValidationDomain,
    ValidationProofKind,
    ValidationStatus,
    count_by_domain,
    count_by_status,
    gap_backlog,
    matrix_capability_ids,
    repo_root,
    validate_unified_execution_matrix,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
EXPECTED_GAP_TARGET_BY_DOMAIN: dict[ValidationDomain, GapTarget] = {
    ValidationDomain.ROOT_STRATEGY: GapTarget.UE_11B,
    ValidationDomain.CHILD_EXECUTION: GapTarget.UE_11D,
    ValidationDomain.CONCURRENCY: GapTarget.UE_11D,
    ValidationDomain.RECOVERY: GapTarget.UE_11E,
    ValidationDomain.OBSERVABILITY: GapTarget.UE_11F,
    ValidationDomain.DIAGNOSTICS: GapTarget.UE_11F,
    ValidationDomain.PRODUCTION_SCENARIO: GapTarget.UE_11G,
}

_FORBIDDEN_QUALITY_PATTERNS = (
    re.compile(r"\btyping\.Any\b"),
    re.compile(r"\bfrom typing import\b.*\bAny\b"),
    re.compile(r"^\s*import\s+inspect\b"),
    re.compile(r"\bgetattr\("),
    re.compile(r"\bsetattr\("),
    re.compile(r"\bhasattr\("),
    re.compile(r"#\s*type:\s*ignore\b"),
)


def test_ue_11a_matrix_has_all_required_domains() -> None:
  present = {entry.domain for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX}
  assert present == set(REQUIRED_DOMAINS)


def test_ue_11a_matrix_has_all_required_capabilities() -> None:
  assert matrix_capability_ids() == REQUIRED_CAPABILITY_IDS


def test_ue_11a_matrix_capability_ids_are_unique() -> None:
  ids = [entry.capability_id for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX]
  assert len(ids) == len(set(ids))


def test_ue_11a_matrix_status_consistency() -> None:
  violations = [
    message
    for message in validate_unified_execution_matrix(repo_root_path=_REPO_ROOT)
    if "COVERED" in message
    or "PARTIAL" in message
    or "GAP" in message
    or "gap_target" in message
    or "requires proofs" in message
    or "must not list proofs" in message
  ]
  assert violations == []


def test_ue_11a_matrix_proof_paths_exist() -> None:
  violations = [
    message
    for message in validate_unified_execution_matrix(repo_root_path=_REPO_ROOT)
    if "missing proof file" in message
  ]
  assert violations == []


def test_ue_11a_matrix_proof_test_symbols_exist() -> None:
  violations = [
    message
    for message in validate_unified_execution_matrix(repo_root_path=_REPO_ROOT)
    if "missing test symbol" in message
  ]
  assert violations == []


def test_ue_11a_matrix_proof_surfaces_are_test_only() -> None:
  violations = [
    message
    for message in validate_unified_execution_matrix(repo_root_path=_REPO_ROOT)
    if "proof path not under tests/" in message
  ]
  assert violations == []


def test_ue_11a_matrix_gate_passes() -> None:
  violations = validate_unified_execution_matrix(repo_root_path=_REPO_ROOT)
  assert violations == [], "UE-11A matrix gate violations:\n" + "\n".join(violations)


def test_ue_11a_covered_capabilities_have_explicit_proofs() -> None:
  missing = [
    entry.capability_id
    for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX
    if entry.status is ValidationStatus.COVERED and not entry.proofs
  ]
  assert missing == []


def test_ue_11a_gap_capabilities_have_gap_targets() -> None:
  missing = [
    entry.capability_id
    for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX
    if entry.status in {ValidationStatus.PARTIAL, ValidationStatus.GAP}
    and entry.gap_target is None
  ]
  assert missing == []


def test_ue_11a_validation_model_has_typed_enums() -> None:
  assert {item.value for item in ValidationDomain} == {
    "ROOT_STRATEGY",
    "LIFECYCLE",
    "FAIL_CLOSED",
    "CHILD_EXECUTION",
    "CONCURRENCY",
    "RECOVERY",
    "OBSERVABILITY",
    "DIAGNOSTICS",
    "ANTI_BYPASS",
    "PRODUCTION_SCENARIO",
  }
  assert {item.value for item in ValidationStatus} == {"covered", "partial", "gap"}
  assert {item.value for item in ValidationProofKind} == {
    "unit",
    "integration",
    "acceptance",
    "architecture_gate",
  }


def test_ue_11a_matrix_has_expected_capability_count() -> None:
  assert len(UNIFIED_EXECUTION_VALIDATION_MATRIX) == len(REQUIRED_CAPABILITY_IDS)


def test_ue_11a_matrix_documents_real_gaps() -> None:
  counts = count_by_status()
  assert counts[ValidationStatus.GAP] >= 5
  assert counts[ValidationStatus.PARTIAL] >= 3
  backlog = gap_backlog()
  assert backlog[GapTarget.UE_11G]
  assert backlog[GapTarget.UE_11B]
  assert backlog[GapTarget.UE_11D]
  assert backlog[GapTarget.UE_11E]
  assert backlog[GapTarget.UE_11F]


def test_ue_11a_partial_and_gap_targets_match_canonical_roadmap() -> None:
  mismatches: list[str] = []
  for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX:
    if entry.status not in {ValidationStatus.PARTIAL, ValidationStatus.GAP}:
      continue
    expected = EXPECTED_GAP_TARGET_BY_DOMAIN.get(entry.domain)
    if expected is None:
      mismatches.append(
        f"{entry.capability_id}: no canonical gap target for domain {entry.domain.value}"
      )
      continue
    if entry.gap_target != expected:
      mismatches.append(
        f"{entry.capability_id}: expected {expected.value}, got {entry.gap_target}"
      )
  assert mismatches == []


def test_ue_11a_validation_module_has_no_forbidden_constructions() -> None:
  path = repo_root() / "testing_support" / "unified_execution_validation.py"
  violations: list[str] = []
  for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    for pattern in _FORBIDDEN_QUALITY_PATTERNS:
      if pattern.search(line):
        violations.append(f"{path.name}:{lineno}: {line.strip()}")
  assert violations == []


def test_ue_11a_domain_summary_is_complete() -> None:
  summary = count_by_domain()
  for domain in ValidationDomain:
    assert domain in summary
    assert sum(summary[domain].values()) > 0
