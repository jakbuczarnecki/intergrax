# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1-R1-Q2 resume baseline evidence validation (derived comparison authority)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from tests.qualification.trace_x._trace_x_p5_discovery import repo_root

RESUME_EVIDENCE_SCHEMA_VERSION: Final[str] = "trace_x_p5_r1_r1_q2_resume_baseline_v2"
RESUME_EVIDENCE_QUALIFICATION_ID: Final[str] = "TRACE-X-P5-R1-R1-Q1"
RESUME_BASELINE_SHA: Final[str] = "98c0d9d7ae9763bce931c60e19b6a91af3a2f4e9"
RESUME_R1_R1_IMPLEMENTATION_SHA: Final[str] = "a452de39a721cd357be3ba5ecd0c3a6d41b630bd"
RESUME_TEST_NODE_IDS: Final[tuple[str, ...]] = (
    "tests/unit/applications/test_effective_profile_revision_adoption.py::test_missing_binding_on_resume_fails_closed",
    "tests/unit/applications/test_effective_profile_revision_adoption.py::test_resume_preserves_pinned_revision_not_current_host_revision",
)
RESUME_TEST_COMMAND: Final[str] = (
    "uv run pytest "
    "tests/unit/applications/test_effective_profile_revision_adoption.py::test_missing_binding_on_resume_fails_closed "
    "tests/unit/applications/test_effective_profile_revision_adoption.py::test_resume_preserves_pinned_revision_not_current_host_revision "
    "-p no:xdist --tb=short -q"
)
RESUME_EXPECTED_EXCEPTION: Final[str] = "CheckpointResumeValidationError"

RESUME_EVIDENCE_REL_PATH: Final[str] = (
    "docs/project/maintainers/qualification/TRACE_X_P5_R1_R1_Q1_RESUME_BASELINE_EVIDENCE.json"
)

_SECRET_LIKE_KEY_RE = re.compile(r"(password|secret|token|api_key)", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]|^/home/|^/Users/")


@dataclass(frozen=True, slots=True)
class ResumeFailureRecord:
    node_id: str
    exception_type: str
    semantic_message: str
    stable_signature: str


@dataclass(frozen=True, slots=True)
class DerivedResumeComparison:
    same_failed_tests: bool
    same_exception_types: bool
    same_failure_signatures: bool
    regression_detected: bool
    conclusion: Literal["PRE_EXISTING_NON_R1_REGRESSION", "R1_REGRESSION_POSSIBLE"]


def resume_evidence_path() -> Path:
    return repo_root() / RESUME_EVIDENCE_REL_PATH


def load_resume_baseline_evidence() -> dict[str, Any]:
    return json.loads(resume_evidence_path().read_text(encoding="utf-8"))


def stable_resume_failure_signature(
    test_node_id: str,
    exception_type: str,
    message: str,
) -> str:
    normalized = message.strip()
    normalized = re.sub(r"task_[0-9a-f]{32}", "task_<REDACTED>", normalized)
    normalized = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "<PATH>", normalized)
    normalized = re.sub(r"/(?:tmp|var|Users|home)[^\s'\"]+", "<PATH>", normalized)
    return f"{test_node_id}|{exception_type}|{normalized}"


def _failure_records_from_section(section: dict[str, Any]) -> list[ResumeFailureRecord] | None:
    raw_failures = section.get("failures")
    if isinstance(raw_failures, list) and raw_failures:
        records: list[ResumeFailureRecord] = []
        for item in raw_failures:
            if not isinstance(item, dict):
                return None
            node_id = item.get("node_id")
            exception_type = item.get("exception_type")
            semantic_message = item.get("semantic_message")
            stable_signature = item.get("stable_signature")
            if not all(isinstance(v, str) for v in (node_id, exception_type, semantic_message, stable_signature)):
                return None
            records.append(
                ResumeFailureRecord(
                    node_id=node_id,
                    exception_type=exception_type,
                    semantic_message=semantic_message,
                    stable_signature=stable_signature,
                )
            )
        return records
    return None


def _failure_map(section: dict[str, Any]) -> dict[str, ResumeFailureRecord] | None:
    structured = _failure_records_from_section(section)
    if structured is not None:
        return {row.node_id: row for row in structured}

    failed_ids = section.get("failed_test_node_ids")
    exception_types = section.get("exception_types")
    signatures = section.get("stable_failure_signatures")
    if not isinstance(failed_ids, list) or not isinstance(exception_types, list) or not isinstance(
        signatures, list
    ):
        return None
    if len(failed_ids) != len(exception_types) or len(failed_ids) != len(signatures):
        return None
    mapping: dict[str, ResumeFailureRecord] = {}
    for node_id, exc, sig in zip(failed_ids, exception_types, signatures, strict=True):
        if not isinstance(node_id, str) or not isinstance(exc, str) or not isinstance(sig, str):
            return None
        parts = sig.split("|", 2)
        semantic = parts[2] if len(parts) == 3 else ""
        mapping[node_id] = ResumeFailureRecord(
            node_id=node_id,
            exception_type=exc,
            semantic_message=semantic,
            stable_signature=sig,
        )
    return mapping


def _validate_section_run_outcome(section_name: str, section: dict[str, Any], violations: list[str]) -> None:
    failed_ids = section.get("failed_test_node_ids")
    if not isinstance(failed_ids, list):
        violations.append(f"{section_name}.failed_test_node_ids not a list")
        return
    expected_set = set(RESUME_TEST_NODE_IDS)
    actual_set = set(failed_ids)
    if actual_set != expected_set:
        if actual_set - expected_set:
            violations.append(f"{section_name} has extra failed test node ids")
        if expected_set - actual_set:
            violations.append(f"{section_name} missing failed test node ids")
    if len(failed_ids) != len(expected_set):
        violations.append(f"{section_name}.failed_test_node_ids length must equal RESUME_TEST_NODE_IDS")

    failed_count = section.get("failed")
    passed_count = section.get("passed")
    exit_code = section.get("exit_code")
    if failed_count != len(RESUME_TEST_NODE_IDS):
        violations.append(f"{section_name}.failed count mismatch")
    if passed_count != 0:
        violations.append(f"{section_name}.passed must be 0 for this evidence")
    if not isinstance(exit_code, int) or exit_code == 0:
        violations.append(f"{section_name}.exit_code must be non-zero")

    failure_map = _failure_map(section)
    if failure_map is None:
        violations.append(f"{section_name} failures could not be parsed")
        return
    if len(failure_map) != len(RESUME_TEST_NODE_IDS):
        violations.append(f"{section_name} failure record count mismatch")

    for node_id in RESUME_TEST_NODE_IDS:
        record = failure_map.get(node_id)
        if record is None:
            violations.append(f"{section_name} missing failure record for {node_id}")
            continue
        if record.exception_type != RESUME_EXPECTED_EXCEPTION:
            violations.append(f"{section_name} unexpected exception for {node_id}")
        expected_sig = stable_resume_failure_signature(
            record.node_id,
            record.exception_type,
            record.semantic_message,
        )
        if record.stable_signature != expected_sig:
            violations.append(f"{section_name} stable_signature not derived for {node_id}")


def derive_resume_comparison(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> DerivedResumeComparison:
    expected = set(RESUME_TEST_NODE_IDS)
    baseline_ids = set(baseline.get("failed_test_node_ids") or [])
    current_ids = set(current.get("failed_test_node_ids") or [])
    same_failed_tests = baseline_ids == current_ids == expected

    baseline_map = _failure_map(baseline) or {}
    current_map = _failure_map(current) or {}

    same_exception_types = same_failed_tests and all(
        baseline_map.get(node_id) is not None
        and current_map.get(node_id) is not None
        and baseline_map[node_id].exception_type == current_map[node_id].exception_type
        for node_id in RESUME_TEST_NODE_IDS
    )

    same_failure_signatures = same_failed_tests and all(
        baseline_map.get(node_id) is not None
        and current_map.get(node_id) is not None
        and baseline_map[node_id].stable_signature == current_map[node_id].stable_signature
        for node_id in RESUME_TEST_NODE_IDS
    )

    if same_failed_tests and same_exception_types and same_failure_signatures:
        return DerivedResumeComparison(
            same_failed_tests=True,
            same_exception_types=True,
            same_failure_signatures=True,
            regression_detected=False,
            conclusion="PRE_EXISTING_NON_R1_REGRESSION",
        )
    return DerivedResumeComparison(
        same_failed_tests=same_failed_tests,
        same_exception_types=same_exception_types,
        same_failure_signatures=same_failure_signatures,
        regression_detected=True,
        conclusion="R1_REGRESSION_POSSIBLE",
    )


def _comparison_projection_matches_derived(
    comparison: dict[str, Any],
    derived: DerivedResumeComparison,
    violations: list[str],
) -> None:
    if comparison.get("same_failed_tests") is not derived.same_failed_tests:
        violations.append("comparison.same_failed_tests inconsistent with derived evidence")
    if comparison.get("same_exception_types") is not derived.same_exception_types:
        violations.append("comparison.same_exception_types inconsistent with derived evidence")
    if comparison.get("same_failure_signatures") is not derived.same_failure_signatures:
        violations.append("comparison.same_failure_signatures inconsistent with derived evidence")
    if comparison.get("regression_detected") is not derived.regression_detected:
        violations.append("comparison.regression_detected inconsistent with derived evidence")
    conclusion = comparison.get("conclusion")
    if conclusion != derived.conclusion:
        violations.append("comparison.conclusion inconsistent with derived evidence")


def validate_resume_baseline_payload(
    payload: dict[str, Any],
    *,
    expected_q1_start_head: str,
    expected_q2_start_head: str | None = None,
) -> list[str]:
    violations: list[str] = []

    if payload.get("schema_version") != RESUME_EVIDENCE_SCHEMA_VERSION:
        violations.append("schema_version mismatch")
    if payload.get("qualification_id") != RESUME_EVIDENCE_QUALIFICATION_ID:
        violations.append("qualification_id mismatch")
    if payload.get("generated_from_q1_start_head") != expected_q1_start_head:
        violations.append("generated_from_q1_start_head mismatch")
    if expected_q2_start_head is not None:
        if payload.get("generated_from_q2_start_head") != expected_q2_start_head:
            violations.append("generated_from_q2_start_head mismatch")

    baseline = payload.get("baseline")
    current = payload.get("current")
    comparison = payload.get("comparison")
    if not isinstance(baseline, dict) or not isinstance(current, dict) or not isinstance(comparison, dict):
        violations.append("missing baseline/current/comparison objects")
        return violations

    if baseline.get("sha") != RESUME_BASELINE_SHA:
        violations.append("baseline.sha mismatch")

    current_head = current.get("sha_or_worktree_head")
    if current_head != RESUME_R1_R1_IMPLEMENTATION_SHA:
        violations.append("current.sha_or_worktree_head must remain R1-R1 implementation comparison SHA")

    for section_name, section in (("baseline", baseline), ("current", current)):
        if section.get("command") != RESUME_TEST_COMMAND:
            violations.append(f"{section_name}.command mismatch")
        _validate_section_run_outcome(section_name, section, violations)

    derived = derive_resume_comparison(baseline, current)
    _comparison_projection_matches_derived(comparison, derived, violations)

    _scan_forbidden_paths_and_secrets(payload, violations)
    return violations


def validate_resume_baseline_evidence(
    *,
    expected_q1_start_head: str,
    expected_q2_start_head: str | None = None,
) -> list[str]:
    try:
        payload = load_resume_baseline_evidence()
    except (OSError, json.JSONDecodeError) as exc:
        return [f"evidence load failed: {exc}"]
    return validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=expected_q1_start_head,
        expected_q2_start_head=expected_q2_start_head,
    )


def _scan_forbidden_paths_and_secrets(value: Any, violations: list[str], key_path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{key_path}.{key}" if key_path else key
            if _SECRET_LIKE_KEY_RE.search(key):
                violations.append(f"secret-like field key: {child_path}")
            _scan_forbidden_paths_and_secrets(child, violations, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_paths_and_secrets(child, violations, f"{key_path}[{index}]")
    elif isinstance(value, str):
        if _ABSOLUTE_PATH_RE.search(value):
            violations.append(f"absolute path in evidence at {key_path}")
        if value.strip().lower() == "unknown":
            violations.append(f"unknown semantic value at {key_path}")


__all__ = [
    "DerivedResumeComparison",
    "RESUME_BASELINE_SHA",
    "RESUME_EVIDENCE_QUALIFICATION_ID",
    "RESUME_EVIDENCE_SCHEMA_VERSION",
    "RESUME_R1_R1_IMPLEMENTATION_SHA",
    "RESUME_TEST_COMMAND",
    "RESUME_TEST_NODE_IDS",
    "ResumeFailureRecord",
    "derive_resume_comparison",
    "load_resume_baseline_evidence",
    "resume_evidence_path",
    "stable_resume_failure_signature",
    "validate_resume_baseline_evidence",
    "validate_resume_baseline_payload",
]
