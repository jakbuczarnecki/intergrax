# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1-R1-Q1 resume baseline evidence validation (committed artifact authority)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Literal

from tests.qualification.trace_x._trace_x_p5_discovery import repo_root

RESUME_EVIDENCE_SCHEMA_VERSION: Final[str] = "trace_x_p5_r1_r1_q1_resume_baseline_v1"
RESUME_EVIDENCE_QUALIFICATION_ID: Final[str] = "TRACE-X-P5-R1-R1-Q1"
RESUME_BASELINE_SHA: Final[str] = "98c0d9d7ae9763bce931c60e19b6a91af3a2f4e9"
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

RESUME_EVIDENCE_REL_PATH: Final[str] = (
    "docs/project/maintainers/qualification/TRACE_X_P5_R1_R1_Q1_RESUME_BASELINE_EVIDENCE.json"
)

_SECRET_LIKE_KEY_RE = re.compile(r"(password|secret|token|api_key)", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]|^/home/|^/Users/")


def resume_evidence_path() -> Path:
    return repo_root() / RESUME_EVIDENCE_REL_PATH


def load_resume_baseline_evidence() -> dict[str, Any]:
    return json.loads(resume_evidence_path().read_text(encoding="utf-8"))


def validate_resume_baseline_payload(
    payload: dict[str, Any],
    *,
    expected_q1_start_head: str,
) -> list[str]:
    violations: list[str] = []

    if payload.get("schema_version") != RESUME_EVIDENCE_SCHEMA_VERSION:
        violations.append("schema_version mismatch")
    if payload.get("qualification_id") != RESUME_EVIDENCE_QUALIFICATION_ID:
        violations.append("qualification_id mismatch")
    if payload.get("generated_from_q1_start_head") != expected_q1_start_head:
        violations.append("generated_from_q1_start_head mismatch")

    baseline = payload.get("baseline")
    current = payload.get("current")
    comparison = payload.get("comparison")
    if not isinstance(baseline, dict) or not isinstance(current, dict) or not isinstance(comparison, dict):
        violations.append("missing baseline/current/comparison objects")
        return violations

    if baseline.get("sha") != RESUME_BASELINE_SHA:
        violations.append("baseline.sha mismatch")

    for section_name, section in (("baseline", baseline), ("current", current)):
        if section.get("command") != RESUME_TEST_COMMAND:
            violations.append(f"{section_name}.command mismatch")
        failed_ids = section.get("failed_test_node_ids")
        if not isinstance(failed_ids, list):
            violations.append(f"{section_name}.failed_test_node_ids not a list")
        else:
            for required in RESUME_TEST_NODE_IDS:
                if required not in failed_ids:
                    violations.append(f"{section_name} missing failed node id: {required}")

    conclusion = comparison.get("conclusion")
    valid_conclusions = {"PRE_EXISTING_NON_R1_REGRESSION", "R1_REGRESSION_POSSIBLE"}
    if conclusion not in valid_conclusions:
        violations.append("comparison.conclusion invalid or missing")

    if comparison.get("same_failed_tests") is not True:
        violations.append("comparison.same_failed_tests must be true for committed evidence")
    if comparison.get("same_exception_types") is not True:
        violations.append("comparison.same_exception_types must be true for committed evidence")
    if comparison.get("same_failure_signatures") is not True:
        violations.append("comparison.same_failure_signatures must be true for committed evidence")
    if comparison.get("regression_detected") is not False:
        violations.append("comparison.regression_detected must be false for PRE_EXISTING conclusion")

    if conclusion == "PRE_EXISTING_NON_R1_REGRESSION":
        if comparison.get("regression_detected") is True:
            violations.append("PRE_EXISTING conclusion inconsistent with regression_detected")
        baseline_sigs = baseline.get("stable_failure_signatures")
        current_sigs = current.get("stable_failure_signatures")
        if isinstance(baseline_sigs, list) and isinstance(current_sigs, list):
            if baseline_sigs != current_sigs and comparison.get("same_failure_signatures") is True:
                violations.append("comparison.same_failure_signatures inconsistent with recorded signatures")

    _scan_forbidden_paths_and_secrets(payload, violations)
    return violations


def validate_resume_baseline_evidence(
    *,
    expected_q1_start_head: str,
) -> list[str]:
    try:
        payload = load_resume_baseline_evidence()
    except (OSError, json.JSONDecodeError) as exc:
        return [f"evidence load failed: {exc}"]
    return validate_resume_baseline_payload(payload, expected_q1_start_head=expected_q1_start_head)


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


__all__ = [
    "RESUME_BASELINE_SHA",
    "RESUME_EVIDENCE_QUALIFICATION_ID",
    "RESUME_EVIDENCE_SCHEMA_VERSION",
    "RESUME_TEST_COMMAND",
    "RESUME_TEST_NODE_IDS",
    "load_resume_baseline_evidence",
    "resume_evidence_path",
    "stable_resume_failure_signature",
    "validate_resume_baseline_evidence",
    "validate_resume_baseline_payload",
]
