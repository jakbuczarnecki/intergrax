# © Artur Czarnecki. All rights reserved.

"""Read-only review of token regression diagnostic artifact folders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REVIEW_SCHEMA_VERSION = 1
_REVIEW_ARTIFACT_KIND = "token_regression_artifact_review"

_REQUIRED_CASE_FIELDS = (
    "artifact_kind",
    "fixture_id",
    "source_type",
    "passed",
    "metadata",
    "input",
    "output",
    "metrics",
    "validation",
    "expectation",
)

_ENCODING_SUSPICIOUS_SEQUENCES = ("â€¦", "Ã", "Â")
_TRUNCATION_SUFFIXES = ("…", "...")
_DOMINANT_SAVINGS_THRESHOLD = 0.50
_LARGE_SAVING_RATIO_THRESHOLD = 0.50
_TRUNCATION_LENGTH_RATIO = 0.40

_CHANGE_TYPE_WHITESPACE = "whitespace_or_structural_compaction"
_CHANGE_TYPE_PROTECTED = "protected_content_preserved"
_CHANGE_TYPE_FALLBACK = "expected_fallback"
_CHANGE_TYPE_TRUNCATION = "likely_truncation"
_CHANGE_TYPE_NO_SAVINGS = "no_savings"
_CHANGE_TYPE_UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TokenRegressionArtifactReviewIssue:
    """Single review issue with stable code and severity."""

    severity: str
    code: str
    message: str
    fixture_id: str | None = None


@dataclass(frozen=True, slots=True)
class TokenRegressionArtifactCaseReview:
    """Per-case interpretation derived from a diagnostic case artifact."""

    fixture_id: str
    source_type: str
    eval_case: str | None
    baseline_tokens: int
    optimized_tokens: int
    saved_tokens: int
    saved_ratio: float
    validation_status: str
    fallback_used: bool
    receipt_present: bool
    passed: bool
    change_type: str
    protected_value_count: int = 0


@dataclass(frozen=True, slots=True)
class TokenRegressionArtifactReview:
    """Complete review of a diagnostic artifact folder."""

    schema_version: int
    artifact_kind: str
    status: str
    artifact_dir: str
    summary: dict[str, Any]
    top_savings: tuple[dict[str, Any], ...]
    safety_checks: tuple[dict[str, Any], ...]
    issues: tuple[TokenRegressionArtifactReviewIssue, ...]
    marketing_interpretation: tuple[str, ...]
    case_reviews: tuple[TokenRegressionArtifactCaseReview, ...] = ()


def review_token_regression_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    """Review an existing token regression diagnostic artifact folder."""
    root = Path(artifact_dir)
    issues: list[TokenRegressionArtifactReviewIssue] = []

    if not root.is_dir():
        issues.append(
            _error_issue(
                "missing_artifact_dir",
                f"Artifact directory does not exist: {root}",
            )
        )
        return _build_review_dict(root, issues, case_reviews=())

    summary_payload, summary_issues = _load_json_file(root / "summary.json", "missing_summary")
    issues.extend(summary_issues)
    if summary_payload is None:
        return _build_review_dict(root, issues, case_reviews=())

    manifest_payload, manifest_issues = _load_json_file(
        root / "manifest.json",
        "missing_manifest",
    )
    issues.extend(manifest_issues)
    if manifest_payload is None:
        return _build_review_dict(root, issues, case_reviews=())

    case_paths = _resolve_case_paths(root, summary_payload, manifest_payload, issues)
    loaded_cases: list[dict[str, Any]] = []
    for relative_path in case_paths:
        case_path = root / relative_path
        if not case_path.is_file():
            fixture_id = Path(relative_path).stem
            issues.append(
                _error_issue(
                    "missing_case_artifact",
                    f"Case artifact missing: {relative_path}",
                    fixture_id=fixture_id,
                )
            )
            continue

        case_payload, case_issues = _load_json_file(case_path, "invalid_json")
        issues.extend(case_issues)
        if case_payload is None:
            continue

        missing_fields = [
            field_name
            for field_name in _REQUIRED_CASE_FIELDS
            if field_name not in case_payload
        ]
        if missing_fields:
            issues.append(
                _error_issue(
                    "missing_required_field",
                    (
                        f"Case {case_payload.get('fixture_id', relative_path)} "
                        f"missing fields: {', '.join(missing_fields)}"
                    ),
                    fixture_id=str(case_payload.get("fixture_id", Path(relative_path).stem)),
                )
            )
            continue

        loaded_cases.append(case_payload)
        if case_payload.get("passed") is False:
            issues.append(
                _error_issue(
                    "case_failed",
                    f"Fixture {case_payload['fixture_id']} did not pass.",
                    fixture_id=case_payload["fixture_id"],
                )
            )

    _verify_case_counts(summary_payload, manifest_payload, loaded_cases, issues)

    if summary_payload.get("failed", 0) > 0:
        issues.append(
            _error_issue(
                "summary_has_failures",
                f"Summary reports {summary_payload['failed']} failed fixture(s).",
            )
        )

    case_reviews = tuple(_build_case_review(case) for case in loaded_cases)
    issues.extend(_collect_warning_issues(root, loaded_cases, case_reviews, summary_payload))

    return _build_review_dict(root, issues, case_reviews, summary_payload)


def format_token_regression_artifact_review(review: Mapping[str, Any]) -> str:
    """Format a review dict as human-readable text."""
    lines = [
        "Token regression diagnostic artifact review",
        f"artifact_dir={review['artifact_dir']}",
        "",
        f"Status: {_format_status_label(review['status'])}",
        "",
        "Summary:",
    ]

    summary = review["summary"]
    lines.extend(
        [
            f"  fixtures: {summary['total_fixtures']}",
            f"  passed: {summary['passed']}",
            f"  failed: {summary['failed']}",
            f"  baseline_tokens: {summary['total_baseline_tokens']}",
            f"  optimized_tokens: {summary['total_optimized_tokens']}",
            f"  saved_tokens: {summary['total_saved_tokens']}",
            f"  saved_ratio: {summary['total_saved_ratio'] * 100:.2f}%",
            "",
            "Top savings:",
        ]
    )

    if review["top_savings"]:
        for index, entry in enumerate(review["top_savings"]):
            if index > 0:
                lines.append("")
            has_warning = entry.get("has_warning")
            prefix = "[WARN]" if has_warning else "[OK]"
            detail_indent = "         " if has_warning else "       "
            lines.append(f"  {prefix} {entry['fixture_id']}")
            lines.append(
                f"{detail_indent}"
                f"{entry['baseline_tokens']} -> {entry['optimized_tokens']} tokens, "
                f"saved={entry['saved_tokens']}, "
                f"ratio={entry['saved_ratio'] * 100:.2f}%"
            )
            lines.append(f"{detail_indent}change_type={entry['change_type']}")
    else:
        lines.append("  (none)")

    lines.extend(["", "Safety checks:"])
    if review["safety_checks"]:
        for entry in review["safety_checks"]:
            lines.append(f"  [OK] {entry['fixture_id']}")
            detail_parts = [
                f"validation={entry['validation_status']}",
                f"fallback={'true' if entry['fallback_used'] else 'false'}",
            ]
            if entry.get("protected_value_count", 0) > 0:
                detail_parts.append(f"protected_values={entry['protected_value_count']}")
            detail_parts.append(f"saved={entry['saved_tokens']}")
            lines.append(f"       {' '.join(detail_parts)}")
            lines.append(f"       {entry['message']}")
    else:
        lines.append("  (none)")

    lines.extend(["", "Issues:"])
    warning_issues = [
        issue for issue in review["issues"] if issue["severity"] == "warning"
    ]
    error_issues = [issue for issue in review["issues"] if issue["severity"] == "error"]
    if warning_issues or error_issues:
        for issue in error_issues + warning_issues:
            prefix = "[FAIL]" if issue["severity"] == "error" else "[WARN]"
            fixture_suffix = (
                f": {issue['fixture_id']}" if issue.get("fixture_id") else ""
            )
            lines.append(f"  {prefix} {issue['code']}{fixture_suffix}")
            lines.append(f"         {issue['message']}")
    else:
        lines.append("  (none)")

    lines.extend(["", "Marketing interpretation:"])
    for line in review["marketing_interpretation"]:
        lines.append(f"  {line}")

    return "\n".join(lines) + "\n"


def _build_review_dict(
    artifact_dir: Path,
    issues: Sequence[TokenRegressionArtifactReviewIssue],
    case_reviews: Sequence[TokenRegressionArtifactCaseReview],
    summary_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = _derive_status(issues)
    summary = _build_summary_block(summary_payload, case_reviews)
    top_savings = _build_top_savings(case_reviews, issues)
    safety_checks = _build_safety_checks(case_reviews)
    marketing = _build_marketing_interpretation(case_reviews, issues, summary)

    return {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "artifact_kind": _REVIEW_ARTIFACT_KIND,
        "status": status,
        "artifact_dir": str(artifact_dir),
        "summary": summary,
        "top_savings": top_savings,
        "safety_checks": safety_checks,
        "issues": [_issue_to_dict(issue) for issue in issues],
        "marketing_interpretation": marketing,
    }


def _derive_status(issues: Sequence[TokenRegressionArtifactReviewIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "fail"
    if any(issue.severity == "warning" for issue in issues):
        return "pass_with_warnings"
    return "pass"


def _build_summary_block(
    summary_payload: Mapping[str, Any] | None,
    case_reviews: Sequence[TokenRegressionArtifactCaseReview],
) -> dict[str, Any]:
    if summary_payload is not None:
        return {
            "total_fixtures": int(summary_payload.get("total_fixtures", 0)),
            "passed": int(summary_payload.get("passed", 0)),
            "failed": int(summary_payload.get("failed", 0)),
            "total_baseline_tokens": int(summary_payload.get("total_baseline_tokens", 0)),
            "total_optimized_tokens": int(summary_payload.get("total_optimized_tokens", 0)),
            "total_saved_tokens": int(summary_payload.get("total_saved_tokens", 0)),
            "total_saved_ratio": float(summary_payload.get("total_saved_ratio", 0.0)),
        }

    return {
        "total_fixtures": len(case_reviews),
        "passed": sum(1 for case in case_reviews if case.passed),
        "failed": sum(1 for case in case_reviews if not case.passed),
        "total_baseline_tokens": sum(case.baseline_tokens for case in case_reviews),
        "total_optimized_tokens": sum(case.optimized_tokens for case in case_reviews),
        "total_saved_tokens": sum(case.saved_tokens for case in case_reviews),
        "total_saved_ratio": 0.0,
    }


def _build_top_savings(
    case_reviews: Sequence[TokenRegressionArtifactCaseReview],
    issues: Sequence[TokenRegressionArtifactReviewIssue],
) -> list[dict[str, Any]]:
    warned_fixtures = {
        issue.fixture_id
        for issue in issues
        if issue.severity == "warning" and issue.fixture_id is not None
    }
    ranked = sorted(case_reviews, key=lambda case: case.saved_tokens, reverse=True)
    return [
        {
            "fixture_id": case.fixture_id,
            "baseline_tokens": case.baseline_tokens,
            "optimized_tokens": case.optimized_tokens,
            "saved_tokens": case.saved_tokens,
            "saved_ratio": case.saved_ratio,
            "change_type": case.change_type,
            "has_warning": case.fixture_id in warned_fixtures,
        }
        for case in ranked
        if case.saved_tokens > 0
    ]


def _build_safety_checks(
    case_reviews: Sequence[TokenRegressionArtifactCaseReview],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for case in sorted(case_reviews, key=lambda item: item.fixture_id):
        if case.change_type == _CHANGE_TYPE_FALLBACK:
            checks.append(
                {
                    "fixture_id": case.fixture_id,
                    "status": "ok",
                    "validation_status": case.validation_status,
                    "fallback_used": case.fallback_used,
                    "saved_tokens": case.saved_tokens,
                    "protected_value_count": case.protected_value_count,
                    "change_type": case.change_type,
                    "message": "expected fallback behavior confirmed",
                }
            )
        elif case.change_type == _CHANGE_TYPE_PROTECTED:
            checks.append(
                {
                    "fixture_id": case.fixture_id,
                    "status": "ok",
                    "validation_status": case.validation_status,
                    "fallback_used": case.fallback_used,
                    "saved_tokens": case.saved_tokens,
                    "protected_value_count": case.protected_value_count,
                    "change_type": case.change_type,
                    "message": "protected content preserved",
                }
            )
    return checks


def _build_marketing_interpretation(
    case_reviews: Sequence[TokenRegressionArtifactCaseReview],
    issues: Sequence[TokenRegressionArtifactReviewIssue],
    summary: Mapping[str, Any],
) -> list[str]:
    lines = [
        "Do not claim aggregate savings as global compression quality.",
    ]

    warning_codes = {issue.code for issue in issues if issue.severity == "warning"}
    if "dominant_savings_case" in warning_codes or "likely_truncation" in warning_codes:
        lines.append("Most savings may come from one long truncation case.")

    claim_parts = ["deterministic artifacts are generated"]
    has_protected = any(
        case.change_type == _CHANGE_TYPE_PROTECTED for case in case_reviews
    )
    has_fallback = any(case.change_type == _CHANGE_TYPE_FALLBACK for case in case_reviews)
    if has_protected or has_fallback:
        claim_parts.append("protected/fallback cases are validated")
    if summary.get("total_saved_tokens", 0) > 0:
        claim_parts.append("long-context payload reduction is measurable")
    lines.append(f"Safe claim: {', '.join(claim_parts)}.")

    return lines


def _resolve_case_paths(
    root: Path,
    summary_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    issues: list[TokenRegressionArtifactReviewIssue],
) -> list[str]:
    manifest_cases = list(manifest_payload.get("cases", []))
    summary_cases = [
        entry["case_artifact"]
        for entry in summary_payload.get("cases", [])
        if isinstance(entry, Mapping) and "case_artifact" in entry
    ]

    summary_case_set = set(summary_cases)
    manifest_case_set = set(manifest_cases)

    for relative_path in summary_cases:
        if relative_path not in manifest_case_set:
            issues.append(
                _error_issue(
                    "case_manifest_mismatch",
                    f"Summary references case not listed in manifest: {relative_path}",
                    fixture_id=Path(relative_path).stem,
                )
            )

    for relative_path in manifest_cases:
        if relative_path not in summary_case_set:
            issues.append(
                _error_issue(
                    "case_summary_mismatch",
                    f"Manifest references case not listed in summary: {relative_path}",
                    fixture_id=Path(relative_path).stem,
                )
            )

    return manifest_cases


def _verify_case_counts(
    summary_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    loaded_cases: Sequence[Mapping[str, Any]],
    issues: list[TokenRegressionArtifactReviewIssue],
) -> None:
    manifest_count = int(manifest_payload.get("case_count", -1))
    summary_count = int(summary_payload.get("total_fixtures", -1))
    loaded_count = len(loaded_cases)
    summary_entries = len(summary_payload.get("cases", []))

    if manifest_count != loaded_count:
        issues.append(
            _error_issue(
                "case_count_mismatch",
                (
                    f"Manifest case_count={manifest_count} "
                    f"does not match loaded cases={loaded_count}."
                ),
            )
        )
    if summary_count != loaded_count:
        issues.append(
            _error_issue(
                "case_count_mismatch",
                (
                    f"Summary total_fixtures={summary_count} "
                    f"does not match loaded cases={loaded_count}."
                ),
            )
        )
    if summary_entries != loaded_count:
        issues.append(
            _error_issue(
                "case_count_mismatch",
                (
                    f"Summary lists {summary_entries} case entries "
                    f"but {loaded_count} case artifacts were loaded."
                ),
            )
        )


def _build_case_review(case: Mapping[str, Any]) -> TokenRegressionArtifactCaseReview:
    metrics = case["metrics"]
    validation = case["validation"]
    metadata = case.get("metadata", {})
    source_type = str(case["source_type"])
    saved_tokens = int(metrics["saved_tokens"])
    saved_ratio = float(metrics["saved_ratio"])
    validation_status = str(validation["validation_status"])
    fallback_used = bool(validation["fallback_used"])
    eval_case = metadata.get("eval_case")
    if eval_case is not None:
        eval_case = str(eval_case)
    protected_value_count = int(metadata.get("protected_value_count", 0))

    original_text = _extract_primary_input_text(source_type, case["input"])
    optimized_text = _extract_primary_output_text(source_type, case["output"])
    truncation = _is_likely_truncation(original_text, optimized_text, saved_ratio)
    change_type = _classify_change_type(
        eval_case=eval_case,
        protected_value_count=protected_value_count,
        validation_status=validation_status,
        fallback_used=fallback_used,
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        likely_truncation=truncation,
    )

    return TokenRegressionArtifactCaseReview(
        fixture_id=str(case["fixture_id"]),
        source_type=source_type,
        eval_case=eval_case,
        baseline_tokens=int(metrics["baseline_tokens"]),
        optimized_tokens=int(metrics["optimized_tokens"]),
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        validation_status=validation_status,
        fallback_used=fallback_used,
        receipt_present=bool(validation["receipt_present"]),
        passed=bool(case["passed"]),
        change_type=change_type,
        protected_value_count=protected_value_count,
    )


def _classify_change_type(
    *,
    eval_case: str | None,
    protected_value_count: int,
    validation_status: str,
    fallback_used: bool,
    saved_tokens: int,
    saved_ratio: float,
    likely_truncation: bool,
) -> str:
    if eval_case == "fallback" and fallback_used and saved_tokens == 0:
        return _CHANGE_TYPE_FALLBACK
    if (
        protected_value_count > 0
        and validation_status == "passed"
        and not fallback_used
    ):
        return _CHANGE_TYPE_PROTECTED
    if saved_ratio >= _LARGE_SAVING_RATIO_THRESHOLD or likely_truncation:
        return _CHANGE_TYPE_TRUNCATION
    if saved_tokens > 0:
        return _CHANGE_TYPE_WHITESPACE
    if saved_tokens == 0:
        return _CHANGE_TYPE_NO_SAVINGS
    return _CHANGE_TYPE_UNKNOWN


def _collect_warning_issues(
    root: Path,
    loaded_cases: Sequence[Mapping[str, Any]],
    case_reviews: Sequence[TokenRegressionArtifactCaseReview],
    summary_payload: Mapping[str, Any],
) -> list[TokenRegressionArtifactReviewIssue]:
    issues: list[TokenRegressionArtifactReviewIssue] = []
    total_saved = int(summary_payload.get("total_saved_tokens", 0))

    if total_saved > 0 and case_reviews:
        dominant = max(case_reviews, key=lambda case: case.saved_tokens)
        if dominant.saved_tokens / total_saved > _DOMINANT_SAVINGS_THRESHOLD:
            issues.append(
                _warning_issue(
                    "dominant_savings_case",
                    (
                        f"{dominant.fixture_id} contributes most aggregate savings "
                        f"({dominant.saved_tokens}/{total_saved} tokens)."
                    ),
                    fixture_id=dominant.fixture_id,
                )
            )

    case_by_id = {case.fixture_id: case for case in case_reviews}
    for case_payload in loaded_cases:
        fixture_id = str(case_payload["fixture_id"])
        case_review = case_by_id[fixture_id]
        source_type = str(case_payload["source_type"])
        metrics = case_payload["metrics"]
        saved_ratio = float(metrics["saved_ratio"])

        if saved_ratio >= _LARGE_SAVING_RATIO_THRESHOLD:
            issues.append(
                _warning_issue(
                    "large_saving_ratio",
                    (
                        f"{fixture_id} saved_ratio={saved_ratio:.4f} "
                        f"is at or above {_LARGE_SAVING_RATIO_THRESHOLD:.2f}."
                    ),
                    fixture_id=fixture_id,
                )
            )

        original_text = _extract_primary_input_text(source_type, case_payload["input"])
        optimized_text = _extract_primary_output_text(source_type, case_payload["output"])
        if _is_likely_truncation(original_text, optimized_text, saved_ratio):
            issues.append(
                _warning_issue(
                    "likely_truncation",
                    f"{fixture_id} output appears truncated.",
                    fixture_id=fixture_id,
                )
            )

        for text in _collect_text_values(case_payload["input"]) + _collect_text_values(
            case_payload["output"]
        ):
            if _contains_encoding_issue(text):
                issues.append(
                    _warning_issue(
                        "possible_encoding_issue",
                        f"{fixture_id} contains suspicious encoding sequences.",
                        fixture_id=fixture_id,
                    )
                )
                break

        validation_status = str(case_payload["validation"]["validation_status"])
        if (
            bool(case_payload["passed"])
            and validation_status != "runner_error"
            and (
                _is_empty_payload(case_payload["input"])
                or _is_empty_payload(case_payload["output"])
            )
        ):
            issues.append(
                _warning_issue(
                    "missing_diagnostic_content",
                    f"{fixture_id} passed but diagnostic input or output is empty.",
                    fixture_id=fixture_id,
                )
            )

    return issues


def _load_json_file(
    path: Path,
    missing_code: str,
) -> tuple[dict[str, Any] | None, list[TokenRegressionArtifactReviewIssue]]:
    if not path.is_file():
        return None, [
            _error_issue(
                missing_code,
                f"Required artifact file missing: {path.name}",
            )
        ]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [
            _error_issue(
                "invalid_json",
                f"Invalid JSON in {path.name}: {exc.msg}",
            )
        ]

    if not isinstance(payload, dict):
        return None, [
            _error_issue(
                "invalid_json",
                f"Expected JSON object in {path.name}.",
            )
        ]

    return payload, []


def _extract_primary_input_text(source_type: str, payload: Mapping[str, Any]) -> str:
    if source_type == "context_pack":
        fragments = payload.get("fragments", [])
        if isinstance(fragments, list) and fragments:
            return "\n".join(
                str(fragment.get("original_content", ""))
                for fragment in fragments
                if isinstance(fragment, Mapping)
            )
        return str(payload.get("original_content", ""))
    if source_type == "memory_summary":
        return str(payload.get("original_summary", ""))
    if source_type == "tool_schema":
        return str(payload.get("original_tool_catalog", ""))
    return str(payload.get("original_content", ""))


def _extract_primary_output_text(source_type: str, payload: Mapping[str, Any]) -> str:
    if source_type == "context_pack":
        fragments = payload.get("fragments", [])
        if isinstance(fragments, list) and fragments:
            return "\n".join(
                str(fragment.get("optimized_content", ""))
                for fragment in fragments
                if isinstance(fragment, Mapping)
            )
        return str(payload.get("optimized_content", ""))
    if source_type == "memory_summary":
        return str(payload.get("optimized_summary", ""))
    if source_type == "tool_schema":
        return str(payload.get("optimized_tool_catalog", ""))
    return str(payload.get("optimized_content", ""))


def _collect_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        texts: list[str] = []
        for nested in value.values():
            texts.extend(_collect_text_values(nested))
        return texts
    if isinstance(value, list):
        texts = []
        for nested in value:
            texts.extend(_collect_text_values(nested))
        return texts
    return []


def _is_empty_payload(payload: Any) -> bool:
    if payload in (None, {}, []):
        return True
    texts = _collect_text_values(payload)
    return not any(text.strip() for text in texts)


def _is_likely_truncation(
    original_text: str,
    optimized_text: str,
    saved_ratio: float,
) -> bool:
    if optimized_text.endswith(_TRUNCATION_SUFFIXES):
        return True
    if not original_text:
        return False
    if (
        saved_ratio >= _LARGE_SAVING_RATIO_THRESHOLD
        and len(optimized_text) < _TRUNCATION_LENGTH_RATIO * len(original_text)
    ):
        return True
    return False


def _contains_encoding_issue(text: str) -> bool:
    return any(sequence in text for sequence in _ENCODING_SUSPICIOUS_SEQUENCES)


def _error_issue(
    code: str,
    message: str,
    *,
    fixture_id: str | None = None,
) -> TokenRegressionArtifactReviewIssue:
    return TokenRegressionArtifactReviewIssue(
        severity="error",
        code=code,
        message=message,
        fixture_id=fixture_id,
    )


def _warning_issue(
    code: str,
    message: str,
    *,
    fixture_id: str | None = None,
) -> TokenRegressionArtifactReviewIssue:
    return TokenRegressionArtifactReviewIssue(
        severity="warning",
        code=code,
        message=message,
        fixture_id=fixture_id,
    )


def _issue_to_dict(issue: TokenRegressionArtifactReviewIssue) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
    }
    if issue.fixture_id is not None:
        payload["fixture_id"] = issue.fixture_id
    return payload


def _format_status_label(status: str) -> str:
    if status == "pass_with_warnings":
        return "PASS WITH WARNINGS"
    return status.upper()
