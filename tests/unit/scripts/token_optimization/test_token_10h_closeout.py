from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_PROOF_DIR = _ROOT / "docs" / "proofs" / "token_optimization"
_RUN_FILES = (
    _PROOF_DIR / "token-10h-live-run-1.safe.json",
    _PROOF_DIR / "token-10h-live-run-2.safe.json",
)
_EVALUATION_FILE = _PROOF_DIR / "token-10h-evaluation.safe.json"
_REPORT_FILE = _PROOF_DIR / "token-10h-live-report.md"
_README_FILE = _PROOF_DIR / "README.md"
_ROADMAP_FILE = _ROOT / "docs" / "features" / "plan" / "TOKEN_OPTIMIZATION.md"

_RUN_KEYS = {
    "schema_version",
    "proof_id",
    "run_id",
    "generated_at",
    "provider",
    "model",
    "adapter_type",
    "temperature",
    "corpus",
    "case_count",
    "evaluation_success",
    "technical_execution_success",
    "runtime_safety_success",
    "model_behavioral_success",
    "cases",
    "aggregate",
}
_CASE_KEYS = {
    "case_id",
    "status",
    "model_behavioral_failure",
    "runtime_safety_failure",
    "model_gate_outcomes",
    "final_runtime_gate_outcomes",
    "safe_reason_codes",
    "policy_override_applied",
    "policy_override_reason",
    "expected_execution",
    "actual_execution_status",
    "protected_counts",
    "measurement_counts_ratios",
    "digests",
}
_GATE_KEYS = {
    "gate_id",
    "status",
    "reason_code",
    "expected_safe_summary",
    "actual_safe_summary",
}
_MISMATCHES = {
    "case-high-risk-lossy-content",
    "case-warm-cache",
}
_FORBIDDEN_MARKERS = (
    "TOP_SECRET",
    "SYNTHETIC_RUN_001",
    "E_SYNTHETIC_PROTECTED",
    "SYNTHETIC_SECURITY_WARNING",
    "SYNTHETIC_FINDING_007",
    "https://synthetic.invalid",
    "synthetic/artifacts/fixed-run",
    "C:\\Users\\",
    "D:\\",
    "/Users/",
    "/home/",
    "127.0.0.1",
    "localhost",
    "authorization:",
    "api_key",
    "bearer ",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_canonical_json(path: Path, payload: dict) -> None:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    assert path.read_text(encoding="utf-8") == f"{canonical}\n"


def _assert_gate_shape(gates: list[dict]) -> None:
    for gate in gates:
        assert set(gate) == _GATE_KEYS
        assert gate["status"] in {
            "PASS",
            "FAIL",
            "UNAVAILABLE",
            "NOT_APPLICABLE",
        }


def test_token_10h_checked_in_evidence_is_safe_and_claim_consistent() -> None:
    runs = [_load(path) for path in _RUN_FILES]
    evaluation = _load(_EVALUATION_FILE)

    for path, run in zip(_RUN_FILES, runs, strict=True):
        _assert_canonical_json(path, run)
        assert set(run) == _RUN_KEYS
        assert run["evaluation_success"] is False
        assert run["technical_execution_success"] is True
        assert run["runtime_safety_success"] is True
        assert run["aggregate"]["technical_failed_count"] == 0
        assert run["aggregate"]["runtime_safety_failure_count"] == 0
        assert run["aggregate"]["integrity_failure_count"] == 1
        assert run["aggregate"]["policy_override_count"] == 1
        assert run["aggregate"]["gate_counts"] == {
            "FAIL": 3,
            "PASS": 349,
            "UNAVAILABLE": 9,
        }
        assert set(run["aggregate"]["model_mismatch_case_ids"]) == _MISMATCHES
        assert run["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"

        for case in run["cases"]:
            assert set(case) == _CASE_KEYS
            assert case["status"] in {"PASS", "FAIL", "UNAVAILABLE"}
            _assert_gate_shape(case["model_gate_outcomes"])
            _assert_gate_shape(case["final_runtime_gate_outcomes"])
            assert all(
                gate["gate_id"].startswith("MODEL_ROUTER_")
                for gate in case["model_gate_outcomes"]
            )
            assert all(
                not gate["gate_id"].startswith("MODEL_ROUTER_")
                for gate in case["final_runtime_gate_outcomes"]
            )
            if case["policy_override_applied"]:
                assert case["policy_override_reason"] == (
                    "security_warning_requires_review"
                )
            else:
                assert case["policy_override_reason"] is None

    assert runs[0]["aggregate"] == runs[1]["aggregate"]
    assert evaluation["live"]["evaluation_success"] is False
    assert evaluation["live"]["runtime_safety_success"] is True
    assert evaluation["live"]["model_case_level_compliance"] == "14/16"
    assert set(evaluation["live"]["model_mismatch_case_ids"]) == _MISMATCHES
    assert evaluation["claim_matrix"]["full_model_behavioral_compliance"] == (
        "NOT_PROVEN"
    )
    assert (
        evaluation["claim_matrix"][
            "protected_value_preservation_at_final_runtime_boundary"
        ]
        == "PROVEN"
    )
    assert (
        evaluation["claim_matrix"]["runtime_safety_for_16_case_synthetic_corpus"]
        == "PROVEN"
    )
    assert (
        evaluation["claim_matrix"]["deterministic_high_risk_runtime_enforcement"]
        == "PROVEN"
    )
    assert evaluation["claim_matrix"]["model_case_level_compliance"] == (
        "PARTIALLY_PROVEN — 14/16"
    )

    high_risk = next(
        case
        for case in runs[0]["cases"]
        if case["case_id"] == "case-high-risk-lossy-content"
    )
    assert high_risk["model_behavioral_failure"] is True
    assert high_risk["runtime_safety_failure"] is False
    assert high_risk["policy_override_applied"] is True
    assert high_risk["policy_override_reason"] == ("security_warning_requires_review")
    assert high_risk["actual_execution_status"]["final_runtime_status"] == (
        "review_required"
    )

    for path in (*_RUN_FILES, _EVALUATION_FILE):
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_MARKERS:
            assert marker not in text

    report = _REPORT_FILE.read_text(encoding="utf-8")
    proof_readme = _README_FILE.read_text(encoding="utf-8")
    roadmap = _ROADMAP_FILE.read_text(encoding="utf-8")
    assert "evaluation_success=false" in report
    assert "Public promotion is `WITHHELD`" in report
    assert "Public promotion: `WITHHELD`" in proof_readme
    assert "TOKEN-10G CLOSED" in roadmap
    assert "TOKEN-10H CLOSED" in roadmap
    assert "NOT QUALIFIED" in roadmap
    assert "TOKEN-10H is not closed" not in roadmap
