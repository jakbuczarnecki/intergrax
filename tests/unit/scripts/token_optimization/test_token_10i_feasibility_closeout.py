from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_PROOF_DIR = _ROOT / "docs" / "proofs" / "token_optimization"
_FEASIBILITY_FILE = _PROOF_DIR / "token-10i-qwen3-14b-awq-feasibility.md"
_README_FILE = _PROOF_DIR / "README.md"
_ROADMAP_FILE = _ROOT / "docs" / "project" / "capabilities" / "plan" / "TOKEN_OPTIMIZATION.md"

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


def test_token_10i_feasibility_evidence_is_safe_and_claim_consistent() -> None:
    feasibility = _FEASIBILITY_FILE.read_text(encoding="utf-8")
    proof_readme = _README_FILE.read_text(encoding="utf-8")
    roadmap = _ROADMAP_FILE.read_text(encoding="utf-8")

    assert "BLOCKED_HARDWARE_CAPACITY_FINAL" in feasibility
    assert "Qwen/Qwen3-14B-AWQ" in feasibility
    assert "9.44 GiB" in feasibility
    assert "-0.89 GiB" in feasibility
    assert "8192" in feasibility
    assert "0.95" in feasibility
    assert "10.79/11.99 GiB" in feasibility
    assert "Frozen qualification for `Qwen/Qwen3-14B-AWQ` did not start" in feasibility
    assert "TOKEN-10I BLOCKED_HARDWARE_CAPACITY_FINAL" in roadmap
    assert "TOKEN-10H CLOSED" in roadmap
    assert "NOT QUALIFIED" in roadmap
    assert "TOKEN-10H is not closed" not in roadmap
    assert "BLOCKED_HARDWARE_CAPACITY_FINAL" in proof_readme
    assert "token-10i-qwen3-14b-awq-feasibility.md" in proof_readme

    for text in (feasibility, proof_readme):
        for marker in _FORBIDDEN_MARKERS:
            assert marker not in text
