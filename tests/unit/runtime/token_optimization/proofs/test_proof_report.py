from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.proofs.contracts import ProofArtifactError
from intergrax.runtime.token_optimization.proofs.evaluation_contracts import (
    CaseEvaluation,
    EvaluationProfile,
    GateResult,
    GateStatus,
    UniversalProofEvaluation,
)
from intergrax.runtime.token_optimization.proofs.report import (
    escape_markdown,
    render_evaluation_markdown,
    write_evaluation_artifacts,
)


def _evaluation() -> UniversalProofEvaluation:
    gate = GateResult(
        gate_id="RAW_CONTENT_ABSENT",
        status=GateStatus.PASS,
        case_id="case-one",
        reason_code="EXPECTATION_SATISFIED",
        expected_safe_summary="safe",
        actual_safe_summary="safe",
        required=True,
    )
    return UniversalProofEvaluation(
        evaluation_id="evaluation-fixed",
        proof_id="proof",
        run_id="run",
        corpus_version="token-optimization-proof-corpus.v1",
        evaluation_version="token-10g.v1",
        run_mode="offline_smoke",
        provider="vllm",
        model="synthetic-model",
        cases=(
            CaseEvaluation(
                case_id="case-one",
                category="short_clean_prompt",
                description="Synthetic safe description.",
                gates=(gate,),
            ),
        ),
        status_counts={
            status.value: (1 if status is GateStatus.PASS else 0)
            for status in GateStatus
        },
        success=True,
        profile=EvaluationProfile.OFFLINE_COMPOSITION,
    )


def test_markdown_is_escaped_and_deterministic() -> None:
    assert escape_markdown("a|b") == "a\\|b"
    assert escape_markdown("a`b") == "a\\`b"
    assert "<b>" not in escape_markdown("<b>")
    assert "\\n" in escape_markdown("a\n# b")
    rendered = render_evaluation_markdown(_evaluation())
    assert rendered == render_evaluation_markdown(_evaluation())
    assert "offline_composition" in rendered
    assert "does not establish behavior-specific LLM routing quality" in rendered


def test_artifacts_are_canonical_hashed_and_duplicate_safe(tmp_path: Path) -> None:
    first = write_evaluation_artifacts(_evaluation(), output_directory=tmp_path)
    second_dir = tmp_path / "evaluation-fixed"
    manifest = json.loads(
        (second_dir / "evaluation-manifest.json").read_text(encoding="utf-8")
    )
    assert first.artifact_refs == (
        "evaluation.json",
        "report.md",
        "evaluation-manifest.json",
    )
    assert (second_dir / "evaluation.json").read_bytes().endswith(b"\n")
    for ref in manifest["files"]:
        path = second_dir / ref["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
    assert "Synthetic safe description." in (second_dir / "report.md").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ProofArtifactError, match="EVALUATION_DIRECTORY_EXISTS"):
        write_evaluation_artifacts(_evaluation(), output_directory=tmp_path)
