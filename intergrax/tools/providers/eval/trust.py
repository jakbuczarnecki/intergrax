# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed trust-boundary contracts for eval judge input composition."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.tools.providers.eval.contracts import EvalJudgeInput

CANDIDATE_PAYLOAD_SCHEMA = "intergrax.eval.candidate.v1"


@dataclass(frozen=True, slots=True)
class EvalTrustedRubricContext:
    """Verifier-controlled rubric authority for one judge evaluation."""

    rubric_id: str
    criteria: tuple[str, ...]
    min_score: float
    reference_context: str | None


@dataclass(frozen=True, slots=True)
class EvalUntrustedCandidateContent:
    """Candidate-controlled output evaluated as untrusted data."""

    text: str


def trusted_rubric_from_judge_input(params: EvalJudgeInput) -> EvalTrustedRubricContext:
    """Materialize trusted rubric context from one wire-level judge input."""
    return EvalTrustedRubricContext(
        rubric_id=params.rubric_id,
        criteria=tuple(params.criteria),
        min_score=params.min_score,
        reference_context=params.reference_context,
    )


def untrusted_candidate_from_judge_input(
    params: EvalJudgeInput,
) -> EvalUntrustedCandidateContent:
    """Materialize untrusted candidate content from one wire-level judge input."""
    return EvalUntrustedCandidateContent(text=params.output_text)
