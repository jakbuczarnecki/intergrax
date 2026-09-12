# © Artur Czarnecki. All rights reserved.

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
    CompletionMode,
)
from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from testing_support.decision_e2e.local_qualification_session.alignment_revision_evidence import (
    infer_alignment_revision_evidence,
)


def test_natural_overcommit_repair_chain() -> None:
    run_id = str(mint_run_id())
    pre = CompletionAlignmentDiagV1(
        run_id=run_id,
        node_id="investigator",
        completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
        alignment_status=AlignmentStatus.MISMATCH,
        alignment_direction=AlignmentDirection.REVERSE,
        mismatch_reason="x",
        correctable=True,
        supported_state_present=False,
        supported_hypothesis_id=None,
        supported_resolution=None,
    )
    post = CompletionAlignmentDiagV1(
        run_id=run_id,
        node_id="investigator",
        completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS,
        alignment_status=AlignmentStatus.MATCH,
        alignment_direction=AlignmentDirection.NONE,
        mismatch_reason=None,
        correctable=False,
        supported_state_present=True,
        supported_hypothesis_id="h1",
        supported_resolution="r1",
    )
    attempts = (
        EvaluatorModelAttemptDiagV1(
            run_id=run_id,
            node_id="investigator",
            attempt_index=0,
            max_iterations=2,
        ),
        EvaluatorModelAttemptDiagV1(
            run_id=run_id,
            node_id="investigator",
            attempt_index=1,
            max_iterations=2,
        ),
    )
    flags = infer_alignment_revision_evidence((pre, post), attempts)
    assert flags.natural_overcommit_repair is True
    assert flags.revision_repaired is True
