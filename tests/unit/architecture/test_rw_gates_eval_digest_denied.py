"""NT-RW-GATES-DIGEST-DENY — tampered eval digests are rejected."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_nt_debt_04_enforce_eval_run_digest_guard() -> None:
    """NT-DEBT-04 red-first: eval-run digest enforcement must exist in frozen scope."""
    mod = importlib.import_module("intergrax.contracts.evaluation_contracts")
    fn = getattr(mod, "enforce_eval_run_digests", None)
    assert callable(fn)


def _minimal_run(*, tenant: str = "t1"):
    from intergrax.contracts.evaluation_contracts import EvalRevisionRef, EvalRun

    agent = EvalRevisionRef(ref_id="a", revision="1")
    return EvalRun(
        tenant_id=tenant,
        agent=agent,
        model=agent,
        prompt=agent,
        profile=agent,
        code_sha="a" * 40,
        dataset_id="ds",
        dataset_version="1",
        dataset_digest="sha256:" + "b" * 64,
        seed=1,
    )


def test_tampered_result_digest_raises() -> None:
    from intergrax.contracts.evaluation_contracts import (
        EvalDigestMismatchError,
        enforce_eval_run_digests,
    )

    run = _minimal_run()
    tampered = run.model_copy(update={"dataset_version": "2"})
    with pytest.raises(EvalDigestMismatchError):
        enforce_eval_run_digests(
            tampered,
            declared_inputs_digest=run.inputs_digest,
            declared_result_digest=run.result_digest,
        )


def test_wrong_declared_result_digest_denied() -> None:
    from intergrax.contracts.evaluation_contracts import (
        EvalDigestMismatchError,
        enforce_eval_run_digests,
    )

    run = _minimal_run()
    with pytest.raises(EvalDigestMismatchError):
        bogus = "sha256:" + "c" * 64
        enforce_eval_run_digests(
            run,
            declared_inputs_digest=run.inputs_digest,
            declared_result_digest=bogus,
        )


def test_comparison_digest_mismatch_denied() -> None:
    from intergrax.contracts.evaluation_contracts import (
        EvalComparison,
        EvalDigestMismatchError,
        EvalThresholds,
        verify_eval_comparison_digests,
    )

    baseline = _minimal_run()
    candidate = _minimal_run()
    comparison = EvalComparison(
        tenant_id="t1",
        dataset_digest=baseline.dataset_digest,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_result_digest=baseline.result_digest,
        candidate_result_digest=candidate.result_digest,
        thresholds=EvalThresholds(),
        metric_deltas=(),
    )
    tampered = comparison.model_copy(update={"candidate_run_id": "evalrun:deadbeef"})
    with pytest.raises(EvalDigestMismatchError):
        verify_eval_comparison_digests(tampered, baseline_run=baseline, candidate_run=candidate)
