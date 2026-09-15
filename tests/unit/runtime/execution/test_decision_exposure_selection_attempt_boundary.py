# © Artur Czarnecki. All rights reserved.

"""I1-A-R2 central single-attempt exposure selection boundary tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_authoritative_exposure import DecisionEvaluationScope
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionFailure,
    DecisionExposureSelectionFailureCode,
    DecisionExposureSelectionSuccessReason,
    HostPublicationClass,
)
from intergrax.contracts.execution_identity import AttemptId, mint_attempt_id
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.runtime.decision_plugin_composition import (
    DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID,
    DecisionPluginLoadPolicy,
)
from intergrax.runtime.execution.decision_exposure_selection_composition import (
    compose_decision_exposure_selection,
)
from intergrax.runtime.execution.decision_exposure_selection_validation import (
    run_validated_decision_exposure_selection,
)
from intergrax.runtime.execution.host_terminal_decision_exposure_selector import (
    default_decision_exposure_selection_strategy,
)
from tests.unit.runtime.execution.test_decision_exposure_selection import (
    _accepted,
    _candidate,
    _graph_policy,
)
from tests.unit.runtime.execution.test_decision_exposure_selection_plugin_admission import (
    _install_eps,
    _exposure_ep,
    _manifest_for,
    _mock_distribution,
    _selection_ref,
)

pytestmark = pytest.mark.unit

_PACKAGE = "exposure-selection-trust-pkg"
_DELEGATE_TARGET = "testing_support.decision_plugins.exposure_selection_trust:ExposureSelectionTrustDelegate"
_PLUGIN_ID = "preload.exposure.trust.delegate"


class _SelectFirstStrategy:
    call_count: int = 0

    @property
    def strategy_id(self) -> str:
        return "test.select_first"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        _SelectFirstStrategy.call_count += 1
        return DecisionExposureSelectionDecision(
            selected=candidates[0].exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


class _SelectSecondStrategy:
    call_count: int = 0

    @property
    def strategy_id(self) -> str:
        return "test.select_second"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        _SelectSecondStrategy.call_count += 1
        return DecisionExposureSelectionDecision(
            selected=candidates[1].exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


@pytest.fixture(autouse=True)
def _reset_strategy_counters() -> None:
    _SelectFirstStrategy.call_count = 0
    _SelectSecondStrategy.call_count = 0
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _mixed_attempt_candidates() -> tuple[
    DecisionExposureCandidate[object],
    DecisionExposureCandidate[object],
]:
    attempt_old = mint_attempt_id()
    attempt_new = mint_attempt_id()
    exposure_old = _accepted("old")
    exposure_new = _accepted("new")
    return (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure_old,
            ordinal=0,
            attempt_id=attempt_old,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure_new,
            ordinal=1,
            attempt_id=attempt_new,
        ),
    )


def test_mixed_attempts_reject_before_strategy_select_first_not_called() -> None:
    candidates = _mixed_attempt_candidates()
    outcome = run_validated_decision_exposure_selection(
        _SelectFirstStrategy(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.INVALID_CANDIDATE_SET
    assert "exactly one effective attempt" in outcome.detail
    assert _SelectFirstStrategy.call_count == 0


def test_mixed_attempts_reject_even_when_strategy_would_pick_new_attempt() -> None:
    candidates = _mixed_attempt_candidates()
    outcome = run_validated_decision_exposure_selection(
        _SelectSecondStrategy(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.INVALID_CANDIDATE_SET
    assert _SelectSecondStrategy.call_count == 0


def test_same_attempt_candidates_invoke_strategy() -> None:
    attempt = mint_attempt_id()
    exposure = _accepted("ok")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=_accepted("alt"),
            ordinal=1,
            subject="subj-b",
            attempt_id=attempt,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        _SelectFirstStrategy(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionDecision)
    assert _SelectFirstStrategy.call_count == 1


def test_mixed_attempt_ids_detected_without_ordering_semantics() -> None:
    attempt_low = AttemptId("attempt_00000000000000000000000000000001")
    attempt_high = AttemptId("attempt_ffffffffffffffffffffffffffffffff")
    exposure = _accepted("x")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt_high,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=1,
            attempt_id=attempt_low,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        _SelectFirstStrategy(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.INVALID_CANDIDATE_SET
    assert _SelectFirstStrategy.call_count == 0


def test_empty_candidate_set_preserves_no_eligible_semantics() -> None:
    outcome = run_validated_decision_exposure_selection(
        default_decision_exposure_selection_strategy(),
        _graph_policy(),
        (),
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.NO_ELIGIBLE_TERMINAL_CANDIDATE


def test_admitted_external_strategy_passes_with_single_attempt_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _selection_ref(_PLUGIN_ID)
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _DELEGATE_TARGET)])
    _mock_distribution(
        monkeypatch,
        _manifest_for(
            entries=(
                ("external_selector", _PLUGIN_ID, DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID),
            ),
        ),
    )
    policy = DecisionPluginLoadPolicy(
        require_manifest_capability_binding=True,
        requested_exposure_selection_strategy_plugins=(ref,),
    )
    composition = compose_decision_exposure_selection(policy=policy, selection_ref=ref)
    attempt = mint_attempt_id()
    exposure = _accepted("plugin-path")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        composition.strategy,
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionDecision)
