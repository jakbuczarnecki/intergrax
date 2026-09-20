# © Artur Czarnecki. All rights reserved.

"""UCA-3 — acquisition service, registry, governance, and selection policy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.capability_acquisition.acquisition_registry import (
    CapabilityAcquisitionStrategyRegistry,
)
from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.default_strategy_selection_policy import (
    DefaultCapabilityAcquisitionStrategySelectionPolicy,
)
from intergrax.contracts.capability_acquisition.acquisition_authorization import (
    CapabilityAcquisitionAuthorizationOutcome,
    CapabilityAcquisitionAuthorizationResult,
)
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_acquisition.errors import (
    CapabilityAcquisitionConfigurationError,
    CapabilityAcquisitionIntegrityError,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.capability_acquisition.strategy_selection import (
    CapabilityAcquisitionGovernanceContext,
    CapabilityAcquisitionStrategySelection,
    CapabilityAcquisitionStrategySelectionOutcome,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def _request(
    gap: CapabilityGap,
    *,
    nonce: str = "nonce-1",
    kinds: tuple[CapabilityKind, ...] = (CapabilityKind.TOOL,),
) -> CapabilityAcquisitionRequest:
    need = CapabilityNeed(need_id=gap.need_id, kinds=kinds)
    return CapabilityAcquisitionRequest(
        request_id=derive_capability_acquisition_request_id(
            gap_id=gap.gap_id,
            request_nonce=nonce,
        ),
        request_nonce=nonce,
        capability_gap=gap,
        capability_need=need,
        correlation_id="corr-acq",
        causation_id="cause-acq",
        requested_at=_CREATED,
    )


class _FakeStrategy:
    def __init__(
        self,
        *,
        strategy_id: str,
        kinds: frozenset[CapabilityKind],
        supports_override: bool | None = None,
    ) -> None:
        self._strategy_id = strategy_id
        self._kinds = kinds
        self._supports_override = supports_override
        self.calls = 0

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return self._kinds

    def supports(self, request: CapabilityAcquisitionRequest) -> bool:
        if self._supports_override is not None:
            return self._supports_override
        need = request.capability_need
        if need is None or not need.kinds:
            return True
        return any(kind in self._kinds for kind in need.kinds)

    def acquire(
        self, request: CapabilityAcquisitionRequest
    ) -> CapabilityAcquisitionResult:
        self.calls += 1
        return CapabilityAcquisitionResult(
            request_id=request.request_id,
            gap_id=request.capability_gap.gap_id,
            strategy_id=self._strategy_id,
            outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
            started_at=_CREATED,
            completed_at=_CREATED,
            evidence=CapabilityAcquisitionEvidence(
                domain_handoff_reference="handoff://test/acquired",
            ),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


def test_valid_gap_request_accepted() -> None:
    gap = _gap()
    request = _request(gap)
    assert request.capability_gap.gap_id == gap.gap_id


def test_no_strategies_returns_no_strategy() -> None:
    service = CapabilityAcquisitionService(())
    result = service.acquire(_request(_gap()))
    assert result.outcome is CapabilityAcquisitionOutcome.NO_STRATEGY
    assert result.reason_code is CapabilityAcquisitionReasonCode.NO_STRATEGY


def test_one_strategy_invoked_once() -> None:
    strategy = _FakeStrategy(
        strategy_id="fake.acquire",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    service = CapabilityAcquisitionService((strategy,))
    result = service.acquire(_request(_gap()))
    assert result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert result.strategy_id == "fake.acquire"
    assert strategy.calls == 1


def test_multiple_eligible_default_policy_conflict() -> None:
    a = _FakeStrategy(strategy_id="a", kinds=frozenset({CapabilityKind.TOOL}))
    b = _FakeStrategy(strategy_id="b", kinds=frozenset({CapabilityKind.TOOL}))
    service = CapabilityAcquisitionService((a, b))
    result = service.acquire(_request(_gap()))
    assert result.outcome is CapabilityAcquisitionOutcome.CONFLICT
    assert a.calls == 0
    assert b.calls == 0


def test_custom_selection_policy_chooses_b() -> None:
    class _PickBPolicy:
        def select(
            self,
            *,
            request: CapabilityAcquisitionRequest,
            candidates: tuple[CapabilityAcquisitionStrategyDescriptor, ...],
            governance_context: CapabilityAcquisitionGovernanceContext,
        ) -> CapabilityAcquisitionStrategySelection:
            del request, governance_context
            chosen = next(c for c in candidates if c.strategy_id == "b")
            return CapabilityAcquisitionStrategySelection(
                outcome=CapabilityAcquisitionStrategySelectionOutcome.SELECTED,
                strategy_id=chosen.strategy_id,
                reason_code=CapabilityAcquisitionReasonCode.NONE,
            )

    a = _FakeStrategy(strategy_id="a", kinds=frozenset({CapabilityKind.TOOL}))
    b = _FakeStrategy(strategy_id="b", kinds=frozenset({CapabilityKind.TOOL}))
    service = CapabilityAcquisitionService(
        (a, b),
        selection_policy=_PickBPolicy(),
    )
    result = service.acquire(_request(_gap()))
    assert result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert result.strategy_id == "b"
    assert a.calls == 0
    assert b.calls == 1


def test_duplicate_strategy_id_fails() -> None:
    first = _FakeStrategy(strategy_id="dup", kinds=frozenset({CapabilityKind.TOOL}))
    second = _FakeStrategy(strategy_id="dup", kinds=frozenset({CapabilityKind.SKILL}))
    with pytest.raises(CapabilityAcquisitionConfigurationError):
        CapabilityAcquisitionStrategyRegistry((first, second))


def test_governance_blocked_skips_strategy() -> None:
    class _BlockAuth:
        def authorize(self, request, eligible):
            del request, eligible
            return CapabilityAcquisitionAuthorizationResult(
                outcome=CapabilityAcquisitionAuthorizationOutcome.BLOCKED,
                reason_detail="policy deny",
            )

    strategy = _FakeStrategy(
        strategy_id="fake.acquire",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=_BlockAuth(),
    )
    result = service.acquire(_request(_gap()))
    assert result.outcome is CapabilityAcquisitionOutcome.BLOCKED
    assert strategy.calls == 0


def test_governance_hitl_skips_strategy() -> None:
    class _HitlAuth:
        def authorize(self, request, eligible):
            del request, eligible
            return CapabilityAcquisitionAuthorizationResult(
                outcome=CapabilityAcquisitionAuthorizationOutcome.REQUIRES_HITL,
            )

    strategy = _FakeStrategy(
        strategy_id="fake.acquire",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=_HitlAuth(),
    )
    result = service.acquire(_request(_gap()))
    assert result.outcome is CapabilityAcquisitionOutcome.REQUIRES_HITL
    assert strategy.calls == 0


def test_result_integrity_mismatch_raises() -> None:
    class _BadStrategy(_FakeStrategy):
        def acquire(
            self, request: CapabilityAcquisitionRequest
        ) -> CapabilityAcquisitionResult:
            self.calls += 1
            return CapabilityAcquisitionResult(
                request_id="wrong-request",
                gap_id=request.capability_gap.gap_id,
                strategy_id=self.strategy_id,
                outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
                reason_code=CapabilityAcquisitionReasonCode.NONE,
                started_at=_CREATED,
                completed_at=_CREATED,
                evidence=CapabilityAcquisitionEvidence(
                    artifact_reference="artifact://x",
                ),
            )

    strategy = _BadStrategy(
        strategy_id="bad",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    service = CapabilityAcquisitionService((strategy,))
    with pytest.raises(CapabilityAcquisitionIntegrityError):
        service.acquire(_request(_gap()))


def test_supports_declared_kinds_inconsistency_fails_closed() -> None:
    strategy = _FakeStrategy(
        strategy_id="inconsistent",
        kinds=frozenset({CapabilityKind.SKILL}),
        supports_override=True,
    )
    service = CapabilityAcquisitionService((strategy,))
    result = service.acquire(_request(_gap(), kinds=(CapabilityKind.TOOL,)))
    assert result.outcome is CapabilityAcquisitionOutcome.FAILED
    assert (
        result.reason_code
        is CapabilityAcquisitionReasonCode.STRATEGY_METADATA_INCONSISTENT
    )
    assert strategy.calls == 0


def test_deterministic_request_id() -> None:
    gap = _gap()
    first = _request(gap, nonce="stable")
    second = _request(gap, nonce="stable")
    assert first.request_id == second.request_id
    different = _request(gap, nonce="other")
    assert different.request_id != first.request_id


def test_default_selection_policy_semantics() -> None:
    policy = DefaultCapabilityAcquisitionStrategySelectionPolicy()
    request = _request(_gap())
    empty = policy.select(
        request=request,
        candidates=(),
        governance_context=CapabilityAcquisitionGovernanceContext(),
    )
    assert empty.outcome is CapabilityAcquisitionStrategySelectionOutcome.NO_STRATEGY

    one = policy.select(
        request=request,
        candidates=(CapabilityAcquisitionStrategyDescriptor(strategy_id="only"),),
        governance_context=CapabilityAcquisitionGovernanceContext(),
    )
    assert one.outcome is CapabilityAcquisitionStrategySelectionOutcome.SELECTED
    assert one.strategy_id == "only"

    conflict = policy.select(
        request=request,
        candidates=(
            CapabilityAcquisitionStrategyDescriptor(strategy_id="a"),
            CapabilityAcquisitionStrategyDescriptor(strategy_id="b"),
        ),
        governance_context=CapabilityAcquisitionGovernanceContext(),
    )
    assert conflict.outcome is CapabilityAcquisitionStrategySelectionOutcome.CONFLICT
