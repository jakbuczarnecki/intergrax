# © Artur Czarnecki. All rights reserved.

"""G5C-2B-1-R1/R2 — External Work immutable scope provenance for governed continuation."""

from __future__ import annotations

import pytest

from external_contractor_adapter.tests.fakes.adapter_test_wiring import (
    EXTERNAL_WORK_TEST_RUN_ID as _RUN_ID,
    EXTERNAL_WORK_TEST_TASK_ID as _TASK_ID,
    allow_adapter,
    bound_external_work_test_execution,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import (
    ExternalTaskCorrelation,
    ExternalWorkCreateRequest,
    QuoteAcceptanceEvidence,
    QuoteAcceptanceScopeIdentity,
    quote_acceptance_side_effect_scope_digest,
)
from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.validation import validate_content_digest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MUTATION = (MeaningfulSideEffectKind.MUTATION,)
_COMMITMENT_MUTATION = (
    MeaningfulSideEffectKind.COMMITMENT,
    MeaningfulSideEffectKind.MUTATION,
)
_DIGEST_1 = "sha256:" + ("ab" * 32)
_DIGEST_2 = "sha256:" + ("cd" * 32)
_IDEM_KEY = "idem-shared-key"
_PRINCIPAL = "principal-ew-1"
_QUOTE_ID = "quote-1"


class _CapturingEvaluator:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision
        self.requests: list[MeaningfulSideEffectRequest] = []

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        self.requests.append(request)
        return self._decision


@pytest.fixture(autouse=True)
def _bound_scope_provenance_execution() -> None:
    with bound_external_work_test_execution():
        yield


def _adapter(decision: PolicyDecision) -> tuple[ExternalWorkAdapter, _CapturingEvaluator]:
    evaluator = _CapturingEvaluator(decision)
    adapter, _ = allow_adapter(
        DeterministicExternalWorkFake(),
        policy=evaluator,
        principal_id=_PRINCIPAL,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        active_task_id=_TASK_ID,
    )
    return adapter, evaluator


def _create_request(*, scope_digest: str) -> ExternalWorkCreateRequest:
    return ExternalWorkCreateRequest(
        provider_id="provider-1",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        requested_capability="external_contractor.adapt",
        scope_description="review",
        scope_digest=scope_digest,
        idempotency_key=_IDEM_KEY,
        workspace_ref="workspace-a",
    )


def _correlation() -> ExternalTaskCorrelation:
    return ExternalTaskCorrelation(
        provider_id="provider-1",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        external_task_id="ext-task-1",
        idempotency_key="corr-idem-1",
        correlation_id="corr-1",
    )


def _acceptance(
    *,
    scope_digest: str,
    quote_version: int = 1,
    quote_id: str = _QUOTE_ID,
) -> QuoteAcceptanceEvidence:
    return QuoteAcceptanceEvidence(
        acceptance_id=f"acc-{quote_version}",
        quote_id=quote_id,
        quote_version=quote_version,
        scope_digest=scope_digest,
        actor=ActorIdentity(
            kind=ActorKind.USER,
            actor_id=_PRINCIPAL,
            tenant_id="tenant-a",
        ),
        accepted_at="2026-08-18T12:00:00+00:00",
    )


def _canonical_digest(
    *,
    scope_digest: str,
    quote_version: int = 1,
    quote_id: str = _QUOTE_ID,
) -> str:
    return quote_acceptance_side_effect_scope_digest(
        _acceptance(
            scope_digest=scope_digest,
            quote_version=quote_version,
            quote_id=quote_id,
        )
    )


def test_create_binds_idempotency_key_and_scope_digest() -> None:
    adapter, evaluator = _adapter(
        PolicyDecision(action=PolicyAction.ALLOW, reason="allow")
    )
    request = _create_request(scope_digest=_DIGEST_1)
    result = adapter.create_and_map(
        request,
        enrich=False,
        principal_id=_PRINCIPAL,
        tenant_id="tenant-a",
    )
    assert result.used is True
    assert evaluator.requests
    captured = evaluator.requests[0]
    assert captured.action == ACTION_CREATE_EXTERNAL_WORK
    assert captured.side_effect_scope_id == _IDEM_KEY
    assert captured.side_effect_scope_digest == _DIGEST_1


def test_same_idempotency_key_different_scope_digest_are_distinct() -> None:
    adapter, evaluator = _adapter(
        PolicyDecision(action=PolicyAction.ALLOW, reason="allow")
    )
    for digest in (_DIGEST_1, _DIGEST_2):
        adapter.create_and_map(
            _create_request(scope_digest=digest),
            enrich=False,
            principal_id=_PRINCIPAL,
            tenant_id="tenant-a",
        )
    assert len(evaluator.requests) == 2
    assert evaluator.requests[0].side_effect_scope_id == evaluator.requests[1].side_effect_scope_id
    assert evaluator.requests[0].side_effect_scope_digest != evaluator.requests[1].side_effect_scope_digest


def test_quote_acceptance_binds_canonical_scope_digest() -> None:
    adapter, evaluator = _adapter(
        PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
    )
    acceptance = _acceptance(scope_digest=_DIGEST_1, quote_version=1)
    expected = quote_acceptance_side_effect_scope_digest(acceptance)
    result = adapter.forward_quote_acceptance(
        _correlation(),
        acceptance,
        idempotency_key=_IDEM_KEY,
        principal_id=_PRINCIPAL,
        enrich=False,
        workspace_id="workspace-a",
    )
    assert evaluator.requests
    captured = evaluator.requests[0]
    assert captured.side_effect_scope_digest == expected
    assert captured.side_effect_scope_digest != acceptance.scope_digest
    assert result.continuation is not None
    assert result.continuation.side_effect_scope_id == _IDEM_KEY
    assert result.continuation.side_effect_scope_digest == expected


def test_same_quote_idempotency_and_scope_digest_different_versions_are_distinct() -> None:
    digest_v1 = _canonical_digest(scope_digest=_DIGEST_1, quote_version=1)
    digest_v2 = _canonical_digest(scope_digest=_DIGEST_1, quote_version=2)
    assert digest_v1 != digest_v2


def test_same_quote_version_and_idempotency_different_scope_digests_are_distinct() -> None:
    digest_d1 = _canonical_digest(scope_digest=_DIGEST_1, quote_version=1)
    digest_d2 = _canonical_digest(scope_digest=_DIGEST_2, quote_version=1)
    assert digest_d1 != digest_d2


def test_different_quote_ids_produce_distinct_scope_digests() -> None:
    digest_q1 = _canonical_digest(scope_digest=_DIGEST_1, quote_id="quote-a")
    digest_q2 = _canonical_digest(scope_digest=_DIGEST_1, quote_id="quote-b")
    assert digest_q1 != digest_q2


def test_quote_scope_digest_ignores_spoofed_dynamic_context() -> None:
    acceptance = _acceptance(scope_digest=_DIGEST_1, quote_version=1, quote_id=_QUOTE_ID)
    expected = QuoteAcceptanceScopeIdentity.from_quote_acceptance_evidence(
        acceptance
    ).compute_side_effect_scope_digest()
    adapter, evaluator = _adapter(
        PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
    )
    adapter.forward_quote_acceptance(
        _correlation(),
        acceptance,
        idempotency_key=_IDEM_KEY,
        principal_id=_PRINCIPAL,
        enrich=False,
        workspace_id="workspace-a",
    )
    assert evaluator.requests[0].side_effect_scope_digest == expected
    assert expected != quote_acceptance_side_effect_scope_digest(
        _acceptance(
            scope_digest="sha256:" + ("ff" * 32),
            quote_version=999,
            quote_id="EVIL",
        )
    )


def test_quote_scopes_with_same_idempotency_key_remain_distinct() -> None:
    adapter, evaluator = _adapter(
        PolicyDecision(action=PolicyAction.ALLOW, reason="allow")
    )
    for version, digest in ((1, _DIGEST_1), (2, _DIGEST_2)):
        acceptance = _acceptance(scope_digest=digest, quote_version=version)
        adapter.forward_quote_acceptance(
            _correlation(),
            acceptance,
            idempotency_key=_IDEM_KEY,
            principal_id=_PRINCIPAL,
            enrich=False,
            workspace_id="workspace-a",
        )
    digests = {req.side_effect_scope_digest for req in evaluator.requests}
    assert len(digests) == 2


def test_cancel_uses_idempotency_key_without_fabricated_digest() -> None:
    adapter, evaluator = _adapter(
        PolicyDecision(action=PolicyAction.ALLOW, reason="allow")
    )
    cancel_idem = "cancel-proposal-1"
    adapter.cancel_and_map(
        _correlation(),
        idempotency_key=cancel_idem,
        principal_id=_PRINCIPAL,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        enrich=False,
    )
    captured = evaluator.requests[0]
    assert captured.action == ACTION_CANCEL_EXTERNAL_WORK
    assert captured.side_effect_scope_id == cancel_idem
    assert captured.side_effect_scope_digest is None


def test_malformed_side_effect_scope_digest_rejected_by_generic_contract() -> None:
    with pytest.raises(ValueError, match="sha256"):
        validate_content_digest("md5:abc")
    with pytest.raises(ValueError, match="sha256"):
        MeaningfulSideEffectRequest(
            action=ACTION_ACCEPT_QUOTE,
            kinds=_COMMITMENT_MUTATION,
            side_effect_scope_id=_IDEM_KEY,
            side_effect_scope_digest="sha256:" + ("AB" * 32),
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            principal_id=_PRINCIPAL,
        )
