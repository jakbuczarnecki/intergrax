# © Artur Czarnecki. All rights reserved.

"""UCA-6A — CodeCraft gap synthesis acquisition strategy and seam."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
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
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.codecraft.gap_synthesis import (
    CodeCraftGapSynthesisOutcome,
    CodeCraftGapSynthesisPort,
    CodeCraftGapSynthesisRequest,
    CodeCraftGapSynthesisResult,
)
from intergrax.runtime.codecraft.acquisition import (
    CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID,
    CodeCraftGapCapabilityAcquisitionStrategy,
)

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 21, 8, 0, tzinfo=UTC)


def _gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-uca6",
        discovery_correlation_id="canonical-discovery-corr",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def _request(
    gap: CapabilityGap,
    *,
    kinds: tuple[CapabilityKind, ...] = (CapabilityKind.TOOL,),
    intent: str = "synthesize ephemeral tool",
) -> CapabilityAcquisitionRequest:
    need = CapabilityNeed(
        need_id=gap.need_id,
        kinds=kinds,
        intent_summary=intent,
    )
    return CapabilityAcquisitionRequest(
        request_id=derive_capability_acquisition_request_id(
            gap_id=gap.gap_id,
            request_nonce="nonce-uca6",
        ),
        request_nonce="nonce-uca6",
        capability_gap=gap,
        capability_need=need,
        correlation_id="corr-uca6",
        causation_id="cause-uca6",
        requested_at=_CREATED,
    )


class _RecordingSynthesisPort(CodeCraftGapSynthesisPort):
    def __init__(self, result: CodeCraftGapSynthesisResult) -> None:
        self.result = result
        self.calls: list[CodeCraftGapSynthesisRequest] = []

    def synthesize_from_gap(
        self,
        request: CodeCraftGapSynthesisRequest,
    ) -> CodeCraftGapSynthesisResult:
        self.calls.append(request)
        return self.result


class _UnavailablePort(CodeCraftGapSynthesisPort):
    def synthesize_from_gap(
        self,
        request: CodeCraftGapSynthesisRequest,
    ) -> CodeCraftGapSynthesisResult:
        return CodeCraftGapSynthesisResult(
            operation_id=request.operation_id,
            gap_id=request.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.UNAVAILABLE,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            reason_detail="codecraft down",
        )


class _ExplodingPort(CodeCraftGapSynthesisPort):
    def synthesize_from_gap(
        self,
        request: CodeCraftGapSynthesisRequest,
    ) -> CodeCraftGapSynthesisResult:
        raise RuntimeError("unexpected programming error")


def test_strategy_supports_is_side_effect_free() -> None:
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id="operation-id",
            gap_id="gap-id",
            outcome=CodeCraftGapSynthesisOutcome.FAILED,
        ),
    )
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(port)
    req = _request(_gap())
    assert strategy.supports(req) is True
    assert port.calls == []


def test_supported_kind_tool_true_agent_false() -> None:
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id="x",
            gap_id="y",
            outcome=CodeCraftGapSynthesisOutcome.FAILED,
        ),
    )
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(port)
    assert strategy.supports(_request(_gap(), kinds=(CapabilityKind.TOOL,))) is True
    assert strategy.supports(_request(_gap(), kinds=(CapabilityKind.AGENT,))) is False


def test_acquire_maps_request_and_succeeds_with_artifact() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:craft-1",
            codecraft_operation_correlation_id="craft-1",
            correlation_id=req.correlation_id,
            causation_id=req.causation_id,
        ),
    )
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(port)
    result = strategy.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert result.strategy_id == CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID
    assert result.evidence is not None
    assert result.evidence.artifact_reference == "codecraft:artifact:craft-1"
    mapped = port.calls[0]
    assert mapped.operation_id == req.request_id
    assert mapped.gap_id == gap.gap_id
    assert mapped.canonical_discovery_correlation_id == gap.discovery_correlation_id
    assert mapped.synthesis_goal == req.capability_need.intent_summary
    assert mapped.correlation_id == req.correlation_id
    assert mapped.causation_id == req.causation_id


def test_succeeded_result_is_qualification_request_compatible() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:craft-q",
            correlation_id=req.correlation_id,
            causation_id=req.causation_id,
        ),
    )
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(port)
    acquisition = strategy.acquire(req)
    qual = CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=req.request_id,
            qualification_nonce="q-nonce",
        ),
        qualification_nonce="q-nonce",
        acquisition_request_id=req.request_id,
        gap_id=gap.gap_id,
        strategy_id=CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID,
        acquisition_result=acquisition,
        correlation_id=req.correlation_id,
        causation_id=req.causation_id,
        requested_at=_CREATED,
    )
    assert qual.acquisition_result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED


def test_outcome_mappings() -> None:
    gap = _gap()
    req = _request(gap)

    def _result(outcome: CodeCraftGapSynthesisOutcome) -> CodeCraftGapSynthesisResult:
        return CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=outcome,
            correlation_id=req.correlation_id,
            causation_id=req.causation_id,
        )

    assert (
        CodeCraftGapCapabilityAcquisitionStrategy(
            _RecordingSynthesisPort(_result(CodeCraftGapSynthesisOutcome.BLOCKED)),
        )
        .acquire(req)
        .outcome
        is CapabilityAcquisitionOutcome.BLOCKED
    )
    assert (
        CodeCraftGapCapabilityAcquisitionStrategy(
            _RecordingSynthesisPort(
                _result(CodeCraftGapSynthesisOutcome.REQUIRES_HITL),
            ),
        )
        .acquire(req)
        .outcome
        is CapabilityAcquisitionOutcome.REQUIRES_HITL
    )
    assert (
        CodeCraftGapCapabilityAcquisitionStrategy(
            _RecordingSynthesisPort(
                _result(CodeCraftGapSynthesisOutcome.NOT_SUPPORTED),
            ),
        )
        .acquire(req)
        .outcome
        is CapabilityAcquisitionOutcome.NOT_SUPPORTED
    )
    assert (
        CodeCraftGapCapabilityAcquisitionStrategy(
            _RecordingSynthesisPort(_result(CodeCraftGapSynthesisOutcome.FAILED)),
        )
        .acquire(req)
        .outcome
        is CapabilityAcquisitionOutcome.FAILED
    )


def test_codecraft_unavailable_maps_via_service() -> None:
    gap = _gap()
    req = _request(gap)
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(_UnavailablePort())
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    result = service.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.UNAVAILABLE
    assert result.strategy_id == CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID


def test_wrong_operation_id_fails_integrity() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id="wrong-id",
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:x",
            correlation_id=req.correlation_id,
            causation_id=req.causation_id,
        ),
    )
    result = CodeCraftGapCapabilityAcquisitionStrategy(port).acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.FAILED
    assert result.reason_code is CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT


def test_wrong_gap_id_fails_integrity() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id="wrong-gap",
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:x",
            correlation_id=req.correlation_id,
            causation_id=req.causation_id,
        ),
    )
    result = CodeCraftGapCapabilityAcquisitionStrategy(port).acquire(req)
    assert result.reason_code is CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT


def test_correlation_mismatch_fails_integrity() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:x",
            correlation_id="wrong-corr",
            causation_id=req.causation_id,
        ),
    )
    result = CodeCraftGapCapabilityAcquisitionStrategy(port).acquire(req)
    assert result.reason_code is CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT


def test_causation_mismatch_fails_integrity() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:x",
            correlation_id=req.correlation_id,
            causation_id="wrong-cause",
        ),
    )
    result = CodeCraftGapCapabilityAcquisitionStrategy(port).acquire(req)
    assert result.reason_code is CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT


def test_missing_intent_summary_is_invalid_request() -> None:
    gap = _gap()
    need = CapabilityNeed(need_id=gap.need_id, kinds=(CapabilityKind.TOOL,))
    req = CapabilityAcquisitionRequest(
        request_id=derive_capability_acquisition_request_id(
            gap_id=gap.gap_id,
            request_nonce="nonce-no-intent",
        ),
        request_nonce="nonce-no-intent",
        capability_gap=gap,
        capability_need=need,
        requested_at=_CREATED,
    )
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:x",
        ),
    )
    result = CodeCraftGapCapabilityAcquisitionStrategy(port).acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.FAILED
    assert result.reason_code is CapabilityAcquisitionReasonCode.INVALID_REQUEST
    assert port.calls == []


def test_programming_error_not_masked() -> None:
    gap = _gap()
    req = _request(gap)
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(_ExplodingPort())
    with pytest.raises(RuntimeError, match="unexpected programming error"):
        strategy.acquire(req)


def test_pluginability_via_acquisition_service() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingSynthesisPort(
        CodeCraftGapSynthesisResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
            artifact_reference="codecraft:artifact:plug",
            correlation_id=req.correlation_id,
            causation_id=req.causation_id,
        ),
    )
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(port)
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    result = service.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert len(port.calls) == 1
