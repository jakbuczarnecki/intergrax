# © Artur Czarnecki. All rights reserved.

"""Fixture-driven LAB planner LLM for offline proof runs and deterministic investigation."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    h1_initially_plausible,
    observations_from_evidence_nodes,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimProposal,
    CompletionIntent,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    DIAGNOSIS_KIND,
    INCIDENT_EVIDENCE_IDS,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    LINE_ID,
    STATION_ID,
    TimeWindowLabel,
)

LAB_PLANNER_ENV = "SCENARIO_AI_INCIDENT_LAB_PLANNER"

_DEFAULT_INITIAL_SEQUENCE: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
)
_DEFAULT_REVISION_SEQUENCE: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_TELEMETRY_READ,
)


def lab_planner_enabled() -> bool:
    import os

    return os.environ.get(LAB_PLANNER_ENV, "").strip().lower() in {"1", "true", "yes"}


def _tool_args(tool_id: str, *, station_id: str = STATION_ID) -> dict[str, str]:
    if tool_id in {TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ}:
        return {"line_id": LINE_ID, "window": TimeWindowLabel.INCIDENT}
    if tool_id in {TOOL_STAFFING_SCHEDULE_READ, TOOL_STAFFING_ATTENDANCE_READ}:
        return {"line_id": LINE_ID, "shift_id": "shift_b", "window": TimeWindowLabel.INCIDENT}
    if tool_id == TOOL_COMPARISON_READ:
        return {
            "reference_line_id": LINE_ID,
            "comparison_line_id": "line3",
            "window": TimeWindowLabel.COMPARISON,
        }
    if tool_id == TOOL_TELEMETRY_READ:
        return {"station_id": station_id, "window": TimeWindowLabel.INCIDENT}
    raise ValueError(f"unsupported tool id: {tool_id}")


def _decision_note(*basis_ids: str, purpose: str) -> str:
    basis = ",".join(basis_ids)
    return f"EVIDENCE_BASIS: {basis}\nPURPOSE: {purpose}"


def _evidence_ids_from_messages(messages: Sequence[ChatMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        content = message.content or ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                candidate = stripped[2:].strip()
                if " availability=" in candidate:
                    candidate = candidate.split(" availability=", 1)[0].strip()
                if candidate.startswith("evidence."):
                    ids.add(candidate)
    return ids


def _telemetry_unavailable_from_messages(messages: Sequence[ChatMessage]) -> bool:
    for message in messages:
        content = message.content or ""
        if "availability=unavailable" in content:
            return True
    return False


def _telemetry_available_from_messages(messages: Sequence[ChatMessage]) -> bool:
    for message in messages:
        content = message.content or ""
        if "availability=available" in content:
            return True
    return False


def build_fixture_reasoning_proposal(
    *,
    evidence_ids: set[str],
    is_revision: bool,
    telemetry_unavailable: bool = False,
    telemetry_available: bool = False,
) -> IncidentReasoningProposal:
    workload = str(WORKLOAD_EVIDENCE_ID)
    throughput = str(THROUGHPUT_EVIDENCE_ID)
    schedule = str(STAFFING_PRELIMINARY_EVIDENCE_ID)
    attendance = str(STAFFING_ATTENDANCE_EVIDENCE_ID)
    comparison = str(COMPARISON_EVIDENCE_ID)
    telemetry = str(TELEMETRY_EVIDENCE_ID)

    nodes: list[dict[str, object]] = []
    for evidence_id in evidence_ids:
        payload: dict[str, object] = {}
        if evidence_id == telemetry and telemetry_unavailable:
            payload = {"availability": "unavailable", "admissible": True}
        elif evidence_id == telemetry:
            continue
        nodes.append({"evidence_id": evidence_id, "payload": payload})

    observations = None
    try:
        observations = observations_from_evidence_nodes(tuple(nodes), INCIDENT_EVIDENCE_IDS)
    except (ValueError, TypeError, KeyError):
        observations = None

    all_six = {workload, throughput, schedule, attendance, comparison, telemetry}.issubset(evidence_ids)

    if all_six and telemetry_unavailable:
        return IncidentReasoningProposal(
            hypotheses=(
                HypothesisProposal(
                    hypothesis_id="H1",
                    disposition=HypothesisDisposition.SUPERSEDED,
                    summary="Overload weakened by comparison evidence.",
                    supporting_evidence_ids=(workload, throughput),
                    contradicting_evidence_ids=(comparison,),
                ),
                HypothesisProposal(
                    hypothesis_id="H2",
                    disposition=HypothesisDisposition.REJECTED,
                    summary="Understaffing rejected by attendance confirmation.",
                    supporting_evidence_ids=(schedule,),
                    contradicting_evidence_ids=(attendance,),
                ),
                HypothesisProposal(
                    hypothesis_id="H3",
                    disposition=HypothesisDisposition.INSUFFICIENT_EVIDENCE,
                    summary="Equipment degradation cannot be confirmed without decisive telemetry.",
                ),
            ),
            preferred_hypothesis_id="H3",
            uncertainty_class="decisive_gap",
            information_gaps=("decisive station telemetry unavailable for incident window",),
            claim_proposals=(
                ClaimProposal(
                    hypothesis_id="H1",
                    statement=(
                        "Production workload on Line 4 increased during the incident window "
                        "while throughput declined — overload hypothesis H1."
                    ),
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(workload, throughput),
                ),
                ClaimProposal(
                    hypothesis_id="H2",
                    statement=(
                        "Understaffing on the affected shift is not supported as initiating cause: "
                        "preliminary roster export conflicts with confirmed attendance for the "
                        "incident window — hypothesis H2 rejected."
                    ),
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(schedule,),
                    contradicting_evidence_ids=(attendance,),
                ),
                ClaimProposal(
                    hypothesis_id="H3",
                    statement=(
                        "Equipment-process degradation hypothesis H3 cannot be accepted: "
                        "decisive station telemetry for the incident window is unavailable."
                    ),
                    claim_kind=str(DIAGNOSIS_KIND),
                ),
            ),
            completion_intent=CompletionIntent.UNRESOLVED,
            action_objective="declare epistemic unresolved due to missing decisive telemetry",
            unresolved_reason="Decisive station telemetry for the incident window is unavailable.",
        )

    h3_supported = all_six and (telemetry_available or not telemetry_unavailable)

    if h3_supported:
        return IncidentReasoningProposal(
            hypotheses=(
                HypothesisProposal(
                    hypothesis_id="H1",
                    disposition=HypothesisDisposition.SUPERSEDED,
                    summary="Overload correlation weakened by peer-line comparison evidence.",
                    supporting_evidence_ids=(workload, throughput),
                    contradicting_evidence_ids=(comparison,),
                ),
                HypothesisProposal(
                    hypothesis_id="H2",
                    disposition=HypothesisDisposition.REJECTED,
                    summary="Understaffing rejected after attendance confirmation.",
                    supporting_evidence_ids=(schedule,),
                    contradicting_evidence_ids=(attendance,),
                ),
                HypothesisProposal(
                    hypothesis_id="H3",
                    disposition=HypothesisDisposition.SUPPORTED,
                    summary="Intermittent equipment degradation supported by telemetry and comparison.",
                    supporting_evidence_ids=(workload, throughput, comparison, telemetry),
                ),
            ),
            preferred_hypothesis_id="H3",
            uncertainty_class="bounded",
            information_gaps=(),
            claim_proposals=(
                ClaimProposal(
                    hypothesis_id="H1",
                    statement=(
                        "Production workload on Line 4 increased during the incident window "
                        "while throughput declined — overload hypothesis H1."
                    ),
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(workload, throughput),
                ),
                ClaimProposal(
                    hypothesis_id="H2",
                    statement=(
                        "Understaffing on the affected shift is not supported as initiating cause: "
                        "preliminary roster export conflicts with confirmed attendance for the "
                        "incident window — hypothesis H2 rejected."
                    ),
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(schedule,),
                    contradicting_evidence_ids=(attendance,),
                ),
                ClaimProposal(
                    hypothesis_id="H3",
                    statement=(
                        "Intermittent station signal degradation on the complex-assembly step "
                        "is the best-supported initiating cause; comparison evidence shows "
                        "similar elevated workload elsewhere without comparable degradation; "
                        "workload growth plausibly amplified impact — bounded H3 diagnosis."
                    ),
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(workload, throughput, comparison, telemetry),
                    replaces_prior_claim=True,
                ),
            ),
            completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
            action_objective="propose bounded H3 diagnosis supported by distinguishing evidence",
        )

    h1_plausible = (
        observations is not None
        and h1_initially_plausible(observations.workload, observations.throughput)
    )
    initial_support = tuple(eid for eid in (workload, throughput) if eid in evidence_ids)
    return IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H1",
                disposition=HypothesisDisposition.PLAUSIBLE if h1_plausible else HypothesisDisposition.REJECTED,
                summary="Workload-throughput correlation is plausible but not yet causal.",
                supporting_evidence_ids=initial_support,
            ),
            HypothesisProposal(
                hypothesis_id="H2",
                disposition=HypothesisDisposition.PENDING,
                summary="Staffing shortage requires attendance confirmation.",
                supporting_evidence_ids=tuple(eid for eid in (schedule,) if eid in evidence_ids),
            ),
            HypothesisProposal(
                hypothesis_id="H3",
                disposition=HypothesisDisposition.PENDING,
                summary="Equipment degradation requires comparison and telemetry evidence.",
            ),
        ),
        preferred_hypothesis_id="H1",
        uncertainty_class="high",
        information_gaps=("comparison line evidence", "staffing attendance confirmation", "station telemetry"),
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H1",
                statement=(
                    "Sustained production overload from workload growth caused Line 4 "
                    "target attainment degradation — hypothesis H1."
                ),
                claim_kind=str(DIAGNOSIS_KIND),
                supporting_evidence_ids=initial_support,
            ),
        )
        if initial_support
        else (
            ClaimProposal(
                hypothesis_id="H1",
                statement="Incident investigation pending operational evidence gathering.",
                claim_kind=str(DIAGNOSIS_KIND),
            ),
        ),
        completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        action_objective="evaluate overload hypothesis pending distinguishing evidence",
        follow_up_objective="gather comparison, attendance, and telemetry evidence" if not is_revision else None,
    )


class FixtureDrivenIncidentInvestigationLLM(LLMAdapter):
    """Fixture-world planner that drives tool order and reasoning for LAB proof runs."""

    provider = "fixture_lab"
    model = "incident_fixture_planner"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def __init__(
        self,
        *,
        initial_sequence: Sequence[str] | None = None,
        revision_sequence: Sequence[str] | None = None,
        station_id: str = STATION_ID,
    ) -> None:
        super().__init__()
        self._initial_sequence = tuple(initial_sequence or _DEFAULT_INITIAL_SEQUENCE)
        self._revision_sequence = tuple(revision_sequence or _DEFAULT_REVISION_SEQUENCE)
        self._station_id = station_id
        self._round_by_phase: dict[str, int] = {"initial": 0, "revision": 0}
        self._prior_tool_call_ids: list[str] = []

    def generate_messages(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, kwargs
        return build_adapter_response(content="")

    def supports_tools(self) -> bool:
        return True

    def _detect_phase(self, messages: Sequence[ChatMessage]) -> str:
        for message in messages:
            if message.role == "system" and "Investigation phase: revision" in (message.content or ""):
                return "revision"
        return "initial"

    def _active_sequence(self, phase: str) -> tuple[str, ...]:
        return self._revision_sequence if phase == "revision" else self._initial_sequence

    def _detect_reasoning_phase(self, messages: Sequence[ChatMessage]) -> str:
        for message in messages:
            if message.role == "system" and "Investigation phase: revision" in (message.content or ""):
                return "revision"
        return "initial"

    def generate_structured(self, messages, output_model, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        if output_model is not IncidentReasoningProposal:
            raise NotImplementedError(f"unsupported structured model: {output_model}")
        evidence_ids = _evidence_ids_from_messages(messages)
        proposal = build_fixture_reasoning_proposal(
            evidence_ids=evidence_ids,
            is_revision=self._detect_reasoning_phase(messages) == "revision",
            telemetry_unavailable=_telemetry_unavailable_from_messages(messages),
            telemetry_available=_telemetry_available_from_messages(messages),
        )
        return LLMStructuredResult(parsed=proposal, response=build_adapter_response(content=""))

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        phase = self._detect_phase(messages)
        has_tool_messages = any(message.role == "tool" for message in messages)
        if not has_tool_messages:
            self._prior_tool_call_ids = []
            self._round_by_phase[phase] = 0
        sequence = self._active_sequence(phase)
        round_index = self._round_by_phase[phase]
        if round_index >= len(sequence):
            return LLMAdapterResponse(content="evidence gathering complete", tool_calls=())

        tool_id = sequence[round_index]
        call_id = f"tc-{phase}-{round_index + 1}"
        self._round_by_phase[phase] = round_index + 1
        purpose = f"gather {tool_id.split('.')[-1]} evidence for incident investigation"
        if round_index == 0:
            content = _decision_note(purpose=purpose)
        else:
            basis = self._prior_tool_call_ids[-1]
            content = _decision_note(basis, purpose=purpose)
        self._prior_tool_call_ids.append(call_id)
        return LLMAdapterResponse(
            content=content,
            tool_calls=(
                LLMToolCall.from_openai_shape(
                    call_id=call_id,
                    name=tool_id,
                    arguments=_tool_args(tool_id, station_id=self._station_id),
                ),
            ),
        )
