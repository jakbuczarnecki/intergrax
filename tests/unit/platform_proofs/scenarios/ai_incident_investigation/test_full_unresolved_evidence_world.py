# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.contracts.evidence_claims import (
    ChallengeResolution,
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
)
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    ObservedTelemetry,
    derive_hypothesis_dispositions,
    parse_telemetry_payload,
    telemetry_is_unavailable,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import (
    build_forged_h3_claim_set,
    evaluate_scenario_run,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    ScenarioVariant,
    TelemetryAvailability,
    TelemetryUnavailabilityReason,
    build_unresolved_fixture,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_UNRESOLVED,
    DIAGNOSIS_KIND,
    H2_CLAIM_ID,
    H3_CLAIM_ID,
    INCIDENT_EVIDENCE_IDS,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    build_runtime_bundle,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    claim_id_for_hypothesis,
    parse_claim_hypothesis_bindings,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    H1_FALLBACK_ERROR,
    H2_FALLBACK_ERROR,
    TELEMETRY_CONTENT_ERROR,
    validate_claim_set_against_observations,
)
from tests.unit.platform_proofs.scenarios.ai_incident_investigation.test_evidence_driven_reasoning import (
    _happy_observations,
)

pytestmark = pytest.mark.unit


def _validation_payload(
    result,
    *,
    claim_set: dict | EvidenceClaimSet | None = None,
    evidence_nodes: list | None = None,
    active_hypothesis: str = "H3",
    completion_mode: str = COMPLETION_UNRESOLVED,
    extra_bindings: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    bindings = list(result.claim_hypothesis_bindings)
    if extra_bindings:
        replaced = {item["hypothesis_id"] for item in extra_bindings}
        bindings = [item for item in bindings if item.get("hypothesis_id") not in replaced]
        bindings.extend(extra_bindings)
    if isinstance(claim_set, EvidenceClaimSet):
        claim_payload = claim_set.model_dump(mode="json")
    else:
        claim_payload = claim_set or result.claim_set
    return {
        "claim_set": claim_payload,
        "claim_hypothesis_bindings": bindings,
        "evidence_nodes": evidence_nodes if evidence_nodes is not None else list(result.evidence_nodes),
        "active_hypothesis": active_hypothesis,
        "completion_mode": completion_mode,
    }


def test_unresolved_fixture_telemetry_unavailable() -> None:
    fixture = build_unresolved_fixture()
    assert fixture.variant is ScenarioVariant.UNRESOLVED
    assert fixture.telemetry.availability is TelemetryAvailability.UNAVAILABLE
    assert (
        fixture.telemetry.unavailability_reason
        is TelemetryUnavailabilityReason.NO_OBSERVATION_FOR_WINDOW
    )
    assert fixture.telemetry.signal_state is None
    assert fixture.private_truth.expected_hypothesis.value == "H3"


def test_parse_unavailable_telemetry_no_fabricated_measurements() -> None:
    fixture = build_unresolved_fixture()
    payload = {
        "availability": TelemetryAvailability.UNAVAILABLE.value,
        "admissible": True,
        "unavailability_reason": TelemetryUnavailabilityReason.NO_OBSERVATION_FOR_WINDOW.value,
        "observed_from": fixture.telemetry.window.observed_from.isoformat(),
        "observed_to": fixture.telemetry.window.observed_to.isoformat(),
    }
    parsed = parse_telemetry_payload(payload)
    assert telemetry_is_unavailable(parsed)
    assert parsed.signal_state is None
    assert parsed.complex_assembly_throughput_pct is None
    assert parsed.baseline_throughput_pct is None


def test_unavailable_telemetry_yields_h3_insufficient() -> None:
    observations = replace(
        _happy_observations(),
        telemetry=ObservedTelemetry(
            availability=TelemetryAvailability.UNAVAILABLE,
            admissible=True,
            unavailability_reason=TelemetryUnavailabilityReason.NO_OBSERVATION_FOR_WINDOW.value,
        ),
    )
    assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    assert assessment.h1.disposition is ClaimResolution.SUPERSEDED
    assert assessment.h2.disposition is ClaimResolution.REJECTED
    assert assessment.h3.disposition is ClaimResolution.INSUFFICIENT_EVIDENCE


def test_same_id_available_vs_unresolved_outcomes() -> None:
    base = _happy_observations()
    available = derive_hypothesis_dispositions(base, INCIDENT_EVIDENCE_IDS)
    unavailable = derive_hypothesis_dispositions(
        replace(
            base,
            telemetry=ObservedTelemetry(
                availability=TelemetryAvailability.UNAVAILABLE,
                admissible=True,
                unavailability_reason=TelemetryUnavailabilityReason.NO_OBSERVATION_FOR_WINDOW.value,
            ),
        ),
        INCIDENT_EVIDENCE_IDS,
    )
    assert available.h3.disposition is ClaimResolution.SUPPORTED
    assert unavailable.h3.disposition is ClaimResolution.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_full_unresolved_evidence_world_passes_evaluator() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_UNRESOLVED
    assert result.critic_verdict_passed
    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed, evaluation.failures


@pytest.mark.asyncio
async def test_resolved_regression_still_passes() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed, evaluation.failures


@pytest.mark.asyncio
async def test_unresolved_challenge_remains_open() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    assert result.evidence_challenge is not None
    assert result.evidence_challenge.resolution is ChallengeResolution.OPEN
    assert TELEMETRY_EVIDENCE_ID not in result.evidence_challenge.evidence_ids


@pytest.mark.asyncio
async def test_unresolved_forged_h3_supported_fails_critic() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    forged = build_forged_h3_claim_set(result)
    forged_h3 = next(
        claim for claim in forged.claims if claim.resolution is ClaimResolution.SUPPORTED
    )
    validation = validate_claim_set_against_observations(
        forged,
        _validation_payload(
            result,
            claim_set=forged,
            completion_mode="supported_diagnosis",
            extra_bindings=[{"claim_id": str(forged_h3.claim_id), "hypothesis_id": "H3"}],
        ),
    )
    assert not validation.valid
    assert TELEMETRY_CONTENT_ERROR in validation.errors


@pytest.mark.asyncio
async def test_unresolved_forged_h1_fallback_fails() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    forged_h1 = EvidenceBackedClaim(
        claim_id=INITIAL_CLAIM_ID,
        statement="Forged H1 supported despite comparison weakening.",
        claim_kind=DIAGNOSIS_KIND,
        supporting_evidence_ids=tuple(
            eid for c in claim_set.claims for eid in c.supporting_evidence_ids
        )[:2],
        resolution=ClaimResolution.SUPPORTED,
    )
    other = [c for c in claim_set.claims if c.claim_id != INITIAL_CLAIM_ID]
    forged = EvidenceClaimSet(claims=(forged_h1, *other), challenges=claim_set.challenges)
    validation = validate_claim_set_against_observations(
        forged,
        _validation_payload(
            result,
            claim_set=forged,
            active_hypothesis="H1",
            completion_mode="supported_diagnosis",
            extra_bindings=[{"claim_id": str(INITIAL_CLAIM_ID), "hypothesis_id": "H1"}],
        ),
    )
    assert not validation.valid
    assert H1_FALLBACK_ERROR in validation.errors


@pytest.mark.asyncio
async def test_unresolved_forged_h2_fallback_fails() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    forged_h2 = EvidenceBackedClaim(
        claim_id=H2_CLAIM_ID,
        statement="Forged H2 supported despite attendance evidence.",
        claim_kind=DIAGNOSIS_KIND,
        supporting_evidence_ids=(),
        resolution=ClaimResolution.SUPPORTED,
    )
    other = [c for c in claim_set.claims if c.claim_id != H2_CLAIM_ID]
    forged = EvidenceClaimSet(claims=(*other, forged_h2), challenges=claim_set.challenges)
    validation = validate_claim_set_against_observations(
        forged,
        _validation_payload(
            result,
            claim_set=forged,
            active_hypothesis="H2",
            completion_mode="supported_diagnosis",
            extra_bindings=[{"claim_id": str(H2_CLAIM_ID), "hypothesis_id": "H2"}],
        ),
    )
    assert not validation.valid
    assert H2_FALLBACK_ERROR in validation.errors


@pytest.mark.asyncio
async def test_unresolved_positive_claim_set_passes_validation() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    validation = validate_claim_set_against_observations(
        claim_set,
        _validation_payload(result),
    )
    assert validation.valid


@pytest.mark.asyncio
async def test_unresolved_telemetry_provider_invoked_once() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    assert result.tool_invocations == 6
    telemetry_nodes = [
        node for node in result.evidence_nodes if node["evidence_id"] == str(TELEMETRY_EVIDENCE_ID)
    ]
    assert len(telemetry_nodes) == 1
    payload = telemetry_nodes[0]["payload"]
    assert isinstance(payload, dict)
    assert payload.get("availability") == TelemetryAvailability.UNAVAILABLE.value


@pytest.mark.asyncio
async def test_mutation_available_resolved_unavailable_unresolved() -> None:
    resolved_bundle = build_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    unresolved_bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    resolved = await execute_resolved_skeleton(resolved_bundle)
    unresolved = await execute_resolved_skeleton(unresolved_bundle)
    assert resolved.outcome == OUTCOME_RESOLVED
    assert unresolved.outcome == OUTCOME_UNRESOLVED
    resolved_claims = EvidenceClaimSet.model_validate(resolved.claim_set)
    unresolved_claims = EvidenceClaimSet.model_validate(unresolved.claim_set)
    resolved_bindings = parse_claim_hypothesis_bindings(resolved.claim_hypothesis_bindings)
    unresolved_bindings = parse_claim_hypothesis_bindings(unresolved.claim_hypothesis_bindings)
    resolved_h3_id = claim_id_for_hypothesis(resolved_bindings, "H3")
    unresolved_h3_id = claim_id_for_hypothesis(unresolved_bindings, "H3")
    assert resolved_h3_id is not None
    assert unresolved_h3_id is not None
    assert any(str(c.claim_id) == resolved_h3_id and c.resolution is ClaimResolution.SUPPORTED for c in resolved_claims.claims)
    assert any(str(c.claim_id) == unresolved_h3_id for c in unresolved_claims.claims)
    assert not any(
        c.resolution is ClaimResolution.SUPPORTED for c in unresolved_claims.claims
    )
