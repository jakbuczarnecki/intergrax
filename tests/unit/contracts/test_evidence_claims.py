# © Artur Czarnecki. All rights reserved.

"""GAP-1A — evidence-backed claim contract tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.evidence_claims import (
    ChallengeDefectFamily,
    ChallengeResolution,
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceChallenge,
    EvidenceClaimId,
    EvidenceClaimSet,
    EvidenceReferenceId,
    mint_evidence_challenge_id,
    mint_evidence_claim_id,
    validate_claim_kind,
    validate_defect_code,
    validate_evidence_challenge_id,
    validate_evidence_claim_id,
    validate_evidence_reference_id,
)

_MODULE_PATH = Path("intergrax/contracts/evidence_claims.py")
_FORBIDDEN_IMPORT_FRAGMENTS = (
    "scripts.proof",
    "platform_proofs",
    "intergrax.applications",
    "applications.",
    "intergrax.runtime.critic",
    "intergrax.runtime.nexus",
)


def _claim_id(suffix: str = "0123456789abcdef0123456789abcdef") -> str:
    return f"eclaim_{suffix}"


def _challenge_id(suffix: str = "0123456789abcdef0123456789abcdef") -> str:
    return f"echlg_{suffix}"


@pytest.mark.unit
@pytest.mark.gate
def test_validate_evidence_claim_id_accepts_valid() -> None:
    claim_id = mint_evidence_claim_id()
    assert validate_evidence_claim_id(claim_id) == claim_id


@pytest.mark.unit
@pytest.mark.gate
def test_validate_evidence_claim_id_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="must start with"):
        validate_evidence_claim_id("claim_bad")
    with pytest.raises(ValueError, match="suffix"):
        validate_evidence_claim_id("eclaim_tooshort")
    with pytest.raises(TypeError):
        validate_evidence_claim_id(123)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_evidence_challenge_id_accepts_valid() -> None:
    challenge_id = mint_evidence_challenge_id()
    assert validate_evidence_challenge_id(challenge_id) == challenge_id


@pytest.mark.unit
@pytest.mark.gate
def test_validate_evidence_challenge_id_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="must start with"):
        validate_evidence_challenge_id("challenge_bad")


@pytest.mark.unit
@pytest.mark.gate
def test_evidence_reference_id_distinct_from_claim_id() -> None:
    claim_id = mint_evidence_claim_id()
    evidence_id = validate_evidence_reference_id("e1")
    assert EvidenceClaimId is not EvidenceReferenceId
    assert validate_evidence_claim_id(claim_id) == claim_id
    assert validate_evidence_reference_id(evidence_id) == evidence_id
    with pytest.raises(ValueError, match="must start with"):
        validate_evidence_claim_id(evidence_id)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_minimal_claim() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id(),
        statement="A bounded claim statement.",
        claim_kind="generic.claim",
    )
    assert claim.resolution is ClaimResolution.PENDING
    assert claim.supporting_evidence_ids == ()
    assert claim.contradicting_evidence_ids == ()


@pytest.mark.unit
@pytest.mark.gate
def test_canonical_evidence_ordering() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id(),
        statement="Ordered evidence refs.",
        claim_kind="generic.claim",
        supporting_evidence_ids=["e3", "e1", "e2"],
    )
    assert claim.supporting_evidence_ids == (
        validate_evidence_reference_id("e1"),
        validate_evidence_reference_id("e2"),
        validate_evidence_reference_id("e3"),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_supporting_evidence_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        EvidenceBackedClaim(
            claim_id=_claim_id(),
            statement="Duplicate support.",
            claim_kind="generic.claim",
            supporting_evidence_ids=["e1", "e1"],
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_contradicting_evidence_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        EvidenceBackedClaim(
            claim_id=_claim_id(),
            statement="Duplicate contradiction.",
            claim_kind="generic.claim",
            contradicting_evidence_ids=["e1", "e1"],
        )


@pytest.mark.unit
@pytest.mark.gate
def test_same_evidence_in_support_and_contradiction_rejected() -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        EvidenceBackedClaim(
            claim_id=_claim_id(),
            statement="Conflicting evidence placement.",
            claim_kind="generic.claim",
            supporting_evidence_ids=["e1"],
            contradicting_evidence_ids=["e1"],
        )


@pytest.mark.unit
@pytest.mark.gate
def test_invalid_claim_kind_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid characters"):
        EvidenceBackedClaim(
            claim_id=_claim_id(),
            statement="Bad kind.",
            claim_kind="bad/kind",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_claim_resolution_enum_validation() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id(),
        statement="Supported claim.",
        claim_kind="generic.claim",
        resolution=ClaimResolution.SUPPORTED,
    )
    assert claim.resolution is ClaimResolution.SUPPORTED


@pytest.mark.unit
@pytest.mark.gate
def test_valid_challenge() -> None:
    claim_id = _claim_id()
    challenge = EvidenceChallenge(
        challenge_id=_challenge_id(),
        claim_id=claim_id,
        defect_family=ChallengeDefectFamily.MISSING_EVIDENCE,
        evidence_ids=["e1"],
        description="Concise externally shareable rationale.",
    )
    assert challenge.resolution is ChallengeResolution.OPEN
    assert challenge.defect_code is None


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_challenge_evidence_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        EvidenceChallenge(
            challenge_id=_challenge_id(),
            claim_id=_claim_id(),
            defect_family=ChallengeDefectFamily.OTHER,
            evidence_ids=["e1", "e1"],
        )


@pytest.mark.unit
@pytest.mark.gate
def test_generic_defect_family_values() -> None:
    assert ChallengeDefectFamily.UNSUPPORTED_INFERENCE.value == "unsupported_inference"
    assert ChallengeDefectFamily.MISSING_EVIDENCE.value == "missing_evidence"


@pytest.mark.unit
@pytest.mark.gate
def test_valid_optional_domain_defect_code() -> None:
    challenge = EvidenceChallenge(
        challenge_id=_challenge_id(),
        claim_id=_claim_id(),
        defect_family=ChallengeDefectFamily.MISSING_EVIDENCE,
        defect_code="domain.missing_financial_evidence",
    )
    assert challenge.defect_code == validate_defect_code("domain.missing_financial_evidence")


@pytest.mark.unit
@pytest.mark.gate
def test_invalid_defect_code_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid characters"):
        EvidenceChallenge(
            challenge_id=_challenge_id(),
            claim_id=_claim_id(),
            defect_family=ChallengeDefectFamily.OTHER,
            defect_code="bad/code",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_no_private_dynamic_metadata_fields() -> None:
    for model in (EvidenceBackedClaim, EvidenceChallenge, EvidenceClaimSet):
        for field in model.model_fields.values():
            annotation = str(field.annotation)
            assert "dict[str, Any]" not in annotation
            assert "Any" not in annotation
        dumped = model.model_fields.keys()
        assert "metadata" not in dumped
        assert "context" not in dumped
        assert "chain_of_thought" not in dumped
        assert "scratchpad" not in dumped


@pytest.mark.unit
@pytest.mark.gate
def test_aggregate_duplicate_claim_id_rejected() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id(),
        statement="First.",
        claim_kind="generic.claim",
    )
    with pytest.raises(ValidationError, match="unique claim_id"):
        EvidenceClaimSet(claims=(claim, claim))


@pytest.mark.unit
@pytest.mark.gate
def test_aggregate_duplicate_challenge_id_rejected() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id(),
        statement="Claim.",
        claim_kind="generic.claim",
    )
    challenge = EvidenceChallenge(
        challenge_id=_challenge_id(),
        claim_id=claim.claim_id,
        defect_family=ChallengeDefectFamily.OTHER,
    )
    with pytest.raises(ValidationError, match="unique challenge_id"):
        EvidenceClaimSet(claims=(claim,), challenges=(challenge, challenge))


@pytest.mark.unit
@pytest.mark.gate
def test_aggregate_dangling_challenge_claim_id_rejected() -> None:
    challenge = EvidenceChallenge(
        challenge_id=_challenge_id(),
        claim_id=_claim_id("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        defect_family=ChallengeDefectFamily.OTHER,
    )
    with pytest.raises(ValidationError, match="existing claim"):
        EvidenceClaimSet(challenges=(challenge,))


@pytest.mark.unit
@pytest.mark.gate
def test_aggregate_self_supersede_rejected() -> None:
    claim_id = _claim_id()
    with pytest.raises(ValidationError, match="supersede itself"):
        EvidenceBackedClaim(
            claim_id=claim_id,
            statement="Self supersede.",
            claim_kind="generic.claim",
            supersedes_claim_id=claim_id,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_manufacturing_example_uses_shared_contracts() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id("11111111111111111111111111111111"),
        statement="Intermittent equipment degradation caused the throughput incident.",
        claim_kind="incident.root_cause",
        supporting_evidence_ids=["e1", "e2"],
        contradicting_evidence_ids=["e3"],
        resolution=ClaimResolution.PENDING,
    )
    challenge = EvidenceChallenge(
        challenge_id=_challenge_id("22222222222222222222222222222222"),
        claim_id=claim.claim_id,
        defect_family=ChallengeDefectFamily.UNSUPPORTED_INFERENCE,
        description="Inference exceeds cited operational evidence.",
    )
    claim_set = EvidenceClaimSet(claims=(claim,), challenges=(challenge,))
    assert claim_set.claims[0].claim_kind == validate_claim_kind("incident.root_cause")
    assert challenge.defect_family is ChallengeDefectFamily.UNSUPPORTED_INFERENCE


@pytest.mark.unit
@pytest.mark.gate
def test_tprm_example_uses_shared_contracts() -> None:
    claim = EvidenceBackedClaim(
        claim_id=_claim_id("33333333333333333333333333333333"),
        statement="Vendor may be approved.",
        claim_kind="tprm.approval",
        supporting_evidence_ids=["v1"],
        resolution=ClaimResolution.INSUFFICIENT_EVIDENCE,
    )
    challenge = EvidenceChallenge(
        challenge_id=_challenge_id("44444444444444444444444444444444"),
        claim_id=claim.claim_id,
        defect_family=ChallengeDefectFamily.MISSING_EVIDENCE,
        defect_code="tprm.missing_financial_evidence",
    )
    claim_set = EvidenceClaimSet(claims=(claim,), challenges=(challenge,))
    assert claim_set.claims[0].resolution is ClaimResolution.INSUFFICIENT_EVIDENCE
    assert challenge.defect_code == validate_defect_code("tprm.missing_financial_evidence")


@pytest.mark.unit
@pytest.mark.gate
def test_import_boundary_no_forbidden_layers() -> None:
    module = inspect.getmodule(EvidenceBackedClaim)
    assert module is not None
    source = inspect.getsource(module)
    lowered = source.lower()
    for fragment in _FORBIDDEN_IMPORT_FRAGMENTS:
        assert fragment not in lowered, f"forbidden fragment {fragment}"


@pytest.mark.unit
@pytest.mark.gate
def test_module_source_has_no_forbidden_terms() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8").lower()
    forbidden_terms = (
        "dict[str, any]",
        ": any",
        "manufacturing",
        "incident",
        "vendor",
        "tprm",
        "contractor",
        "scenario",
        "critic",
    )
    for term in forbidden_terms:
        assert term not in source, f"forbidden term in production module: {term}"
