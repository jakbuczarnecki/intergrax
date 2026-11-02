# © Artur Czarnecki. All rights reserved.

"""Fail-closed tests for the EVOL-X automated B6 attestation contract.

In-memory Ed25519 keys are disposable fixtures. They are not independent custody
and the published governance record stays ``PENDING_REQUIRED_APPROVALS``.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.qualification.evol_x_b6_attestation import (
    CREDENTIAL_PRODUCTION,
    CREDENTIAL_TEST,
    CUSTODY_INDEPENDENT,
    FINDINGS_REGISTER_ID,
    PINNED_FINAL,
    PINNED_TES,
    POLICY_ID,
    POLICY_VERSION,
    QUALIFIED_CI_RUN_ID,
    SCHEMA_VERSION,
    SIGNATURE_ALGORITHM,
    TOOL_ID,
    TOOL_VERSION,
    TRUST_ROOT_SCHEMA,
    ProductionSigningForbidden,
    canonical_bytes,
    evaluate_acceptance,
    load_governance,
    public_key_b64,
    sign_attestation,
)
from scripts.qualification.evol_x_b6_independent_verification import (
    decision_fields_from_run,
)

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[3]
_GOVERNANCE = _ROOT / "docs/project/maintainers/qualification/evol_x/EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json"
_AMENDMENT = _ROOT / (
    "docs/project/maintainers/qualification/evol_x/"
    "EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md"
)
_REGISTER = _ROOT / "docs/project/maintainers/qualification/evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md"
_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
_DIGEST = "ab" * 32


def _body() -> dict[str, object]:
    mode = {"status": "PASS", "m09a": "MEASURED_PASS", "m09h": "NOT_MEASURED"}
    return {
        "schema_version": SCHEMA_VERSION,
        "ecap_policy_id": POLICY_ID,
        "ecap_policy_version": POLICY_VERSION,
        "operator_decision_ref": "OD-EVOL-14",
        "certification_profile": "AUTOMATION_FIRST_SCOPED",
        "test_execution_sha": PINNED_TES,
        "evidence_final_publication_sha": PINNED_FINAL,
        "verification_tool_id": TOOL_ID,
        "verification_tool_version": TOOL_VERSION,
        "verification_tool_digest": _DIGEST,
        "ci_provenance": {
            "run_id": QUALIFIED_CI_RUN_ID,
            "conclusion": "success",
            "head_sha": PINNED_TES,
            "profile": "evol_x_p3_l3",
        },
        "mode_a_result": dict(mode),
        "mode_b_result": dict(mode),
        "findings_disposition": {
            "register_id": FINDINGS_REGISTER_ID,
            "complete": True,
            "open_blocking": 0,
            "m09h_disposition": "NOT_MEASURED",
            "m09a_disposition": "MEASURED_PASS",
        },
        "frz_scope_declaration": {
            "scope": "EVOL-X-ONLY",
            "promotes_frz": False,
            "frz_evol_07": "OPEN_NOT_PROMOTED",
            "declares_evol_x_closed": False,
        },
        "attestor_identity": "fixture-attestor",
        "evidence_producer_identity": "evol-x-evidence-publisher",
        "key_id": "k-test",
        "trust_root_ref": "tr-fixture",
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "signing_timestamp": "2026-11-02T08:00:00+00:00",
        "validity_not_before": "2026-11-02T00:00:00+00:00",
        "validity_not_after": "2026-11-03T00:00:00+00:00",
        "attestation_id": "fixture-attestation-1",
        "replay_binding": _DIGEST,
        "credential_class": CREDENTIAL_TEST,
        "formal_acceptance_disposition": "NOT_ACCEPTED",
    }


def _root(
    private_key: Ed25519PrivateKey,
    *,
    custody: str = CREDENTIAL_TEST,
    attestor: str = "fixture-attestor",
    credential: str = CREDENTIAL_TEST,
    status: str = "ACTIVE",
    controlled: bool = False,
    trust_id: str = "tr-fixture",
) -> dict[str, object]:
    return {
        "schema_version": TRUST_ROOT_SCHEMA,
        "trust_root_id": trust_id,
        "custody_domain": custody,
        "keys": [
            {
                "key_id": "k-test",
                "public_key_b64": public_key_b64(private_key),
                "status": status,
                "credential_class": credential,
                "attestor_identity": attestor,
                "controlled_by_implementer": controlled,
            }
        ],
    }


def _sign_unchecked(private_key: Ed25519PrivateKey, body: dict[str, object]) -> dict[str, object]:
    unsigned = dict(body)
    unsigned.pop("signature", None)
    unsigned.pop("replay_binding", None)
    unsigned["replay_binding"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    unsigned["signature"] = base64.b64encode(private_key.sign(canonical_bytes(unsigned))).decode("ascii")
    return unsigned


def _evaluate(
    attestation: dict[str, object],
    trust_root: dict[str, object],
    *,
    governance: dict[str, object] | None = None,
    observed: dict[str, str] | None = None,
    replay_ledger: set[str] | None = None,
    now: datetime | None = None,
    technical: str = "PASS",
) -> dict[str, object]:
    return evaluate_acceptance(
        technical_verdict=technical,
        mode_a="PASS",
        mode_b="PASS",
        governance=governance if governance is not None else load_governance(_GOVERNANCE),
        attestation=attestation,
        trust_root=trust_root,
        observed=observed if observed is not None else {"mode_a": "PASS", "mode_b": "PASS"},
        expected_tool_digest=_DIGEST,
        replay_ledger=set() if replay_ledger is None else replay_ledger,
        now=_NOW if now is None else now,
    )


def _governance_in_force() -> dict[str, object]:
    governance = dict(load_governance(_GOVERNANCE))
    governance["ecap_amendment_activation"] = "IN_FORCE"
    governance["normative_effect"] = "IN_FORCE"
    governance["independent_attestation_environment"] = True
    governance["required_approvals_present"] = {
        "operator_direction": True,
        "ecap_maintainer_publication": True,
        "security_review": True,
        "independent_key_custody": True,
    }
    return governance


def test_published_records_keep_formal_b6_pending() -> None:
    governance = load_governance(_GOVERNANCE)
    assert governance["ecap_amendment_activation"] == "PENDING_REQUIRED_APPROVALS"
    assert governance["normative_effect"] == "NOT_IN_FORCE"
    assert governance["b6_acceptance_publication_sha"] is None
    assert governance["evol_x_closed"] is False
    amendment = _AMENDMENT.read_text(encoding="utf-8")
    assert "PENDING_REQUIRED_APPROVALS" in amendment
    assert "NOT_IN_FORCE" in amendment
    register = _REGISTER.read_text(encoding="utf-8")
    assert "OD-EVOL-14" in register
    assert "**OD-EVOL-12** | **PENDING**" in register
    assert "**OD-EVOLX-08** | **PENDING**" in register


def test_technical_pass_does_not_publish_acceptance() -> None:
    fields = decision_fields_from_run(
        mode_a_ok=True,
        mode_b_ok=True,
        skip_mode_b=False,
        exit_code=0,
        governance=load_governance(_GOVERNANCE),
        attestation=None,
        trust_root=None,
        replay_ledger=None,
        now=_NOW,
        expected_tool_digest=_DIGEST,
    )
    assert fields["technical_verification"]["verdict"] == "PASS"
    assert fields["authority_verification"]["verdict"] == "PENDING_REQUIRED_APPROVALS"
    assert fields["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"
    assert fields["b6_acceptance_publication_sha"] is None
    assert fields["evol_x_closed"] is False
    assert fields["authority_verification"]["evol_x_closure"] == "NOT_AUTHORIZED"


def test_valid_test_credential_cannot_authorize_formal_b6() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    result = _evaluate(signed, _root(private_key))
    assert result["attestation_verification"]["verdict"] == "VALID_NON_PRODUCTION_TEST_ONLY"
    assert result["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"
    assert "NON_PRODUCTION_TEST_ONLY" in result["formal_acceptance_blockers"]


def test_test_signer_refuses_production_and_acceptance_claims() -> None:
    private_key = Ed25519PrivateKey.generate()
    production = _body()
    production["credential_class"] = CREDENTIAL_PRODUCTION
    with pytest.raises(ProductionSigningForbidden):
        sign_attestation(private_key, production)
    accepted = _body()
    accepted["formal_acceptance_disposition"] = "ACCEPTED_AUTOMATED_B6"
    with pytest.raises(ProductionSigningForbidden):
        sign_attestation(private_key, accepted)


def test_unsigned_record_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    unsigned = _body()
    unsigned["signature"] = ""
    result = _evaluate(unsigned, _root(private_key))
    assert result["attestation_verification"]["reasons"] == ["UNSIGNED_OR_INCOMPLETE"]
    assert result["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"


def test_forged_signature_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    raw = bytearray(base64.b64decode(str(signed["signature"])))
    raw[0] ^= 0x01
    signed["signature"] = base64.b64encode(bytes(raw)).decode("ascii")
    result = _evaluate(signed, _root(private_key))
    assert "FORGED_SIGNATURE" in result["attestation_verification"]["reasons"]


def test_unknown_trust_root_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    result = _evaluate(signed, _root(private_key, trust_id="tr-unknown"))
    assert "UNKNOWN_TRUST_ROOT" in result["attestation_verification"]["reasons"]


def test_revoked_signing_key_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    result = _evaluate(signed, _root(private_key, status="REVOKED"))
    assert "REVOKED_SIGNING_KEY" in result["attestation_verification"]["reasons"]


def test_wrong_attestor_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    body = _body()
    body["attestor_identity"] = "other-attestor"
    signed = sign_attestation(private_key, body)
    result = _evaluate(signed, _root(private_key))
    assert "WRONG_ATTESTOR" in result["attestation_verification"]["reasons"]


def test_implementer_controlled_attestor_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    result = _evaluate(signed, _root(private_key, controlled=True))
    assert "IMPLEMENTER_CONTROLLED_ATTESTOR" in result["attestation_verification"]["reasons"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda body: body.__setitem__("test_execution_sha", "0" * 40), "WRONG_TES"),
        (lambda body: body.__setitem__("evidence_final_publication_sha", "1" * 40), "WRONG_FINAL"),
        (lambda body: body.__setitem__("ecap_policy_version", "1999.0-substituted"), "WRONG_POLICY_VERSION"),
        (lambda body: body.__setitem__("ecap_policy_id", "ECAP-POLICY-SUBSTITUTION"), "WRONG_POLICY_ID"),
        (
            lambda body: body.__setitem__(
                "verification_tool_digest",
                "cd" * 32,
            ),
            "MUTATED_ARTIFACTS",
        ),
        (lambda body: body["findings_disposition"].__setitem__("complete", False), "INCOMPLETE_FINDINGS"),
        (lambda body: body["frz_scope_declaration"].__setitem__("promotes_frz", True), "UNSUPPORTED_FRZ_PROMOTION"),
        (lambda body: body["mode_a_result"].__setitem__("m09h", "PASS"), "M09H_PASS_CLAIM"),
        (lambda body: body["ci_provenance"].__setitem__("run_id", ""), "MISSING_CI_EVIDENCE"),
        (lambda body: body.__setitem__("validity_not_after", "2026-11-02T01:00:00+00:00"), "STALE_ATTESTATION"),
        (
            lambda body: (
                body.__setitem__("signing_timestamp", "2026-11-02T01:00:00+00:00")
                or body.__setitem__("validity_not_after", "2026-11-02T02:00:00+00:00")
            ),
            "STALE_ATTESTATION",
        ),
    ],
)
def test_semantic_mutations_fail_closed(mutate: Callable[[dict[str, object]], object], reason: str) -> None:
    private_key = Ed25519PrivateKey.generate()
    body = _body()
    mutate(body)
    signed = sign_attestation(private_key, body)
    result = _evaluate(signed, _root(private_key))
    assert result["attestation_verification"]["verdict"] == "REJECTED"
    assert reason in result["attestation_verification"]["reasons"]
    assert result["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"


def test_stale_window_when_clock_is_outside_validity() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    later = datetime(2026, 11, 4, 0, 0, tzinfo=UTC)
    result = _evaluate(signed, _root(private_key), now=later)
    assert "STALE_ATTESTATION" in result["attestation_verification"]["reasons"]


def test_replayed_attestation_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    ledger: set[str] = set()
    first = _evaluate(signed, _root(private_key), replay_ledger=ledger)
    assert first["attestation_verification"]["verdict"] == "VALID_NON_PRODUCTION_TEST_ONLY"
    ledger.add(str(signed["replay_binding"]))
    second = _evaluate(signed, _root(private_key), replay_ledger=ledger)
    assert "REPLAYED_ATTESTATION" in second["attestation_verification"]["reasons"]


def test_altered_mode_results_are_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    result = _evaluate(signed, _root(private_key), observed={"mode_a": "FAIL", "mode_b": "PASS"})
    assert "ALTERED_MODE_RESULTS" in result["attestation_verification"]["reasons"]


def test_self_acceptance_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    body = _body()
    body["attestor_identity"] = "fixture-attestor"
    body["evidence_producer_identity"] = "fixture-attestor"
    signed = sign_attestation(private_key, body)
    result = _evaluate(signed, _root(private_key))
    assert "SELF_ACCEPTANCE" in result["attestation_verification"]["reasons"]


def test_insufficient_authority_blocks_formal_eligibility() -> None:
    private_key = Ed25519PrivateKey.generate()
    signed = sign_attestation(private_key, _body())
    governance = _governance_in_force()
    approvals = dict(governance["required_approvals_present"])
    approvals["security_review"] = False
    governance["required_approvals_present"] = approvals
    result = _evaluate(signed, _root(private_key), governance=governance)
    assert result["authority_verification"]["verdict"] == "INSUFFICIENT"
    assert "MISSING_SECURITY_REVIEW" in result["authority_verification"]["reasons"]
    assert result["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"


def test_in_force_flag_without_approvals_is_insufficient() -> None:
    governance = dict(load_governance(_GOVERNANCE))
    governance["ecap_amendment_activation"] = "IN_FORCE"
    governance["normative_effect"] = "IN_FORCE"
    result = evaluate_acceptance(
        technical_verdict="PASS",
        mode_a="PASS",
        mode_b="PASS",
        governance=governance,
        attestation=None,
        trust_root=None,
        observed=None,
        expected_tool_digest=_DIGEST,
        replay_ledger=set(),
        now=_NOW,
    )
    assert result["authority_verification"]["verdict"] == "INSUFFICIENT"
    assert result["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"


def test_conjunction_fixture_does_not_publish_acceptance() -> None:
    """Synthetic IN_FORCE governance exercises the conjunction. It is not custody."""
    private_key = Ed25519PrivateKey.generate()
    body = _body()
    body["credential_class"] = CREDENTIAL_PRODUCTION
    body["formal_acceptance_disposition"] = "ACCEPTED_AUTOMATED_B6"
    body["attestor_identity"] = "independent-attestor"
    body["evidence_producer_identity"] = "evol-x-evidence-publisher"
    signed = _sign_unchecked(private_key, body)
    root = _root(
        private_key,
        custody=CUSTODY_INDEPENDENT,
        attestor="independent-attestor",
        credential=CREDENTIAL_PRODUCTION,
    )
    fields = decision_fields_from_run(
        mode_a_ok=True,
        mode_b_ok=True,
        skip_mode_b=False,
        exit_code=0,
        governance=_governance_in_force(),
        attestation=signed,
        trust_root=root,
        replay_ledger=set(),
        now=_NOW,
        expected_tool_digest=_DIGEST,
    )
    assert fields["attestation_verification"]["verdict"] == "VALID_PRODUCTION_INDEPENDENT"
    assert fields["formal_acceptance_eligibility"] == "ELIGIBLE"
    assert fields["b6_acceptance_publication_sha"] is None
    assert fields["evol_x_closed"] is False
    assert fields["authority_verification"]["evol_x_closure"] == "NOT_AUTHORIZED"


def test_production_claim_on_test_custody_is_not_eligible() -> None:
    private_key = Ed25519PrivateKey.generate()
    body = _body()
    body["credential_class"] = CREDENTIAL_PRODUCTION
    body["formal_acceptance_disposition"] = "ACCEPTED_AUTOMATED_B6"
    signed = _sign_unchecked(private_key, body)
    result = _evaluate(signed, _root(private_key), governance=_governance_in_force())
    assert result["formal_acceptance_eligibility"] == "NOT_ELIGIBLE"
    assert result["attestation_verification"]["verdict"] == "REJECTED"
