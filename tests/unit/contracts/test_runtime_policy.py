# © Artur Czarnecki. All rights reserved.

"""PolicyDecision contract invariants (G1B-3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = [pytest.mark.unit]

_COMPLETE_DIGEST = "sha256:" + ("ab" * 32)


def _decision(**overrides: object) -> PolicyDecision:
    payload: dict[str, object] = {"action": PolicyAction.ALLOW}
    payload.update(overrides)
    return PolicyDecision.model_validate(payload)


def test_policy_decision_is_immutable() -> None:
    decision = _decision()
    with pytest.raises(ValidationError):
        decision.action = PolicyAction.DENY
    with pytest.raises(ValidationError):
        decision.policy_rule_id = "mutated"


def test_policy_decision_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision.model_validate({"action": PolicyAction.ALLOW, "unknown_field": "x"})


def test_empty_bundle_provenance_is_valid() -> None:
    decision = _decision(
        policy_bundle_id="",
        policy_bundle_version="",
        policy_bundle_digest="",
    )
    assert decision.policy_bundle_id == ""
    assert decision.policy_bundle_version == ""
    assert decision.policy_bundle_digest == ""
    assert decision.has_attested_policy_bundle_refs() is False


def test_complete_bundle_provenance_is_valid() -> None:
    decision = _decision(
        policy_bundle_id="bundle-1",
        policy_bundle_version="1.0.0",
        policy_bundle_digest=_COMPLETE_DIGEST,
    )
    assert decision.has_attested_policy_bundle_refs() is True
    assert decision.policy_bundle_digest.startswith("sha256:")


@pytest.mark.parametrize(
    "overrides",
    [
        {"policy_bundle_id": "bundle-1"},
        {"policy_bundle_id": "bundle-1", "policy_bundle_version": "1.0.0"},
        {"policy_bundle_digest": _COMPLETE_DIGEST},
        {"policy_bundle_version": "1.0.0", "policy_bundle_digest": _COMPLETE_DIGEST},
        {"policy_bundle_id": "bundle-1", "policy_bundle_digest": _COMPLETE_DIGEST},
        {"policy_bundle_version": "1.0.0"},
    ],
)
def test_partial_bundle_provenance_is_rejected(overrides: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="policy_bundle_provenance_incomplete"):
        _decision(**overrides)


def test_non_sha256_bundle_digest_is_rejected() -> None:
    with pytest.raises(ValidationError, match="policy_bundle_digest_must_be_sha256"):
        _decision(
            policy_bundle_id="bundle-1",
            policy_bundle_version="1.0.0",
            policy_bundle_digest="md5:abc",
        )


def test_sha256_bundle_digest_is_accepted() -> None:
    decision = _decision(
        policy_bundle_id="bundle-1",
        policy_bundle_version="1.0.0",
        policy_bundle_digest="sha256:deadbeef",
    )
    assert decision.policy_bundle_digest == "sha256:deadbeef"


def test_empty_decision_id_remains_valid() -> None:
    decision = _decision(decision_id="")
    assert decision.decision_id == ""


def test_empty_policy_rule_id_remains_valid() -> None:
    decision = _decision(policy_rule_id="")
    assert decision.policy_rule_id == ""


def test_audit_payload_carries_domain_diagnostics() -> None:
    decision = _decision(
        audit_payload={"request_digest": "sha256:ff", "match_action": "CREATE_EXTERNAL_WORK"}
    )
    assert decision.audit_payload["request_digest"] == "sha256:ff"
    assert decision.audit_payload["match_action"] == "CREATE_EXTERNAL_WORK"


def test_audit_payload_does_not_substitute_canonical_bundle_provenance() -> None:
    decision = _decision(
        audit_payload={
            "policy_bundle_id": "from-audit",
            "policy_bundle_version": "9",
            "policy_bundle_digest": _COMPLETE_DIGEST,
        }
    )
    assert decision.has_attested_policy_bundle_refs() is False
    assert decision.policy_bundle_id == ""
    assert decision.policy_bundle_version == ""
    assert decision.policy_bundle_digest == ""


def test_provenance_identifiers_are_stripped() -> None:
    decision = _decision(
        policy_rule_id="  rule.a  ",
        policy_bundle_id="  bundle-1  ",
        policy_bundle_version="  1.0.0  ",
        policy_bundle_digest=f"  {_COMPLETE_DIGEST}  ",
        decision_id="  dec-1  ",
    )
    assert decision.policy_rule_id == "rule.a"
    assert decision.policy_bundle_id == "bundle-1"
    assert decision.policy_bundle_version == "1.0.0"
    assert decision.policy_bundle_digest == _COMPLETE_DIGEST
    assert decision.decision_id == "dec-1"


def test_schema_version_preserved() -> None:
    assert _decision().schema_version == "policy_decision.v1"
