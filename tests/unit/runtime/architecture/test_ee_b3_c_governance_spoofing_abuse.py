# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-04/AC-06 governance bypass and policy decision spoofing abuse."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionRequest,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    DenyingRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionEvaluator,
    UnavailableRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 14, 9, 0, 0, tzinfo=timezone.utc)


def _admission_request() -> RuntimeExecutionPolicyAdmissionRequest:
    return RuntimeExecutionPolicyAdmissionRequest(
        tenant_id="tenant-b3c",
        workspace_id="ws-b3c",
        principal_id="principal-b3c",
        collaborative_authority_scopes=("workspace.read",),
    )


def test_ee_b3_c_unconfigured_root_admission_fail_closed_deny() -> None:
    result = RuntimeExecutionPolicyAdmissionEvaluator(
        policy_engine=RuntimePolicyEngine(),
    ).evaluate(_admission_request())
    assert result.policy_decision.action is PolicyAction.DENY
    assert result.policy_decision.action is not PolicyAction.ALLOW


def test_ee_b3_c_denying_and_unavailable_adapters_never_allow() -> None:
    deny = DenyingRuntimeExecutionPolicyAdmission().evaluate(_admission_request())
    unavailable = UnavailableRuntimeExecutionPolicyAdmission().evaluate(
        _admission_request()
    )
    assert deny.policy_decision.action is PolicyAction.DENY
    assert unavailable.policy_decision.action is PolicyAction.DENY


def test_ee_b3_c_spoofed_evaluated_policy_decision_rejected() -> None:
    bundle = build_immutable_runtime_policy_bundle(
        bundle_id="b3c-deny-pack",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="r.deny",
                effect="deny",
                match_action="external_work.create",
            ),
        ),
        issued_at=_T0,
    )
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        policy_rule_id="r.deny",
        policy_bundle_id=bundle.bundle_id,
        policy_bundle_version=bundle.version,
        policy_bundle_digest=bundle.canonical_digest,
        decision_id="forged-allow",
    )
    spoofed = EvaluatedPolicyDecision(
        decision=decision,
        bundle_id=bundle.bundle_id,
        bundle_version=bundle.version,
        bundle_digest=bundle.canonical_digest,
        matched_rule_id="r.deny",
        evaluated_at=_T0,
        request_digest="sha256:" + ("22" * 32),
    )
    with pytest.raises(ValueError, match="decision_action_mismatch_with_rule"):
        spoofed.assert_consistent_with_bundle(bundle)


def test_ee_b3_c_allowing_runtime_admission_not_wired_in_intergrax_tree() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    intergrax_root = repo / "intergrax"
    hits: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        if path.name == "runtime_execution_policy_admission.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "AllowingRuntimeExecutionPolicyAdmission" in text:
            hits.append(str(path.relative_to(repo)))
    assert hits == []
