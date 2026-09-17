# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — contract-first pluginability via composition (no monkeypatch)."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_execution_policy_admission import RuntimeExecutionPolicyAdmissionPort
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionResult,
)

from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionRequest,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _CustomDenyAdmission(RuntimeExecutionPolicyAdmissionPort):
    def evaluate(self, request: object) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.DENY,
                reason="gov_final_4_custom_admission",
            ),
        )


def test_scenario_plugin_custom_runtime_execution_policy_admission_port() -> None:
    port = _CustomDenyAdmission()
    result = port.evaluate(object())
    assert result.policy_decision.action is PolicyAction.DENY


def test_scenario_plugin_allowing_vs_denying_admission_via_composition() -> None:
    allow = AllowingRuntimeExecutionPolicyAdmission()
    deny = DenyingRuntimeExecutionPolicyAdmission()
    request = RuntimeExecutionPolicyAdmissionRequest(
        tenant_id="tenant-plugin",
        workspace_id="workspace-plugin",
        principal_id="principal-plugin",
        collaborative_authority_scopes=("workspace.read",),
    )
    assert allow.evaluate(request).policy_decision.action is PolicyAction.ALLOW
    assert deny.evaluate(request).policy_decision.action is PolicyAction.DENY
