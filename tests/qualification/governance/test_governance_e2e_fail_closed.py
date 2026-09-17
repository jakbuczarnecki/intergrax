# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — fail-closed matrix anchors (see catalog for full evidence map)."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    UnavailableRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionEvaluator,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionRequest,
)

from tests.qualification.governance.catalog import GOV_FINAL_4_FAILURE_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_failure_matrix_catalog_documents_policy_evaluation_fail_closed() -> None:
    windows = {entry.failure_window for entry in GOV_FINAL_4_FAILURE_CATALOG}
    assert "policy evaluation exception" in windows


def test_failure_injection_unconfigured_root_admission_denies() -> None:
    evaluator = RuntimeExecutionPolicyAdmissionEvaluator(policy_engine=RuntimePolicyEngine())
    result = evaluator.evaluate(
        RuntimeExecutionPolicyAdmissionRequest(
            tenant_id="tenant-fc",
            workspace_id="workspace-fc",
            principal_id="principal-fc",
            collaborative_authority_scopes=("workspace.read",),
        ),
    )
    assert result.policy_decision.action is PolicyAction.DENY


def test_failure_injection_unavailable_admission_port_denies() -> None:
    result = UnavailableRuntimeExecutionPolicyAdmission().evaluate(
        RuntimeExecutionPolicyAdmissionRequest(
            tenant_id="tenant-fc",
            workspace_id="workspace-fc",
            principal_id="principal-fc",
            collaborative_authority_scopes=("workspace.read",),
        ),
    )
    assert result.policy_decision.action is PolicyAction.DENY
