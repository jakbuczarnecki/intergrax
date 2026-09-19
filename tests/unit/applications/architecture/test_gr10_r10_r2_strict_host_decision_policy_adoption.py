# © Artur Czarnecki. All rights reserved.

"""GR-10-R10-R2 — strict production host DecisionRequirementPolicy adoption gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    OrchestrationDecisionBoundCompositionError,
    build_production_orchestration_meaningful_side_effect_authorization_boundary,
    resolve_orchestration_decision_requirement_policy,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from tests.unit.runtime.governance.gr3_test_support import (
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_STRICT_HOST_DECISION_POLICY_INVENTORY,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]

_STRICT_HOST_RUNTIME_CALL_SITES: tuple[tuple[str, Path], ...] = (
    (
        "governed_contractor_application",
        _REPO / "applications" / "governed_contractor_application" / "host" / "factory.py",
    ),
    (
        "research_application",
        _REPO / "applications" / "research_application" / "host" / "factory.py",
    ),
    (
        "legal_application",
        _REPO / "applications" / "legal_application" / "host" / "factory.py",
    ),
    (
        "dispute_sim_application",
        _REPO / "applications" / "dispute_sim_application" / "host" / "factory.py",
    ),
    (
        "local_workspace_application",
        _REPO
        / "applications"
        / "local_workspace_application"
        / "host"
        / "host_runtime_composition.py",
    ),
)

_SCAFFOLD_PRODUCT_FACTORY = _REPO / "intergrax" / "scaffold" / "new_application_product.py"


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, context: object) -> object:
        from intergrax.contracts.decision_requirement_policy import DecisionRequirement

        self.calls += 1
        return DecisionRequirement.NOT_REQUIRED


def _repos_bundle() -> dict[str, object]:
    return {
        "profile_repository": InMemoryCollaborativeOperationPolicyProfileRepository(),
        "membership_repository": InMemoryWorkspaceMembershipRepository(),
        "principal_authority_repository": InMemoryPrincipalAuthorityRepository(),
        "delegation_repository": InMemoryAuthorityDelegationRepository(),
        "collaborative_policy_repository": InMemoryCollaborativePolicyRepository(),
        "runtime_policy_evaluator": RuntimePolicyEngine(),
    }


def _call_has_explicit_orchestration_policy(source: str, tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name != "build_harness_host_runtime":
            continue
        for keyword in node.keywords:
            if keyword.arg == "orchestration_decision_requirement_policy":
                return True
    return False


def test_gr10_r10_r2_strict_host_inventory_has_no_gap_rows() -> None:
    assert GR10_ORCHESTRATION_STRICT_HOST_DECISION_POLICY_INVENTORY
    for row in GR10_ORCHESTRATION_STRICT_HOST_DECISION_POLICY_INVENTORY:
        if row.orchestration_mse_applicable and row.strict_capable and row.production:
            assert row.coverage == "QUALIFIED"
            assert row.explicit_policy is True


@pytest.mark.parametrize(("host_id", "path"), _STRICT_HOST_RUNTIME_CALL_SITES)
def test_gr10_r10_r2_strict_host_factory_passes_explicit_policy_kwarg(
    host_id: str,
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    assert _call_has_explicit_orchestration_policy(source, tree), (
        f"{host_id}: build_harness_host_runtime must pass orchestration_decision_requirement_policy"
    )


def test_gr10_r10_r2_scaffold_product_factory_template_wires_policy() -> None:
    source = _SCAFFOLD_PRODUCT_FACTORY.read_text(encoding="utf-8-sig")
    assert "def orchestration_decision_requirement_policy_py" in source
    assert "orchestration_decision_requirement_policy=" in source
    assert "resolve_{short}_harness_orchestration_decision_requirement_policy" in source


def test_gr10_r10_r2_missing_generic_production_policy_still_fail_closed() -> None:
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        resolve_orchestration_decision_requirement_policy(None, production_mode=True)
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        build_production_orchestration_meaningful_side_effect_authorization_boundary(
            **_repos_bundle(),
            decision_requirement_policy=None,
            inner_execution_guard=default_gr3_inner_guard(default_gr3_identity_bundle()[0]),
            production_mode=True,
        )


def test_gr10_r10_r2_custom_policy_reaches_production_boundary() -> None:
    custom = _RecordingPolicy()
    boundary = build_production_orchestration_meaningful_side_effect_authorization_boundary(
        **_repos_bundle(),
        decision_requirement_policy=custom,
        inner_execution_guard=default_gr3_inner_guard(default_gr3_identity_bundle()[0]),
        production_mode=True,
    )
    assert boundary._decision_requirement_policy is custom


def test_gr10_r10_r2_explicit_permissive_policy_composition_succeeds() -> None:
    boundary = build_production_orchestration_meaningful_side_effect_authorization_boundary(
        **_repos_bundle(),
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        inner_execution_guard=default_gr3_inner_guard(default_gr3_identity_bundle()[0]),
        production_mode=True,
    )
    assert boundary is not None


def test_gr10_r10_r2_non_strict_resolver_allows_lab_policy() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr10.r10r2.nonstrict")
    assert env.execution_mode == ExecutionMode.BALANCED
    policy = resolve_orchestration_decision_requirement_policy(None, production_mode=False)
    assert policy is not None
