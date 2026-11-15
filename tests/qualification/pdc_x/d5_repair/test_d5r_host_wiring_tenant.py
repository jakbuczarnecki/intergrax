# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-E (F-D5-01): the hosts that hold a tenant bind it to the evaluation registry.

A full host start is not needed: the call of ``wire_application_evaluation`` in each host
module is pinned on the AST (the keyword ``tenant_id`` exists and derives from the tenant
the function holds). The behaviour of the wiring itself is pinned on the real function.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest
from intergrax.applications._shared import (
    harness_host_runtime,
    policy_wiring,
    runtime_config_bridge,
    scenario_runtime_baseline,
)
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    TenantScopedOnlineEvaluationRegistry,
)

pytestmark = pytest.mark.unit


def _calls(module: ModuleType) -> list[ast.Call]:
    source = Path(str(module.__file__)).read_text(encoding="utf-8")
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "wire_application_evaluation"
    ]


def _names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            out.add(item.id)
        elif isinstance(item, ast.Attribute):
            out.add(item.attr)
    return out


@pytest.mark.parametrize(
    ("module", "tenant_name"),
    [
        (harness_host_runtime, "tenant_id"),
        (runtime_config_bridge, "tenant_id"),
        (scenario_runtime_baseline, "resolved_tenant_id"),
    ],
)
def test_host_binds_the_tenant_it_holds(module: ModuleType, tenant_name: str) -> None:
    calls = _calls(module)
    assert calls, "host no longer wires evaluation"
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "tenant_id" in keywords, "wire_application_evaluation called without the tenant"
        assert tenant_name in _names(keywords["tenant_id"])


def _env(tmp_path: Path) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="d5r.tenant")
    return env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(update={"registry_path": tmp_path / "wired.json"})
        }
    )


def test_policy_wiring_never_touches_the_registry(tmp_path: Path) -> None:
    env = _env(tmp_path)
    policy_wiring.wire_policy_bundle(env)
    assert not (tmp_path / "wired.json").exists()


def _obs(observation_id: str) -> OnlineEvaluationObservation:
    return OnlineEvaluationObservation(
        observation_id=observation_id,
        run_id="run-1",
        agent_id="agent:research",
        mode=OnlineEvaluationMode.SHADOW,
        scenario_id="s-math",
        passed=True,
        score=1.0,
    )


def test_tenant_bound_wiring_cannot_see_another_tenant(tmp_path: Path) -> None:
    env = _env(tmp_path)
    a = wire_application_evaluation(env, tenant_id="tenant-a")
    b = wire_application_evaluation(env, tenant_id="tenant-b")
    assert isinstance(a.registry, TenantScopedOnlineEvaluationRegistry)
    a.registry.append(_obs("a-1"))
    assert b.registry is not None
    assert b.registry.list_observations() == []


def test_tenantless_wiring_cannot_see_tenant_rows(tmp_path: Path) -> None:
    env = _env(tmp_path)
    scoped = wire_application_evaluation(env, tenant_id="tenant-a").registry
    assert scoped is not None
    scoped.append(_obs("a-1"))
    unscoped = wire_application_evaluation(env)
    assert isinstance(unscoped.registry, FileOnlineEvaluationRegistry)
    assert unscoped.registry.list_observations() == []
