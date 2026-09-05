# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-6 — ApplicationEnvironmentDiff wiring."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_diff_wiring import (
    build_application_environment_diff,
    diff_roster,
    diff_structured,
)
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications.contracts.application_environment_diff import (
    DiffRiskLevel,
    RosterChangeKind,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_diff_structured_reports_nested_change() -> None:
    diff = diff_structured({"a": {"b": 1}}, {"a": {"b": 2}})
    assert any(change.path.endswith("b") for change in diff.changes)


def test_diff_roster_detects_capability_change() -> None:
    left = [AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])]
    right = [AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.advanced"])]
    changes = diff_roster(left, right)
    assert len(changes) == 1
    assert changes[0].kind is RosterChangeKind.CAPABILITIES_CHANGED


def test_self_diff_is_low_risk() -> None:
    product_id, manifest = next(iter(iter_strict_product_manifests()))
    env = manifest.resolved_environment()
    diff = build_application_environment_diff(manifest, env, manifest, env)
    assert diff.risk_level is DiffRiskLevel.LOW
    assert diff.left_snapshot_id == diff.right_snapshot_id
    assert product_id


def test_execution_mode_change_is_high_risk() -> None:
    manifest = ApplicationManifest.product(
        app_id="diff_demo",
        name="Diff Demo",
        route_prefix="/v1/diff_demo",
        env_prefix="DIFF_DEMO_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="diff_demo.product")
    relaxed = env.model_copy(update={"execution_mode": ExecutionMode.BALANCED})
    strict = env.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    diff = build_application_environment_diff(manifest, relaxed, manifest, strict)
    assert diff.risk_level is DiffRiskLevel.HIGH
