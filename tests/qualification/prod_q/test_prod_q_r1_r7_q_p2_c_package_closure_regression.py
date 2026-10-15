# © Artur Czarnecki. All rights reserved.

"""P2-C-R1 — package closure must not hide AGENT roster mismatches."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.package_wiring import (
    build_application_package,
    validate_application_package_closure,
)
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.application_package import ApplicationDependencyKind
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from tests.unit.applications.test_production_registry_startup_ac3_final import _APP, _ENV

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def _empty_snapshot() -> HarnessRegistrySnapshot:
    return HarnessRegistrySnapshot(
        integration_profile=IntegrationProfile(),
        tool_registry=None,
        skill_registry=None,
        agent_registry=None,
        prompt_registry=None,
        policy_bundle=None,
    )


def test_mount_echo_search_surfaces_agent_roster_package_violation() -> None:
    """Lab mount resolves materialized contract id ≠ declarative contract_id (search)."""
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="AC3 mount (unsafe for package closure)",
        agents=[AgentBinding.mount(EchoAgent, contract_id="search")],
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    package = build_application_package(manifest, env)
    agent_refs = {
        dep.ref
        for dep in package.dependencies
        if dep.kind is ApplicationDependencyKind.AGENT
    }
    assert "search" not in agent_refs
    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        _empty_snapshot(),
    )
    assert any(
        violation.startswith("package agent dependency ")
        for violation in violations
    )


def test_echo_contract_id_aligns_package_agent_ref_with_roster() -> None:
    """Declarative contract_id must match materialized AgentContract.id (echo)."""
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="AC3 echo-aligned",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    package = build_application_package(manifest, env)
    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        _empty_snapshot(),
    )
    agent_violations = [
        v for v in violations if v.startswith("package agent dependency ")
    ]
    assert agent_violations == []
