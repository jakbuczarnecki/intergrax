# © Artur Czarnecki. All rights reserved.

"""P2-C-R1 — package closure must not hide AGENT roster mismatches."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    AgentRuntimeGovernanceMaterializationError,
    validated_capabilities_for_binding,
)
from intergrax.applications.contracts.application_capability_projection import (
    resolve_binding_contract_id,
)
from intergrax.applications._shared.package_wiring import (
    build_application_package,
    validate_application_package_closure,
)
from intergrax.applications._shared.roster_agent_contract_authority import (
    materialize_manifest_contract_authority_lab_compat,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
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
    authority = materialize_manifest_contract_authority_lab_compat(manifest)
    package = build_application_package(manifest, env, contract_authority=authority)
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
        contract_authority=authority,
    )
    assert any(
        violation.startswith("package agent dependency ")
        for violation in violations
    )


def test_declarative_search_binding_contract_id_matches_revision_roster() -> None:
    """Declarative host binding: projection/governance/package roster key = search (no class mount)."""
    binding = AgentBinding(contract_id="search")
    assert binding.agent_type is None and binding.import_path is None
    assert resolve_binding_contract_id(binding) == "search"
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="AC3 revision-bound serving",
        agents=[binding],
    )
    roster_ids = {
        resolve_binding_contract_id(b) for b in manifest.enabled_agents()
    }
    assert roster_ids == {"search"}


class _SearchRegistryStub(AgentRegistryRead):
    """Typed registry stub with AC3 revision-bound search contract metadata."""

    _contract = AgentContract(
        id="search",
        name="search",
        description="AC3 revision-bound search slot",
        version="1.0.0",
        capabilities=["echo.basic"],
    )

    def has(self, agent_id: str) -> bool:
        return agent_id == "search"

    def get_contract(self, agent_id: str) -> AgentContract:
        if agent_id != "search":
            raise KeyError(agent_id)
        return self._contract

    def list_agent_ids(self) -> list[str]:
        return ["search"]

    def find_by_capability(self, capability: str) -> list[str]:
        if capability in self._contract.capabilities:
            return ["search"]
        return []


def test_governance_accepts_declarative_search_against_revision_registry() -> None:
    binding = AgentBinding(contract_id="search")
    caps = validated_capabilities_for_binding(binding, _SearchRegistryStub())
    assert caps == frozenset({"echo.basic"})


def test_governance_rejects_unregistered_declarative_contract_id() -> None:
    binding = AgentBinding(contract_id="missing-agent")
    with pytest.raises(AgentRuntimeGovernanceMaterializationError, match="not registered"):
        validated_capabilities_for_binding(binding, _SearchRegistryStub())


def test_mount_echo_search_still_fails_package_closure_without_filtering() -> None:
    """Class mount materializes echo; declarative roster contract_id search — canonical fail-closed."""
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="AC3 unsafe mount",
        agents=[AgentBinding.mount(EchoAgent, contract_id="search")],
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    authority = materialize_manifest_contract_authority_lab_compat(manifest)
    package = build_application_package(manifest, env, contract_authority=authority)
    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        _empty_snapshot(),
        contract_authority=authority,
    )
    assert any(v.startswith("package agent dependency ") for v in violations)
