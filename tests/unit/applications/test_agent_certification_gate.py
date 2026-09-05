# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-4 — STRICT product roster agent certification gate."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.agent_certification_wiring import (
    apply_roster_agent_governance,
    check_strict_product_agent_certification,
    materialize_roster_certifications_for_agents,
    validate_strict_roster_agent_certification,
)
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications.contracts.agent_governance import AgentGovernanceProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_materialize_roster_certifications_for_enabled_agents() -> None:
    agents = [AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])]
    profile = materialize_roster_certifications_for_agents(agents, app_id="demo")
    assert len(profile.certifications) == 1
    assert profile.certifications[0].agent_id == "echo"


def test_strict_gate_blocks_experimental_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = ApplicationManifest.product(
        app_id="gate_cert",
        name="Gate Cert",
        route_prefix="/v1/gate_cert",
        env_prefix="GATE_CERT_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="gate_cert.product")
    env = apply_roster_agent_governance(env, agents=manifest.agents, app_id="gate_cert")

    original_get_contract = EchoAgent.get_contract

    def _experimental_contract(self: EchoAgent) -> object:
        contract = original_get_contract(self)
        return contract.model_copy(update={"lifecycle_state": AgentLifecycleState.EXPERIMENTAL})

    monkeypatch.setattr(EchoAgent, "get_contract", _experimental_contract)

    violations = validate_strict_roster_agent_certification(manifest, env)
    assert any("blocked on STRICT product host" in item for item in violations)


def test_strict_gate_requires_certification_for_staging_agent() -> None:
    manifest = ApplicationManifest.product(
        app_id="gate_cert2",
        name="Gate Cert 2",
        route_prefix="/v1/gate_cert2",
        env_prefix="GATE_CERT2_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="gate_cert2.product").model_copy(
        update={"agent_governance_profile": AgentGovernanceProfile()},
    )
    violations = validate_strict_roster_agent_certification(manifest, env)
    assert any("requires AgentCertificationRecord" in item for item in violations)


@pytest.mark.parametrize("product_id,manifest", list(iter_strict_product_manifests()))
def test_reference_strict_product_hosts_pass_certification_gate(
    product_id: str,
    manifest: ApplicationManifest,
) -> None:
    assert check_strict_product_agent_certification(product_id, manifest) == []
