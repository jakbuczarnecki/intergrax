# © Artur Czarnecki. All rights reserved.

"""ACP-CLOSE-ORG-2 — UC-11 compliance golden per product host manifest."""

from __future__ import annotations

import pytest

from dispute_sim_application.manifest import DISPUTE_SIM_APPLICATION_MANIFEST
from intergrax.agents.run_environment import merge_environment
from intergrax.applications._shared.uc11_compliance_golden import (
    assert_golden_compliance_zero_denials,
    run_uc11_kernel_happy_path_step,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.org_policy import product_host_org_envelope
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from legal_application.manifest import LEGAL_APPLICATION_MANIFEST
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from poc_template_application.manifest import APPLICATION_MANIFEST as POC_TEMPLATE_MANIFEST
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _primary_capability(binding: AgentBinding, contract_capabilities: list[str]) -> str:
    if binding.capabilities:
        return binding.capabilities[0]
    if contract_capabilities:
        return contract_capabilities[0]
    raise AssertionError("binding has no capabilities for UC-11 scenario trigger")


def _uc11_cases() -> list[tuple[str, ApplicationManifest, AgentBinding]]:
    from intergrax_assistant_application.manifest import build_intergrax_assistant_manifest

    manifests: list[tuple[str, ApplicationManifest]] = [
        ("legal", LEGAL_APPLICATION_MANIFEST),
        ("research", RESEARCH_APPLICATION_MANIFEST),
        ("dispute_sim", DISPUTE_SIM_APPLICATION_MANIFEST),
        ("local_workspace", LOCAL_WORKSPACE_APPLICATION_MANIFEST),
        ("poc_template", POC_TEMPLATE_MANIFEST),
        ("intergrax_assistant", build_intergrax_assistant_manifest()),
    ]
    cases: list[tuple[str, ApplicationManifest, AgentBinding]] = []
    for product_id, manifest in manifests:
        binding = next((item for item in manifest.enabled_agents() if item.agent_type is not None), None)
        if binding is None:
            continue
        cases.append((product_id, manifest, binding))
    return cases


@pytest.mark.parametrize(
    ("product_id", "manifest", "binding"),
    _uc11_cases(),
    ids=[case[0] for case in _uc11_cases()],
)
def test_product_host_uc11_merge_materializes_organizational_context(
    product_id: str,
    manifest: ApplicationManifest,
    binding: AgentBinding,
) -> None:
    assert manifest.environment is not None
    assert binding.agent_type is not None
    agent = binding.agent_type()
    contract = agent.get_contract()
    capability = _primary_capability(binding, list(contract.capabilities))
    envelope = product_host_org_envelope(
        product_id=product_id,
        primary_capability=capability,
    )
    env = manifest.environment.with_uc11_organizational_policy(envelope)
    uc11_binding = binding.model_copy(update={"org_role_id": f"{product_id}_agent"})
    merged = merge_environment(
        contract=contract,
        request=AgentRunRequest(
            input="uc11-golden",
            identity=RequestIdentity(
                tenant_id=f"tenant-{product_id}",
                user_id="user-uc11",
                principal_type=PrincipalType.USER,
            ),
            metadata={"channel": "chat", "scenario_id": f"{product_id}_primary"},
        ),
        app_profile=env,
        binding=uc11_binding,
    )
    assert merged.organizational is not None
    assert merged.organizational.organization_id == f"{product_id}.org"
    assert merged.organizational.org_role_id == f"{product_id}_agent"
    assert merged.organizational.domain_fragments.get("compliance_profile_id") == (
        f"{product_id}.org.compliance.v1"
    )
    assert merged.organizational.active_scenario_id == f"{product_id}_primary"


@pytest.mark.parametrize(
    ("product_id", "manifest", "binding"),
    _uc11_cases(),
    ids=[case[0] for case in _uc11_cases()],
)
@pytest.mark.asyncio
async def test_product_host_uc11_golden_compliance_zero_denials(
    product_id: str,
    manifest: ApplicationManifest,
    binding: AgentBinding,
) -> None:
    assert manifest.environment is not None
    assert binding.agent_type is not None
    agent = binding.agent_type()
    contract = agent.get_contract()
    capability = _primary_capability(binding, list(contract.capabilities))
    envelope = product_host_org_envelope(
        product_id=product_id,
        primary_capability=capability,
    )
    env = manifest.environment.with_uc11_organizational_policy(envelope)
    uc11_binding = binding.model_copy(update={"org_role_id": f"{product_id}_agent"})
    merged = merge_environment(
        contract=contract,
        request=AgentRunRequest(
            input="uc11-golden-kernel",
            identity=RequestIdentity(
                tenant_id=f"tenant-{product_id}",
                user_id="user-uc11",
                principal_type=PrincipalType.USER,
            ),
            metadata={"channel": "chat", "scenario_id": f"{product_id}_primary"},
        ),
        app_profile=env,
        binding=uc11_binding,
    )
    trace = await run_uc11_kernel_happy_path_step(merged, channel="chat")
    summary = assert_golden_compliance_zero_denials(trace)
    assert summary.deny_count == 0
