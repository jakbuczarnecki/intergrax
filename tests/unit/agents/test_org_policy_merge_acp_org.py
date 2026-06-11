# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.compliance_summary import build_compliance_summary
from intergrax.agents.run_environment import merge_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.applications.contracts.org_policy import lab_strict_org_envelope
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType, StepNextAction
from intergrax.contracts.agent_run_trace import AgentRunTrace, AgentStepRecord, AgentStepStatus, PolicyCheckPhase, PolicyVerdictRecord
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.skills.core.contracts import SkillManifest
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


_TOOL = ToolContract(
    tool_id="demo.tool",
    name="demo.tool",
    description="demo",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=False,
    risk_level=ToolRiskLevel.LOW,
)

_SKILL = SkillManifest(
    skill_id="demo.skill",
    description="demo",
    tool_ids=("demo.tool",),
)


def _contract() -> AgentContract:
    return AgentContract(
        id="cs_agent",
        name="CS",
        description="customer service",
        capabilities=["customer_service.intake"],
        skills=[_SKILL],
        extra_tools=[_TOOL],
        allowed_tools=["rag.retrieve"],
        risk_level=AgentRiskLevel.LOW,
        max_steps=4,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_merge_environment_materializes_organizational_context() -> None:
    profile = ApplicationEnvironmentProfile.lab_org_virtual_workforce_defaults()
    binding = AgentBinding.reference(
        contract_id="cs_agent",
        org_role_id="customer_service_rep",
        tool_denylist=["sandbox.exec"],
    )
    merged = merge_environment(
        contract=_contract(),
        request=AgentRunRequest(
            input="hello",
            identity=RequestIdentity(
                tenant_id="tenant-lab",
                user_id="user-1",
                principal_type=PrincipalType.USER,
            ),
            metadata={"channel": "chat"},
        ),
        app_profile=profile,
        binding=binding,
    )
    assert merged.organizational is not None
    assert merged.organizational.organization_id == "lab.virtual_org"
    assert merged.organizational.org_role_id == "customer_service_rep"
    assert merged.organizational.active_scenario_id == "customer_intake"
    assert "sop.customer_intake" in merged.organizational.active_playbook_ids
    assert "phone.*" in merged.organizational.effective_tool_denies
    assert "sandbox.exec" in merged.organizational.effective_tool_denies
    assert merged.organizational.channel_policy.default_channel == "chat"


@pytest.mark.unit
@pytest.mark.gate
def test_lab_strict_envelope_fixture_shape() -> None:
    envelope = lab_strict_org_envelope()
    assert envelope.schema_version == "org_policy_envelope.v1"
    assert "phone" in envelope.channel_policy.denied_channels
    assert envelope.compliance_profile_id == "lab.org.compliance.v1"


@pytest.mark.unit
@pytest.mark.gate
def test_compliance_summary_counts_policy_denials() -> None:
    trace = AgentRunTrace(
        run_id="run-1",
        steps=[
            AgentStepRecord(
                step_id="step-0000",
                step_index=0,
                status=AgentStepStatus.SUCCEEDED,
                next_action=StepNextAction.CONTINUE,
                state_version=1,
                policy_verdicts=[
                    PolicyVerdictRecord(
                        phase=PolicyCheckPhase.PRE,
                        action=PolicyAction.ALLOW,
                        reason="ok",
                        policy_rule_id="kernel.default_allow",
                    ),
                    PolicyVerdictRecord(
                        phase=PolicyCheckPhase.PRE,
                        action=PolicyAction.DENY,
                        reason="channel denied",
                        policy_rule_id="org.channel.denied",
                    ),
                ],
            )
        ],
    )
    summary = build_compliance_summary(trace)
    assert summary.deny_count == 1
    assert "org.channel.denied" in summary.rules_triggered
