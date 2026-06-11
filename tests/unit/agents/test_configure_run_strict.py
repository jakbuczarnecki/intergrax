# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.configure_run_strict import (
    ConfigureRunStrictViolation,
    clamp_environment_overrides_strict,
    sanitize_configure_run_overlay_strict,
    validate_configure_run_overlay_strict,
)
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment, merge_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.org_policy import lab_strict_org_envelope
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_run import (
    AgentEnvironmentOverrides,
    AgentRunRequest,
    RequestIdentity,
)
from intergrax.contracts.agent_run_enums import AgentRunStatus, PrincipalType, TerminalReason
from intergrax.skills.core.contracts import SkillManifest
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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
        id="strict_probe",
        name="Strict Probe",
        description="strict enforcement probe",
        capabilities=["strict.probe"],
        skills=[_SKILL],
        extra_tools=[_TOOL],
        allowed_tools=["rag.retrieve"],
        risk_level=AgentRiskLevel.LOW,
        max_steps=2,
    )


def _strict_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="strict.host")
    return profile.model_copy(
        update={
            "execution_mode": ExecutionMode.STRICT,
            "organizational_policy": lab_strict_org_envelope(),
        },
    )


@pytest.mark.unit
@pytest.mark.gate
def test_validate_configure_run_overlay_rejects_tool_widen_keys() -> None:
    violations = validate_configure_run_overlay_strict(
        {"allowed_tools": ["evil.tool"], "prompt_id": "ok"},
        execution_mode=ExecutionMode.STRICT,
    )
    assert "configure_run.forbidden_key:allowed_tools" in violations


@pytest.mark.unit
@pytest.mark.gate
def test_clamp_environment_overrides_rejects_tool_widen() -> None:
    overrides = AgentEnvironmentOverrides(tool_allowlist_add=["evil.tool"])
    with pytest.raises(ConfigureRunStrictViolation, match="tool_widen"):
        clamp_environment_overrides_strict(
            overrides,
            execution_mode=ExecutionMode.STRICT,
            ceiling_tools={"rag.retrieve"},
        )


@pytest.mark.unit
@pytest.mark.gate
def test_merge_environment_rejects_configure_run_overlay_widen() -> None:
    with pytest.raises(ConfigureRunStrictViolation):
        merge_environment(
            contract=_contract(),
            request=AgentRunRequest(
                input="hello",
                identity=RequestIdentity(
                    tenant_id="tenant-a",
                    user_id="user-1",
                    principal_type=PrincipalType.USER,
                ),
            ),
            app_profile=_strict_profile(),
            configure_run_overlay={"allowed_tools": ["evil.tool"]},
        )


class _WidenConfigureRunAgent(IntergraxAgent):
    contract_id = "strict_probe"
    capabilities = ("strict.probe",)
    agent_name = "Widen Configure Run Agent"

    def build_context(self, request):  # type: ignore[no-untyped-def]
        from intergrax.runtime.nexus.config import RuntimeConfig
        from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
        from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
        from intergrax.runtime.nexus.session.session_manager import SessionManager
        from testing_support.builder import FakeLLMAdapter

        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    def configure_run(self, merged: EffectiveAgentRunEnvironment) -> dict[str, object]:
        _ = merged
        return {"allowed_tools": ["evil.tool"]}


@pytest.mark.asyncio
async def test_acp_run_fails_fast_on_configure_run_widen_in_strict() -> None:
    agent = _WidenConfigureRunAgent()
    from intergrax.agents.authoring.acp_session_host import (
        ACP_HOST_CONTEXT_KEY,
        ACPSessionHostContext,
    )

    request = AgentRunRequest(
        input="strict-deny",
        identity=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
        ),
        metadata={
            ACP_HOST_CONTEXT_KEY: ACPSessionHostContext(app_profile=_strict_profile()),
        },
    )
    result = await agent.run(request)
    assert result.status == AgentRunStatus.FAILED
    assert result.terminal_reason == TerminalReason.POLICY_DENIED
    assert result.errors
    assert "configure_run.forbidden_key:allowed_tools" in result.errors[0].message


@pytest.mark.unit
@pytest.mark.gate
def test_sanitize_allows_safe_configure_run_overlay_in_strict() -> None:
    sanitized = sanitize_configure_run_overlay_strict(
        {"prompt_catalog_id": "legal.v2", "threshold": 0.8},
        execution_mode=ExecutionMode.STRICT,
    )
    assert sanitized == {"prompt_catalog_id": "legal.v2", "threshold": 0.8}
