# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.run_environment import merge_environment, render_namespace_template
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_run import AgentEnvironmentOverrides, AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.memory_scope import MemoryScope
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


def _contract(**updates: object) -> AgentContract:
    base = AgentContract(
        id="research",
        name="Research",
        description="Research agent",
        capabilities=["research.scan"],
        skills=[_SKILL],
        extra_tools=[_TOOL],
        allowed_tools=["rag.retrieve", "websearch.query"],
        risk_level=AgentRiskLevel.MEDIUM,
        max_steps=12,
        default_rag_collection="research_cache",
    )
    return base.model_copy(update=updates)


def _request(**updates: object) -> AgentRunRequest:
    base = AgentRunRequest(
        input="query",
        identity=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
        ),
        metadata={"task_id": "task-9"},
    )
    return base.model_copy(update=updates)


@pytest.mark.unit
@pytest.mark.gate
def test_merge_environment_binding_slices_override_rag_and_tools() -> None:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="research.host")
    profile.context_profile.enable_rag = False
    binding = AgentBinding.reference(
        contract_id="research",
        capabilities=["research.scan"],
        rag_collection_override="host_collection",
        tool_allowlist_extra=["extra.tool"],
        tool_denylist=["websearch.query"],
    )
    merged = merge_environment(
        contract=_contract(),
        request=_request(),
        app_profile=profile,
        binding=binding,
    )
    assert merged.profile_id == "research.host"
    assert merged.enable_rag is False
    assert merged.rag_collection_ids == ["host_collection"]
    assert "extra.tool" in merged.allowed_tools
    assert "websearch.query" not in merged.allowed_tools
    assert merged.memory_namespace == "research/tenant-a/user-1"


@pytest.mark.unit
@pytest.mark.gate
def test_merge_environment_org_scope_without_user_id() -> None:
    binding = AgentBinding.reference(
        contract_id="batch",
        memory_scope_override=MemoryScope.ORG,
    )
    merged = merge_environment(
        contract=_contract(id="batch", memory_scope=MemoryScope.ORG),
        request=_request(
            identity=RequestIdentity(
                tenant_id="tenant-a",
                user_id=None,
                principal_type=PrincipalType.ORG_SYSTEM,
            ),
        ),
        binding=binding,
    )
    assert merged.memory_scope == MemoryScope.ORG
    assert merged.memory_namespace == "org/tenant-a/batch"


@pytest.mark.unit
@pytest.mark.gate
def test_merge_environment_rejects_user_scope_without_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        merge_environment(
            contract=_contract(),
            request=_request(
                identity=RequestIdentity(
                    tenant_id="tenant-a",
                    user_id=None,
                    principal_type=PrincipalType.USER,
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_render_namespace_template_supports_metadata_placeholders() -> None:
    rendered = render_namespace_template(
        "legal/{tenant_id}/{matter_id}",
        identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        agent_id="legal",
        metadata={"matter_id": "m-42"},
    )
    assert rendered == "legal/t1/m-42"
