# © Artur Czarnecki. All rights reserved.

"""merge_environment and effective run environment (architecture §30 · ACP-DX-2)."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agents.org_policy_merge import merge_organizational_policy_context
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.applications.contracts.org_policy import OrganizationalPolicyContext
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import (
    AgentEnvironmentOverrides,
    AgentRunRequest,
    RequestIdentity,
    require_user_id_for_user_memory_scope,
)
from intergrax.contracts.agent_run_enums import AgentRunAutonomyLevel, SideEffectMode
from intergrax.contracts.memory_scope import MemoryScope

_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


class EffectiveAgentRunEnvironment(BaseModel):
    """Merged Tier-0/3 slices for one agent session (architecture §30.1)."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    contract_id: str
    tenant_id: str
    user_id: str | None = None
    principal_type: str = "user"
    memory_scope: MemoryScope = MemoryScope.USER
    memory_namespace: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    rag_collection_ids: list[str] = Field(default_factory=list)
    llm_profile_id: str | None = None
    allowed_llm_models: list[str] = Field(default_factory=lambda: ["balanced"])
    default_llm_model: str = "balanced"
    enable_rag: bool = True
    enable_websearch: bool = True
    side_effect_mode: SideEffectMode = SideEffectMode.IMMEDIATE
    max_steps: int | None = None
    checkpoint_every_step: bool = True
    autonomy_level: AgentRunAutonomyLevel = AgentRunAutonomyLevel.BALANCED
    merged_metadata: dict[str, Any] = Field(default_factory=dict)
    profile_id: str | None = None
    organizational: OrganizationalPolicyContext | None = None


def render_namespace_template(
    template: str,
    *,
    identity: RequestIdentity,
    agent_id: str,
    metadata: dict[str, Any],
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Render ``{tenant_id}``-style placeholders (architecture §30.9)."""
    values: dict[str, str] = {
        "tenant_id": identity.tenant_id,
        "user_id": identity.user_id or "",
        "agent_id": agent_id,
        "org_id": identity.tenant_id,
        "session_id": session_id or "",
        "task_id": task_id or str(metadata.get("task_id") or ""),
    }
    for key, raw in metadata.items():
        if isinstance(raw, (str, int, float)) and key.isidentifier():
            values[key] = str(raw)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return values.get(name, "")

    return _PLACEHOLDER_RE.sub(_replace, template)


def resolve_memory_scope(
    *,
    contract: AgentContract,
    request: AgentRunRequest,
    binding: AgentBinding | None,
    app_profile: ApplicationEnvironmentProfile | None,
) -> MemoryScope:
    overrides = request.environment_overrides
    if overrides is not None and overrides.memory_scope is not None:
        return overrides.memory_scope
    if binding is not None and binding.memory_scope_override is not None:
        return binding.memory_scope_override
    if contract.memory_scope is not None:
        return contract.memory_scope
    if app_profile is not None and not app_profile.memory_profile.enable_user_memory:
        if app_profile.memory_profile.enable_org_memory:
            return MemoryScope.ORG
    return MemoryScope.USER


def resolve_memory_namespace(
    *,
    scope: MemoryScope,
    contract: AgentContract,
    identity: RequestIdentity,
    request: AgentRunRequest,
    binding: AgentBinding | None,
) -> str:
    overrides = request.environment_overrides
    if overrides is not None and overrides.memory_namespace:
        return render_namespace_template(
            overrides.memory_namespace,
            identity=identity,
            agent_id=contract.id,
            metadata=request.metadata,
            session_id=request.session_id,
        )

    if scope == MemoryScope.CUSTOM and contract.memory_namespace_template:
        template = contract.memory_namespace_template
    elif scope == MemoryScope.ORG:
        template = "org/{tenant_id}/{agent_id}"
    elif scope == MemoryScope.TASK:
        template = "task/{tenant_id}/{task_id}/{agent_id}"
    else:
        template = "{agent_id}/{tenant_id}/{user_id}"

    return render_namespace_template(
        template,
        identity=identity,
        agent_id=contract.id,
        metadata=request.metadata,
        session_id=request.session_id,
    )


def merge_allowed_tools(
    *,
    contract: AgentContract,
    binding: AgentBinding | None,
    overrides: AgentEnvironmentOverrides | None,
) -> list[str]:
    tools = {tool_id for tool_id in contract.allowed_tools if tool_id}
    if binding is not None:
        tools.update(binding.tool_allowlist_extra)
        tools -= set(binding.tool_denylist)
    if overrides is not None:
        tools.update(overrides.tool_allowlist_add)
        tools -= set(overrides.tool_allowlist_remove)
    return sorted(tools)


def merge_rag_collections(
    *,
    contract: AgentContract,
    binding: AgentBinding | None,
    overrides: AgentEnvironmentOverrides | None,
) -> list[str]:
    if overrides is not None and overrides.rag_collection_ids:
        return list(overrides.rag_collection_ids)
    if overrides is not None and overrides.rag_collection:
        return [overrides.rag_collection]
    if binding is not None and binding.rag_collection_override:
        return [binding.rag_collection_override]
    if contract.default_rag_collection:
        return [contract.default_rag_collection]
    return []


def merge_environment(
    *,
    contract: AgentContract,
    request: AgentRunRequest,
    app_profile: ApplicationEnvironmentProfile | None = None,
    binding: AgentBinding | None = None,
    configure_run_overlay: dict[str, Any] | None = None,
) -> EffectiveAgentRunEnvironment:
    """
    Merge platform → application → binding → request overrides (architecture §30.1).

    Raises ``ValueError`` when memory scope validation fails (§30.9).
    """
    identity = request.identity
    memory_scope = resolve_memory_scope(
        contract=contract,
        request=request,
        binding=binding,
        app_profile=app_profile,
    )
    if memory_scope == MemoryScope.USER:
        require_user_id_for_user_memory_scope(identity, memory_scope="user")

    memory_namespace = resolve_memory_namespace(
        scope=memory_scope,
        contract=contract,
        identity=identity,
        request=request,
        binding=binding,
    )

    overrides = request.environment_overrides
    options = request.execution_options

    enable_rag = True
    enable_websearch = True
    llm_profile_id: str | None = None
    profile_id: str | None = None
    if app_profile is not None:
        enable_rag = app_profile.context_profile.enable_rag
        enable_websearch = app_profile.context_profile.enable_websearch
        profile_id = app_profile.profile_id
        if app_profile.llm_profile is not None:
            llm_profile_id = app_profile.llm_profile.model or str(
                app_profile.llm_profile.provider.value
            )

    if overrides is not None:
        if overrides.llm_profile_id:
            llm_profile_id = overrides.llm_profile_id
        elif overrides.llm_profile_slug:
            llm_profile_id = overrides.llm_profile_slug

    merged_metadata = dict(request.metadata)
    if overrides is not None and overrides.metadata_patch:
        merged_metadata.update(overrides.metadata_patch)
    if configure_run_overlay:
        merged_metadata.update(configure_run_overlay)

    max_steps = contract.max_steps
    if options is not None and options.max_steps is not None:
        max_steps = options.max_steps

    org_envelope = app_profile.organizational_policy if app_profile is not None else None
    capability = contract.capabilities[0] if contract.capabilities else None
    organizational = merge_organizational_policy_context(
        envelope=org_envelope,
        binding=binding,
        metadata=merged_metadata,
        capability=capability,
    )

    return EffectiveAgentRunEnvironment(
        agent_id=contract.id,
        contract_id=contract.id,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        principal_type=identity.principal_type.value,
        memory_scope=memory_scope,
        memory_namespace=memory_namespace,
        allowed_tools=merge_allowed_tools(
            contract=contract,
            binding=binding,
            overrides=overrides,
        ),
        rag_collection_ids=merge_rag_collections(
            contract=contract,
            binding=binding,
            overrides=overrides,
        ),
        llm_profile_id=llm_profile_id,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
        side_effect_mode=(
            options.side_effect_mode if options is not None else SideEffectMode.IMMEDIATE
        ),
        max_steps=max_steps,
        checkpoint_every_step=(
            options.checkpoint_every_step if options is not None else True
        ),
        autonomy_level=(
            options.autonomy_level
            if options is not None
            else AgentRunAutonomyLevel.BALANCED
        ),
        merged_metadata=merged_metadata,
        profile_id=profile_id,
        organizational=organizational,
    )
