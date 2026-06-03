# © Artur Czarnecki. All rights reserved.

"""Environment vs agent contract consistency checks (Phase H-APP.1.7)."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.errors import ApplicationManifestConformanceError
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentContract


class EnvironmentSkillToolConsistencyCheck:
    """Warn/fail when agent contracts exceed environment profiles."""

    def __init__(self, *, fail_on_violation: bool = True) -> None:
        self._fail = fail_on_violation

    def validate_binding(
        self,
        binding: AgentBinding,
        env: ApplicationEnvironmentProfile,
    ) -> list[str]:
        if binding.factory is not None or binding.factory_path:
            return []
        agent_type = binding.resolved_agent_type()
        contract = _contract_for_agent(agent_type, binding)
        violations: list[str] = []

        env_tools = _environment_tool_ids(env)
        for tool_id in contract.allowed_tools:
            if tool_id not in env_tools:
                violations.append(
                    f"{binding.display_name()}: allowed_tools {tool_id!r} not in environment"
                )

        env_skills = _environment_skill_ids(env)
        for skill_manifest in contract.skills:
            skill_id = skill_manifest.skill_id
            if skill_id not in env_skills:
                violations.append(
                    f"{binding.display_name()}: skill {skill_id!r} not in environment"
                )

        if violations and self._fail:
            raise ApplicationManifestConformanceError("; ".join(violations))
        return violations

    def validate_roster(
        self,
        bindings: list[AgentBinding],
        env: ApplicationEnvironmentProfile,
    ) -> list[str]:
        all_violations: list[str] = []
        for binding in bindings:
            if not binding.enabled:
                continue
            all_violations.extend(self.validate_binding(binding, env))
        return all_violations


def _contract_for_agent(agent_type: type[Agent], binding: AgentBinding) -> AgentContract:
    agent = agent_type()
    contract = agent.get_contract()
    if binding.contract_id:
        return contract.model_copy(update={"id": binding.contract_id})
    return contract


def _environment_tool_ids(env: ApplicationEnvironmentProfile) -> frozenset[str]:
    profile = env.tool_profile
    if profile.register_all_catalog_bundles:
        return frozenset(profile.enabled)
    return frozenset(profile.enabled)


def _environment_skill_ids(env: ApplicationEnvironmentProfile) -> frozenset[str]:
    from intergrax.skills.registry.factory import enabled_skill_ids_for_profile

    return frozenset(enabled_skill_ids_for_profile(env.skill_profile))
