# © Artur Czarnecki. All rights reserved.

"""Environment vs agent contract consistency checks (Phase H-APP.1.7)."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.errors import ApplicationManifestConformanceError
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentContract


def _is_contract_reference_binding(binding: AgentBinding) -> bool:
    """Contract-id-only roster entry — harness/scenario catalogs without agent imports."""
    contract_id = (binding.contract_id or "").strip()
    if not contract_id:
        return False
    return (
        binding.agent_type is None
        and binding.import_path is None
        and binding.factory is None
        and not binding.factory_path
    )


def _binding_identity_resolvable(binding: AgentBinding) -> bool:
    if binding.factory is not None or binding.factory_path:
        return True
    if _is_contract_reference_binding(binding):
        return True
    return binding.agent_type is not None or binding.import_path is not None


class EnvironmentSkillToolConsistencyCheck:
    """Warn/fail when agent contracts exceed environment profiles."""

    def __init__(self, *, fail_on_violation: bool = True) -> None:
        self._fail = fail_on_violation

    def validate_binding(
        self,
        binding: AgentBinding,
        env: ApplicationEnvironmentProfile,
    ) -> list[str]:
        if not _binding_identity_resolvable(binding):
            label = (binding.contract_id or "").strip() or binding.display_name()
            violations = [
                f"{label}: AgentBinding has no resolvable agent identity "
                "(requires agent_type, import_path, contract_id reference, or factory)",
            ]
            if self._fail:
                raise ApplicationManifestConformanceError(violations[0])
            return violations

        if binding.factory is not None or binding.factory_path:
            return []

        if _is_contract_reference_binding(binding):
            # Skill/tool consistency is resolved at runtime from harness catalogs;
            # contract-reference bindings intentionally omit importable agent types here.
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


class ProfileInvariantValidator:
    """Cross-bundle semantic checks beyond roster/tool consistency (APP-EVOL-8)."""

    def __init__(self, *, fail_on_violation: bool = True) -> None:
        self._fail = fail_on_violation

    def validate(self, env: ApplicationEnvironmentProfile) -> list[str]:
        from intergrax.integrations.contracts.base import IntegrationCategory

        violations: list[str] = []
        if env.context_profile.enable_rag:
            vector_slug = env.integration_profile.slug_for_category(IntegrationCategory.VECTOR_STORE)
            if not vector_slug:
                violations.append(
                    "context.enable_rag=true requires integration_profile vector_store slug",
                )
        if env.context_profile.enable_websearch:
            enabled_tools = set(env.tool_profile.enabled)
            if (
                "websearch.query" not in enabled_tools
                and not env.tool_profile.register_all_catalog_bundles
            ):
                violations.append(
                    "context.enable_websearch=true requires tool_profile websearch.query",
                )
        if violations and self._fail:
            raise ApplicationManifestConformanceError("; ".join(violations))
        return violations
