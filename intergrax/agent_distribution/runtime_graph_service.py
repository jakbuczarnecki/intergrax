# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Candidate runtime graph builder and validation gates (AP-7 §17)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from intergrax.agent_distribution.agent_project_metadata import (
    AgentProjectMetadata,
    AgentProjectMetadataProvider,
)
from intergrax.agent_distribution.dependency import (
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.errors import CandidateRuntimeGraphError
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_graph import (
    GRAPH_SCHEMA_VERSION_V3,
    CandidateApplicationRuntimeGraph,
    RuntimeGraphAgentRef,
    RuntimeGraphThirdPartyRef,
)
from intergrax.runtime_graph_semantics import (
    GraphVisitState,
    format_agent_dependency_cycle,
    is_agent_distribution,
    is_application_distribution,
    is_platform_dependency,
    normalize_distribution_name,
    parse_dependency_name,
)
def _lock_package_index(
    lock: MaterializedRuntimeLock,
) -> dict[str, MaterializedLockPackage]:
    return {
        normalize_distribution_name(package.distribution_name): package
        for package in lock.packages
    }


def _lock_agent_closure_index(
    lock: MaterializedRuntimeLock,
) -> dict[str, MaterializedAgentClosureEntry]:
    return {entry.distribution_package_id: entry for entry in lock.agent_closure}


class CandidateRuntimeGraphBuilder:
    """Build structural candidate graphs from lock + roster + metadata (§17)."""

    def __init__(self, metadata_provider: AgentProjectMetadataProvider) -> None:
        self._metadata_provider = metadata_provider

    def build(
        self,
        *,
        lock: MaterializedRuntimeLock,
        effective_roster: EffectiveRoster,
        repository_declaration: RepositoryDependencyDeclaration,
        agent_metadata_refs: Mapping[str, str],
    ) -> CandidateApplicationRuntimeGraph:
        if lock.lock_id is None:
            raise CandidateRuntimeGraphError("lock must have content identity before graph build")

        direct_agents = self._direct_agents_from_roster(effective_roster)
        self._validate_direct_agents_against_lock(direct_agents, lock)

        packages_by_name = _lock_package_index(lock)
        closure_by_id = _lock_agent_closure_index(lock)

        transitive_agents, _ = self._traverse_transitive_agents(
            direct_agents=direct_agents,
            agent_metadata_refs=agent_metadata_refs,
            closure_by_id=closure_by_id,
            packages_by_name=packages_by_name,
        )

        direct_third_party = self._direct_third_party_from_declaration(
            repository_declaration=repository_declaration,
            packages_by_name=packages_by_name,
        )

        graph = CandidateApplicationRuntimeGraph(
            graph_schema_version=GRAPH_SCHEMA_VERSION_V3,
            application_id=effective_roster.application_id,
            materialized_runtime_lock_id=lock.lock_id,
            direct_agents=tuple(
                sorted(
                    direct_agents,
                    key=lambda item: (
                        item.logical_agent_id,
                        item.distribution_package_id,
                        item.package_digest,
                    ),
                )
            ),
            transitive_agents=tuple(
                sorted(
                    transitive_agents,
                    key=lambda item: (
                        item.logical_agent_id,
                        item.distribution_package_id,
                        item.package_digest,
                    ),
                )
            ),
            direct_third_party_distributions=direct_third_party,
            tier_violations=(),
        )
        return graph.with_content_identity()

    def _direct_agents_from_roster(
        self,
        effective_roster: EffectiveRoster,
    ) -> list[RuntimeGraphAgentRef]:
        agents: list[RuntimeGraphAgentRef] = []
        for entry in effective_roster.entries:
            if not entry.effective_enablement:
                continue
            agents.append(
                RuntimeGraphAgentRef(
                    logical_agent_id=entry.logical_agent_id,
                    distribution_package_id=entry.distribution_package_id,
                    package_digest=entry.package_digest,
                )
            )
        return agents

    def _validate_direct_agents_against_lock(
        self,
        direct_agents: Sequence[RuntimeGraphAgentRef],
        lock: MaterializedRuntimeLock,
    ) -> None:
        closure_by_id = _lock_agent_closure_index(lock)
        packages_by_name = _lock_package_index(lock)
        for agent in direct_agents:
            closure_entry = closure_by_id.get(agent.distribution_package_id)
            if closure_entry is None:
                raise CandidateRuntimeGraphError(
                    f"direct agent {agent.distribution_package_id} missing from lock closure"
                )
            if closure_entry.package_digest != agent.package_digest:
                raise CandidateRuntimeGraphError(
                    f"direct agent digest mismatch for {agent.distribution_package_id}"
                )
            package_key = normalize_distribution_name(agent.distribution_package_id)
            if package_key not in packages_by_name:
                raise CandidateRuntimeGraphError(
                    f"direct agent package {agent.distribution_package_id} missing from lock packages"
                )

    def _metadata_for_agent(
        self,
        *,
        agent: RuntimeGraphAgentRef,
        agent_metadata_refs: Mapping[str, str],
    ) -> AgentProjectMetadata:
        metadata_ref = agent_metadata_refs.get(agent.distribution_package_id)
        if metadata_ref is None:
            raise CandidateRuntimeGraphError(
                f"missing agent_project_metadata_ref for {agent.distribution_package_id}"
            )
        metadata = self._metadata_provider.get_metadata(metadata_ref)
        if metadata is None:
            raise CandidateRuntimeGraphError(
                f"unresolved agent metadata ref {metadata_ref}"
            )
        if metadata.distribution_package_id != agent.distribution_package_id:
            raise CandidateRuntimeGraphError(
                f"metadata package line {metadata.distribution_package_id} "
                f"conflicts with graph agent {agent.distribution_package_id}"
            )
        return metadata

    def _traverse_transitive_agents(
        self,
        *,
        direct_agents: Sequence[RuntimeGraphAgentRef],
        agent_metadata_refs: Mapping[str, str],
        closure_by_id: dict[str, MaterializedAgentClosureEntry],
        packages_by_name: dict[str, MaterializedLockPackage],
    ) -> tuple[list[RuntimeGraphAgentRef], None]:
        direct_keys = {
            normalize_distribution_name(agent.distribution_package_id)
            for agent in direct_agents
        }
        visit_state: dict[str, GraphVisitState] = {
            key: GraphVisitState.UNVISITED for key in direct_keys
        }
        transitive: list[RuntimeGraphAgentRef] = []
        stack: list[str] = []

        def ensure_agent_in_lock(dist_name: str) -> MaterializedAgentClosureEntry:
            if not is_agent_distribution(dist_name):
                raise CandidateRuntimeGraphError(
                    f"undeclared Tier-2 agent dependency {dist_name}"
                )
            closure_entry = closure_by_id.get(dist_name)
            if closure_entry is None:
                raise CandidateRuntimeGraphError(
                    f"undeclared Tier-2 agent {dist_name} absent from lock closure"
                )
            package_key = normalize_distribution_name(dist_name)
            if package_key not in packages_by_name:
                raise CandidateRuntimeGraphError(
                    f"agent package {dist_name} missing from lock packages"
                )
            return closure_entry

        def dfs(agent: RuntimeGraphAgentRef) -> None:
            key = normalize_distribution_name(agent.distribution_package_id)
            state = visit_state.get(key, GraphVisitState.UNVISITED)
            if state is GraphVisitState.VISITING:
                raise CandidateRuntimeGraphError(
                    format_agent_dependency_cycle(stack, agent.distribution_package_id)
                )
            if state is GraphVisitState.VISITED:
                return

            visit_state[key] = GraphVisitState.VISITING
            stack.append(agent.distribution_package_id)
            metadata = self._metadata_for_agent(
                agent=agent,
                agent_metadata_refs=agent_metadata_refs,
            )

            for dep in metadata.dependencies:
                dep_name = parse_dependency_name(dep)
                if is_platform_dependency(dep_name):
                    continue
                if is_application_distribution(dep_name):
                    raise CandidateRuntimeGraphError(
                        f"AGENT_TIER_VIOLATION: agent {agent.distribution_package_id} "
                        f"depends on forbidden application distribution {dep_name}"
                    )
                if not is_agent_distribution(dep_name):
                    continue

                closure_entry = ensure_agent_in_lock(dep_name)
                dep_key = normalize_distribution_name(dep_name)
                dep_state = visit_state.get(dep_key, GraphVisitState.UNVISITED)
                if dep_state is GraphVisitState.VISITING:
                    raise CandidateRuntimeGraphError(
                        format_agent_dependency_cycle(stack, dep_name)
                    )
                if dep_state is GraphVisitState.VISITED:
                    continue

                transitive_agent = RuntimeGraphAgentRef(
                    logical_agent_id=dep_name,
                    distribution_package_id=dep_name,
                    package_digest=closure_entry.package_digest,
                )
                if dep_key not in direct_keys and all(
                    normalize_distribution_name(item.distribution_package_id) != dep_key
                    for item in transitive
                ):
                    transitive.append(transitive_agent)
                dfs(transitive_agent)

            stack.pop()
            visit_state[key] = GraphVisitState.VISITED

        for direct in direct_agents:
            dfs(direct)

        return transitive, None

    def _direct_third_party_from_declaration(
        self,
        *,
        repository_declaration: RepositoryDependencyDeclaration,
        packages_by_name: dict[str, MaterializedLockPackage],
    ) -> tuple[RuntimeGraphThirdPartyRef, ...]:
        third_party: list[RuntimeGraphThirdPartyRef] = []
        seen: set[str] = set()
        for dep in repository_declaration.direct_dependencies:
            dep_name = parse_dependency_name(dep)
            if is_platform_dependency(dep_name) or is_agent_distribution(dep_name):
                continue
            if is_application_distribution(dep_name):
                raise CandidateRuntimeGraphError(
                    f"APPLICATION_TIER_VIOLATION: application depends on {dep_name}"
                )
            key = normalize_distribution_name(dep_name)
            if key in seen:
                continue
            seen.add(key)
            resolved = packages_by_name.get(key)
            if resolved is None:
                raise CandidateRuntimeGraphError(
                    f"unresolved required third-party dependency {dep_name}"
                )
            third_party.append(
                RuntimeGraphThirdPartyRef(
                    distribution_name=resolved.distribution_name,
                    version=resolved.version,
                )
            )
        return tuple(
            sorted(
                third_party,
                key=lambda item: (normalize_distribution_name(item.distribution_name), item.version),
            )
        )


class CandidateRuntimeGraphValidator:
    """Fail-closed simulation gate over lock, roster, and candidate graph."""

    def validate(
        self,
        *,
        lock: MaterializedRuntimeLock,
        effective_roster: EffectiveRoster,
        graph: CandidateApplicationRuntimeGraph,
    ) -> CandidateApplicationRuntimeGraph:
        if lock.lock_id is None:
            raise CandidateRuntimeGraphError("lock must have content identity")
        if graph.materialized_runtime_lock_id != lock.lock_id:
            raise CandidateRuntimeGraphError(
                "graph materialized_runtime_lock_id does not match lock"
            )
        if graph.application_id != effective_roster.application_id:
            raise CandidateRuntimeGraphError(
                "graph application_id does not match effective roster"
            )

        identity = graph.with_content_identity()
        if identity.runtime_graph_digest != graph.runtime_graph_digest:
            raise CandidateRuntimeGraphError("runtime_graph_digest is not content-addressed")

        closure_by_id = _lock_agent_closure_index(lock)
        packages_by_name = _lock_package_index(lock)

        enabled_roster = {
            entry.logical_agent_id: entry
            for entry in effective_roster.entries
            if entry.effective_enablement
        }
        direct_by_logical = {agent.logical_agent_id: agent for agent in graph.direct_agents}

        for logical_id, roster_entry in enabled_roster.items():
            graph_agent = direct_by_logical.get(logical_id)
            if graph_agent is None:
                raise CandidateRuntimeGraphError(
                    f"enabled roster agent {logical_id} missing from direct_agents"
                )
            if graph_agent.package_digest != roster_entry.package_digest:
                raise CandidateRuntimeGraphError(
                    f"roster digest mismatch for logical agent {logical_id}"
                )

        for agent in (*graph.direct_agents, *graph.transitive_agents):
            closure_entry = closure_by_id.get(agent.distribution_package_id)
            if closure_entry is None:
                raise CandidateRuntimeGraphError(
                    f"graph agent {agent.distribution_package_id} absent from lock closure"
                )
            if closure_entry.package_digest != agent.package_digest:
                raise CandidateRuntimeGraphError(
                    f"graph agent digest mismatch for {agent.distribution_package_id}"
                )
            package_key = normalize_distribution_name(agent.distribution_package_id)
            if package_key not in packages_by_name:
                raise CandidateRuntimeGraphError(
                    f"graph agent package {agent.distribution_package_id} missing from lock"
                )

        if graph.tier_violations:
            raise CandidateRuntimeGraphError("graph contains tier violations")

        return graph
