# © Artur Czarnecki. All rights reserved.

"""AP-7 candidate runtime graph builder and validation gate tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    DependencyResolverInput,
    InstalledAgentPackageRequirement,
    LockPackageRole,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.errors import CandidateRuntimeGraphError
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
    CandidateRuntimeGraphValidator,
)
from intergrax.agent_distribution.runtime_lock import MaterializedRuntimeLockProducer
from intergrax.agent_distribution.resolver import ResolvedDependencyClosure

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_DIGEST_C = "sha256:" + ("c" * 64)
_AGENT_A = "intergrax-local-search-agent"
_AGENT_B = "intergrax-local-indexer-agent"
_APP_ID = "local_workspace_application"


class _InMemoryMetadataProvider:
    def __init__(self, records: dict[str, AgentProjectMetadata]) -> None:
        self._records = records

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        return self._records.get(metadata_ref)


def _lock_with_agents(
    *,
    agents: tuple[tuple[str, str], ...],
    third_party: tuple[MaterializedLockPackage, ...] = (
        MaterializedLockPackage(distribution_name="requests", version="2.32.0"),
    ),
) -> MaterializedRuntimeLock:
    spec = CandidateDependencySpecification(
        application_release_id="rel-ap7",
        platform_version="0.1.0",
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=("requests>=2.32", "Intergrax-ai"),
        ),
        agent_packages=tuple(
            InstalledAgentPackageRequirement(
                distribution_package_id=package_id,
                package_digest=digest,
                agent_project_metadata_ref=f"meta://{package_id}",
            )
            for package_id, digest in agents
        ),
    )
    resolver_input = DependencyResolverInput(
        specification=spec,
        resolver_algorithm_id="intergrax.test-resolver",
        resolver_algorithm_version="1.0.0",
    )
    packages = list(third_party)
    for package_id, digest in agents:
        packages.append(
            MaterializedLockPackage(
                distribution_name=package_id,
                version="1.0.0",
                package_digest=digest,
            )
        )
    resolved = ResolvedDependencyClosure(
        resolver_algorithm_id="intergrax.test-resolver",
        resolver_algorithm_version="1.0.0",
        python_version="3.12",
        packages=tuple(packages),
        transitive_agent_closure=tuple(
            MaterializedAgentClosureEntry(
                distribution_package_id=package_id,
                package_digest=digest,
                role=LockPackageRole.TRANSITIVE,
            )
            for package_id, digest in agents[1:]
        ),
    )
    return MaterializedRuntimeLockProducer().produce(resolver_input, resolved)


def _roster(
    *,
    entries: tuple[EffectiveRosterEntry, ...],
) -> EffectiveRoster:
    roster = EffectiveRoster(
        application_id=_APP_ID,
        application_environment_id="env-prod",
        manifest_release_id="rel-ap7",
        binding_revisions=(1,),
        entries=entries,
    )
    return roster.with_revision_id()


def test_valid_lock_and_roster_build_candidate_graph() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {
            f"meta://{_AGENT_A}": AgentProjectMetadata(
                distribution_package_id=_AGENT_A,
                dependencies=("Intergrax-ai",),
            ),
        }
    )
    graph = CandidateRuntimeGraphBuilder(provider).build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=("requests>=2.32", "Intergrax-ai"),
        ),
        agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
    )
    validated = CandidateRuntimeGraphValidator().validate(
        lock=lock,
        effective_roster=roster,
        graph=graph,
    )
    assert validated.materialized_runtime_lock_id == lock.lock_id
    assert len(validated.direct_agents) == 1
    assert validated.direct_agents[0].logical_agent_id == "search"


def test_disabled_roster_agent_not_required_in_direct_agents() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
            EffectiveRosterEntry(
                logical_agent_id="indexer",
                installation_slot_id="slot-indexer",
                package_digest=_DIGEST_B,
                distribution_package_id=_AGENT_B,
                effective_enablement=False,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {
            f"meta://{_AGENT_A}": AgentProjectMetadata(
                distribution_package_id=_AGENT_A,
                dependencies=(),
            ),
        }
    )
    graph = CandidateRuntimeGraphBuilder(provider).build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=("requests>=2.32",),
        ),
        agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
    )
    assert len(graph.direct_agents) == 1


def test_graph_digest_is_deterministic() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {f"meta://{_AGENT_A}": AgentProjectMetadata(distribution_package_id=_AGENT_A, dependencies=())}
    )
    builder = CandidateRuntimeGraphBuilder(provider)
    declaration = RepositoryDependencyDeclaration(
        application_release_id="rel-ap7",
        direct_dependencies=("requests>=2.32",),
    )
    first = builder.build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=declaration,
        agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
    )
    second = builder.build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=declaration,
        agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
    )
    assert first.runtime_graph_digest == second.runtime_graph_digest


def test_graph_agent_absent_from_lock_fails_closed() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {
            f"meta://{_AGENT_A}": AgentProjectMetadata(
                distribution_package_id=_AGENT_A,
                dependencies=(_AGENT_B,),
            ),
        }
    )
    with pytest.raises(CandidateRuntimeGraphError, match="undeclared Tier-2 agent"):
        CandidateRuntimeGraphBuilder(provider).build(
            lock=lock,
            effective_roster=roster,
            repository_declaration=RepositoryDependencyDeclaration(
                application_release_id="rel-ap7",
                direct_dependencies=("requests>=2.32",),
            ),
            agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
        )


def test_roster_digest_mismatch_fails_closed() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_B,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {f"meta://{_AGENT_A}": AgentProjectMetadata(distribution_package_id=_AGENT_A, dependencies=())}
    )
    with pytest.raises(CandidateRuntimeGraphError, match="digest mismatch"):
        CandidateRuntimeGraphBuilder(provider).build(
            lock=lock,
            effective_roster=roster,
            repository_declaration=RepositoryDependencyDeclaration(
                application_release_id="rel-ap7",
                direct_dependencies=("requests>=2.32",),
            ),
            agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
        )


def test_tier_violation_agent_depends_on_application_fails_closed() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {
            f"meta://{_AGENT_A}": AgentProjectMetadata(
                distribution_package_id=_AGENT_A,
                dependencies=("intergrax-demo-application",),
            ),
        }
    )
    with pytest.raises(CandidateRuntimeGraphError, match="AGENT_TIER_VIOLATION"):
        CandidateRuntimeGraphBuilder(provider).build(
            lock=lock,
            effective_roster=roster,
            repository_declaration=RepositoryDependencyDeclaration(
                application_release_id="rel-ap7",
                direct_dependencies=(),
            ),
            agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
        )


def test_cycle_detection_fails_closed() -> None:
    lock = _lock_with_agents(
        agents=(
            (_AGENT_A, _DIGEST_A),
            (_AGENT_B, _DIGEST_B),
        ),
    )
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {
            f"meta://{_AGENT_A}": AgentProjectMetadata(
                distribution_package_id=_AGENT_A,
                dependencies=(_AGENT_B,),
            ),
            f"meta://{_AGENT_B}": AgentProjectMetadata(
                distribution_package_id=_AGENT_B,
                dependencies=(_AGENT_A,),
            ),
        }
    )
    with pytest.raises(CandidateRuntimeGraphError, match="AGENT_DEPENDENCY_CYCLE"):
        CandidateRuntimeGraphBuilder(provider).build(
            lock=lock,
            effective_roster=roster,
            repository_declaration=RepositoryDependencyDeclaration(
                application_release_id="rel-ap7",
                direct_dependencies=(),
            ),
            agent_metadata_refs={
                _AGENT_A: f"meta://{_AGENT_A}",
                _AGENT_B: f"meta://{_AGENT_B}",
            },
        )


def test_transitive_agent_from_authoritative_metadata() -> None:
    lock = _lock_with_agents(
        agents=(
            (_AGENT_A, _DIGEST_A),
            (_AGENT_B, _DIGEST_B),
        ),
    )
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {
            f"meta://{_AGENT_A}": AgentProjectMetadata(
                distribution_package_id=_AGENT_A,
                dependencies=(_AGENT_B,),
            ),
            f"meta://{_AGENT_B}": AgentProjectMetadata(
                distribution_package_id=_AGENT_B,
                dependencies=(),
            ),
        }
    )
    graph = CandidateRuntimeGraphBuilder(provider).build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=(),
        ),
        agent_metadata_refs={
            _AGENT_A: f"meta://{_AGENT_A}",
            _AGENT_B: f"meta://{_AGENT_B}",
        },
    )
    assert len(graph.transitive_agents) == 1
    assert graph.transitive_agents[0].distribution_package_id == _AGENT_B


def test_validator_rejects_foreign_application_id() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {f"meta://{_AGENT_A}": AgentProjectMetadata(distribution_package_id=_AGENT_A, dependencies=())}
    )
    graph = CandidateRuntimeGraphBuilder(provider).build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=(),
        ),
        agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
    )
    foreign_graph = graph.model_copy(update={"application_id": "foreign-app"})
    with pytest.raises(CandidateRuntimeGraphError, match="application_id"):
        CandidateRuntimeGraphValidator().validate(
            lock=lock,
            effective_roster=roster,
            graph=foreign_graph,
        )


def test_validator_rejects_wrong_lock_binding() -> None:
    lock = _lock_with_agents(agents=((_AGENT_A, _DIGEST_A),))
    roster = _roster(
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        )
    )
    provider = _InMemoryMetadataProvider(
        {f"meta://{_AGENT_A}": AgentProjectMetadata(distribution_package_id=_AGENT_A, dependencies=())}
    )
    graph = CandidateRuntimeGraphBuilder(provider).build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=(),
        ),
        agent_metadata_refs={_AGENT_A: f"meta://{_AGENT_A}"},
    )
    tampered = graph.model_copy(update={"materialized_runtime_lock_id": "sha256:" + ("f" * 64)})
    with pytest.raises(CandidateRuntimeGraphError, match="does not match lock"):
        CandidateRuntimeGraphValidator().validate(
            lock=lock,
            effective_roster=roster,
            graph=tampered,
        )


def test_agent_distribution_package_has_no_forbidden_imports() -> None:
    repo = Path(__file__).resolve().parents[3]
    package_root = repo / "intergrax" / "agent_distribution"
    agent_dirs = {
        p.name
        for p in (repo / "agents").iterdir()
        if p.is_dir() and (p / "__init__.py").is_file() and not p.name.startswith("_")
    }
    app_dirs = {
        p.name
        for p in (repo / "applications").iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    }
    violations: list[str] = []
    forbidden_tokens = ("subprocess", "uv ", "pip ")
    for path in package_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        for token in forbidden_tokens:
            if token.strip() in source and path.name in {
                "resolver.py",
                "runtime_lock.py",
                "runtime_graph_service.py",
            }:
                pass
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.append(node.module)
            for module in modules:
                top = module.split(".", 1)[0]
                if top == "agents" or top in agent_dirs:
                    violations.append(f"{path.relative_to(repo)} imports {module}")
                if top == "applications" or top in app_dirs:
                    violations.append(f"{path.relative_to(repo)} imports {module}")
    assert not violations, "\n".join(violations)
