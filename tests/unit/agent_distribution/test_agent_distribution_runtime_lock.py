# © Artur Czarnecki. All rights reserved.

"""AP-7 materialized runtime lock producer and store tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    DependencyResolverInput,
    InstalledAgentPackageRequirement,
    LockPackageRole,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedLockReproducibilityEvidence,
    MaterializedLockRollbackEvidence,
    MaterializedRuntimeLock,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.errors import (
    DependencyResolutionError,
    MaterializedRuntimeLockConflict,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryMaterializedRuntimeLockStore,
)
from intergrax.agent_distribution.resolver import (
    CallableDependencyResolver,
    ResolvedDependencyClosure,
)
from intergrax.agent_distribution.runtime_lock import (
    MaterializedRuntimeLockProducer,
    MaterializedRuntimeLockService,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_DIGEST_C = "sha256:" + ("c" * 64)
_AGENT_A = "intergrax-local-search-agent"
_AGENT_B = "intergrax-local-indexer-agent"
_RESOLVER_ID = "intergrax.test-resolver"
_RESOLVER_VERSION = "1.0.0"


def _spec(
    *,
    agent_packages: tuple[InstalledAgentPackageRequirement, ...] | None = None,
    platform_extras: tuple[str, ...] = ("dev",),
    repository_lock_hint_ref: str | None = "lock://uv/workspace",
) -> CandidateDependencySpecification:
    return CandidateDependencySpecification(
        application_release_id="rel-ap7",
        platform_version="0.1.0",
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-ap7",
            direct_dependencies=("requests>=2.32",),
        ),
        agent_packages=agent_packages
        or (
            InstalledAgentPackageRequirement(
                distribution_package_id=_AGENT_A,
                package_digest=_DIGEST_A,
                agent_project_metadata_ref="meta://search",
            ),
        ),
        platform_extras=platform_extras,
        repository_lock_hint_ref=repository_lock_hint_ref,
    )


def _resolver_input(
    spec: CandidateDependencySpecification | None = None,
) -> DependencyResolverInput:
    return DependencyResolverInput(
        specification=spec or _spec(),
        resolver_algorithm_id=_RESOLVER_ID,
        resolver_algorithm_version=_RESOLVER_VERSION,
    )


def _resolved_closure(
    *,
    packages: tuple[MaterializedLockPackage, ...] | None = None,
    transitive_agent_closure: tuple[MaterializedAgentClosureEntry, ...] = (),
) -> ResolvedDependencyClosure:
    default_packages = (
        MaterializedLockPackage(
            distribution_name=_AGENT_A,
            version="1.0.0",
            package_digest=_DIGEST_A,
        ),
        MaterializedLockPackage(
            distribution_name="requests",
            version="2.32.0",
            dependency_of=_AGENT_A,
        ),
    )
    return ResolvedDependencyClosure(
        resolver_algorithm_id=_RESOLVER_ID,
        resolver_algorithm_version=_RESOLVER_VERSION,
        python_version="3.12",
        packages=packages if packages is not None else default_packages,
        transitive_agent_closure=transitive_agent_closure,
    )


def _produce_lock(
    resolver_input: DependencyResolverInput | None = None,
    resolved: ResolvedDependencyClosure | None = None,
    **kwargs: object,
) -> MaterializedRuntimeLock:
    producer = MaterializedRuntimeLockProducer()
    return producer.produce(
        resolver_input or _resolver_input(),
        resolved or _resolved_closure(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_valid_resolver_input_produces_materialized_runtime_lock() -> None:
    lock = _produce_lock()
    assert lock.lock_id is not None
    assert lock.lock_digest == lock.lock_id
    assert lock.inputs_digest == _resolver_input().inputs_digest()
    assert lock.resolver_algorithm_id == _RESOLVER_ID
    assert lock.resolver_algorithm_version == _RESOLVER_VERSION
    assert lock.agent_closure[0].role is LockPackageRole.DIRECT
    assert lock.agent_closure[0].package_digest == _DIGEST_A
    assert len(lock.packages) == 2


def test_lock_inputs_digest_matches_resolver_input_authority() -> None:
    resolver_input = _resolver_input()
    lock = _produce_lock(resolver_input)
    assert lock.inputs_digest == resolver_input.inputs_digest()


def test_direct_agent_roots_preserve_exact_digest() -> None:
    lock = _produce_lock()
    direct = [entry for entry in lock.agent_closure if entry.role is LockPackageRole.DIRECT]
    assert len(direct) == 1
    assert direct[0].distribution_package_id == _AGENT_A
    assert direct[0].package_digest == _DIGEST_A


def test_permuted_resolver_package_ordering_yields_same_lock_digest() -> None:
    packages = (
        MaterializedLockPackage(distribution_name="requests", version="2.32.0"),
        MaterializedLockPackage(
            distribution_name=_AGENT_A,
            version="1.0.0",
            package_digest=_DIGEST_A,
        ),
    )
    first = _produce_lock(resolved=_resolved_closure(packages=packages))
    second = _produce_lock(
        resolved=_resolved_closure(
            packages=(
                packages[1],
                packages[0],
            )
        )
    )
    assert first.lock_digest == second.lock_digest


def test_different_created_at_yields_same_lock_digest() -> None:
    first = _produce_lock(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second = _produce_lock(
        created_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )
    assert first.lock_digest == second.lock_digest
    assert first.created_at != second.created_at


def test_duplicate_package_conflicting_version_fails_closed() -> None:
    packages = (
        MaterializedLockPackage(distribution_name="requests", version="2.32.0"),
        MaterializedLockPackage(distribution_name="requests", version="2.31.0"),
        MaterializedLockPackage(
            distribution_name=_AGENT_A,
            version="1.0.0",
            package_digest=_DIGEST_A,
        ),
    )
    with pytest.raises(DependencyResolutionError, match="conflicting resolved versions"):
        _produce_lock(resolved=_resolved_closure(packages=packages))


def test_duplicate_package_conflicting_digest_fails_closed() -> None:
    packages = (
        MaterializedLockPackage(
            distribution_name=_AGENT_A,
            version="1.0.0",
            package_digest=_DIGEST_A,
        ),
        MaterializedLockPackage(
            distribution_name=_AGENT_A,
            version="1.0.0",
            package_digest=_DIGEST_B,
        ),
    )
    with pytest.raises(DependencyResolutionError, match="conflicting digests"):
        _produce_lock(resolved=_resolved_closure(packages=packages))


def test_missing_direct_agent_root_fails_closed() -> None:
    packages = (
        MaterializedLockPackage(distribution_name="requests", version="2.32.0"),
    )
    with pytest.raises(DependencyResolutionError, match="missing direct agent"):
        _produce_lock(resolved=_resolved_closure(packages=packages))


def test_agent_digest_mismatch_fails_closed() -> None:
    resolved = ResolvedDependencyClosure(
        resolver_algorithm_id=_RESOLVER_ID,
        resolver_algorithm_version=_RESOLVER_VERSION,
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name=_AGENT_A,
                version="1.0.0",
                package_digest=_DIGEST_B,
            ),
        ),
        transitive_agent_closure=(),
    )
    with pytest.raises(DependencyResolutionError, match="agent digest mismatch"):
        _produce_lock(resolved=resolved)


def test_resolver_identity_mismatch_fails_closed() -> None:
    resolved = _resolved_closure().model_copy(
        update={"resolver_algorithm_version": "9.9.9"},
    )
    with pytest.raises(DependencyResolutionError, match="resolver_algorithm_version mismatch"):
        _produce_lock(resolved=resolved)


def test_lock_store_idempotent_persist() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryMaterializedRuntimeLockStore(state)
    lock = _produce_lock()
    first = store.persist_lock(lock)
    second = store.persist_lock(lock)
    assert first.lock_id == second.lock_id
    assert len(state.locks) == 1


def test_lock_store_identity_collision_with_different_content_fails_closed() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryMaterializedRuntimeLockStore(state)
    lock = _produce_lock()
    store.persist_lock(lock)

    other = _produce_lock(
        resolved=_resolved_closure(
            packages=(
                MaterializedLockPackage(
                    distribution_name=_AGENT_A,
                    version="9.9.9",
                    package_digest=_DIGEST_A,
                ),
            )
        )
    )
    tampered = other.model_copy(update={"lock_id": lock.lock_id, "lock_digest": lock.lock_digest})
    with pytest.raises(MaterializedRuntimeLockConflict):
        store.persist_lock(tampered)


def test_service_resolves_and_produces_lock() -> None:
    service = MaterializedRuntimeLockService(
        CallableDependencyResolver(lambda _input: _resolved_closure())
    )
    lock = service.produce_lock(
        _resolver_input(),
        reproducibility_evidence=MaterializedLockReproducibilityEvidence(
            resolver_log_ref="log://resolver/1",
            input_snapshot_ref="snapshot://input/1",
        ),
        rollback_evidence=MaterializedLockRollbackEvidence(
            supersedes_lock_id="sha256:" + ("d" * 64),
            rollback_eligible=True,
        ),
    )
    assert lock.reproducibility_evidence is not None
    assert lock.rollback_evidence is not None
    assert lock.repository_lock_hint_digest is not None


def test_platform_extras_match_specification_only() -> None:
    spec = _spec(platform_extras=("zeta", "alpha", "alpha"))
    lock = _produce_lock(_resolver_input(spec))
    assert lock.platform_extras == ("alpha", "zeta")


def test_transitive_agent_closure_merged_from_resolver() -> None:
    resolved = _resolved_closure(
        packages=(
            MaterializedLockPackage(
                distribution_name=_AGENT_A,
                version="1.0.0",
                package_digest=_DIGEST_A,
            ),
            MaterializedLockPackage(
                distribution_name=_AGENT_B,
                version="1.0.0",
                package_digest=_DIGEST_B,
            ),
        ),
        transitive_agent_closure=(
            MaterializedAgentClosureEntry(
                distribution_package_id=_AGENT_B,
                package_digest=_DIGEST_B,
                role=LockPackageRole.TRANSITIVE,
            ),
        ),
    )
    lock = _produce_lock(resolved=resolved)
    roles = {entry.distribution_package_id: entry.role for entry in lock.agent_closure}
    assert roles[_AGENT_A] is LockPackageRole.DIRECT
    assert roles[_AGENT_B] is LockPackageRole.TRANSITIVE
