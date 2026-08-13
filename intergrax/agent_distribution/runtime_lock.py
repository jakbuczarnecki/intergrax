# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Materialized runtime lock producer (AP-7 §16)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from intergrax.agent_distribution.dependency import (
    DependencyResolverInput,
    InstalledAgentPackageRequirement,
    LockPackageRole,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedLockReproducibilityEvidence,
    MaterializedLockRollbackEvidence,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.errors import (
    DependencyResolutionError,
    MaterializedRuntimeLockError,
)
from intergrax.agent_distribution.resolver import DependencyResolver, ResolvedDependencyClosure
from intergrax.runtime.attestation.canonical_json import stable_payload_hash


def _normalize_distribution_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _canonicalize_packages(
    packages: Sequence[MaterializedLockPackage],
) -> tuple[MaterializedLockPackage, ...]:
    return tuple(
        sorted(
            packages,
            key=lambda item: (
                _normalize_distribution_name(item.distribution_name),
                item.version,
                item.package_digest or "",
                item.dependency_of or "",
            ),
        )
    )


def _canonicalize_agent_closure(
    entries: Sequence[MaterializedAgentClosureEntry],
) -> tuple[MaterializedAgentClosureEntry, ...]:
    return tuple(
        sorted(
            entries,
            key=lambda item: (
                item.distribution_package_id,
                item.package_digest,
                item.role.value,
            ),
        )
    )


def _canonicalize_platform_extras(extras: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in extras if item.strip()}))


def _repository_lock_hint_digest(hint_ref: str | None) -> str | None:
    if hint_ref is None:
        return None
    return stable_payload_hash({"repository_lock_hint_ref": hint_ref.strip()})


def _validate_packages(packages: Sequence[MaterializedLockPackage]) -> None:
    by_name: dict[str, MaterializedLockPackage] = {}
    for package in packages:
        key = _normalize_distribution_name(package.distribution_name)
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = package
            continue
        if existing.version != package.version:
            raise DependencyResolutionError(
                f"conflicting resolved versions for {package.distribution_name}: "
                f"{existing.version} vs {package.version}"
            )
        if (
            existing.package_digest is not None
            and package.package_digest is not None
            and existing.package_digest != package.package_digest
        ):
            raise DependencyResolutionError(
                f"conflicting digests for {package.distribution_name}@{package.version}"
            )


def _validate_resolver_identity(
    resolver_input: DependencyResolverInput,
    resolved: ResolvedDependencyClosure,
) -> None:
    if resolved.resolver_algorithm_id != resolver_input.resolver_algorithm_id:
        raise DependencyResolutionError("resolver_algorithm_id mismatch")
    if resolved.resolver_algorithm_version != resolver_input.resolver_algorithm_version:
        raise DependencyResolutionError("resolver_algorithm_version mismatch")


def _validate_direct_agent_roots(
    *,
    required_agents: Sequence[InstalledAgentPackageRequirement],
    packages: Sequence[MaterializedLockPackage],
    agent_closure: Sequence[MaterializedAgentClosureEntry],
) -> None:
    package_ids = {
        _normalize_distribution_name(package.distribution_name) for package in packages
    }
    closure_by_id = {entry.distribution_package_id: entry for entry in agent_closure}

    for requirement in required_agents:
        package_key = _normalize_distribution_name(requirement.distribution_package_id)
        if package_key not in package_ids:
            raise DependencyResolutionError(
                f"missing direct agent package {requirement.distribution_package_id} "
                "from resolved package closure"
            )
        closure_entry = closure_by_id.get(requirement.distribution_package_id)
        if closure_entry is None:
            raise DependencyResolutionError(
                f"missing direct agent root {requirement.distribution_package_id} "
                "from agent closure"
            )
        if closure_entry.role is not LockPackageRole.DIRECT:
            raise DependencyResolutionError(
                f"agent root {requirement.distribution_package_id} must have role=direct"
            )
        if closure_entry.package_digest != requirement.package_digest:
            raise DependencyResolutionError(
                f"agent digest mismatch for {requirement.distribution_package_id}"
            )
        package_match = next(
            (
                package
                for package in packages
                if _normalize_distribution_name(package.distribution_name)
                == package_key
            ),
            None,
        )
        if package_match is not None and package_match.package_digest is not None:
            if package_match.package_digest != requirement.package_digest:
                raise DependencyResolutionError(
                    f"agent digest mismatch for {requirement.distribution_package_id}"
                )


def _merge_agent_closure(
    *,
    direct_agents: Sequence[InstalledAgentPackageRequirement],
    transitive_agents: Sequence[MaterializedAgentClosureEntry],
) -> tuple[MaterializedAgentClosureEntry, ...]:
    merged: dict[str, MaterializedAgentClosureEntry] = {}
    for requirement in direct_agents:
        entry = MaterializedAgentClosureEntry(
            distribution_package_id=requirement.distribution_package_id,
            package_digest=requirement.package_digest,
            role=LockPackageRole.DIRECT,
        )
        existing = merged.get(entry.distribution_package_id)
        if existing is not None and existing.package_digest != entry.package_digest:
            raise DependencyResolutionError(
                f"conflicting digests for direct agent {entry.distribution_package_id}"
            )
        merged[entry.distribution_package_id] = entry

    for entry in transitive_agents:
        if entry.role is LockPackageRole.DIRECT:
            raise DependencyResolutionError(
                f"resolver must not emit direct role for {entry.distribution_package_id}"
            )
        existing = merged.get(entry.distribution_package_id)
        if existing is not None:
            if existing.package_digest != entry.package_digest:
                raise DependencyResolutionError(
                    f"conflicting digests for agent {entry.distribution_package_id}"
                )
            if existing.role is LockPackageRole.DIRECT:
                continue
        merged[entry.distribution_package_id] = entry

    return _canonicalize_agent_closure(merged.values())


class MaterializedRuntimeLockProducer:
    """Build immutable runtime locks from resolver input and output (§16)."""

    def produce(
        self,
        resolver_input: DependencyResolverInput,
        resolved: ResolvedDependencyClosure,
        *,
        created_at: datetime | None = None,
        reproducibility_evidence: MaterializedLockReproducibilityEvidence | None = None,
        rollback_evidence: MaterializedLockRollbackEvidence | None = None,
    ) -> MaterializedRuntimeLock:
        """Assemble and validate one materialized runtime lock artifact."""
        _validate_resolver_identity(resolver_input, resolved)
        spec = resolver_input.specification
        packages = _canonicalize_packages(resolved.packages)
        _validate_packages(packages)

        direct_agents = tuple(
            sorted(
                spec.agent_packages,
                key=lambda item: (
                    item.distribution_package_id,
                    item.package_digest,
                    item.agent_project_metadata_ref,
                ),
            )
        )
        agent_closure = _merge_agent_closure(
            direct_agents=direct_agents,
            transitive_agents=resolved.transitive_agent_closure,
        )
        _validate_direct_agent_roots(
            required_agents=direct_agents,
            packages=packages,
            agent_closure=agent_closure,
        )

        hint_digest = _repository_lock_hint_digest(spec.repository_lock_hint_ref)
        lock = MaterializedRuntimeLock(
            resolver_algorithm_id=resolver_input.resolver_algorithm_id,
            resolver_algorithm_version=resolver_input.resolver_algorithm_version,
            created_at=created_at,
            inputs_digest=resolver_input.inputs_digest(),
            intergrax_version=spec.platform_version,
            python_version=resolved.python_version,
            platform_extras=_canonicalize_platform_extras(spec.platform_extras),
            packages=packages,
            agent_closure=agent_closure,
            repository_lock_hint_digest=hint_digest,
            reproducibility_evidence=reproducibility_evidence,
            rollback_evidence=rollback_evidence,
        )
        if lock.inputs_digest != resolver_input.inputs_digest():
            raise MaterializedRuntimeLockError("inputs_digest authority mismatch")
        return lock.with_content_identity()


class MaterializedRuntimeLockService:
    """Resolve and produce one content-addressed runtime lock."""

    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver
        self._producer = MaterializedRuntimeLockProducer()

    def produce_lock(
        self,
        resolver_input: DependencyResolverInput,
        *,
        created_at: datetime | None = None,
        reproducibility_evidence: MaterializedLockReproducibilityEvidence | None = None,
        rollback_evidence: MaterializedLockRollbackEvidence | None = None,
    ) -> MaterializedRuntimeLock:
        resolved = self._resolver.resolve(resolver_input)
        return self._producer.produce(
            resolver_input,
            resolved,
            created_at=created_at,
            reproducibility_evidence=reproducibility_evidence,
            rollback_evidence=rollback_evidence,
        )
