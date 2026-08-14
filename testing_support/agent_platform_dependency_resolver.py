# © Artur Czarnecki. All rights reserved.

"""Deterministic test-only dependency resolver helpers for Agent Platform."""

from __future__ import annotations

from intergrax.agent_distribution.dependency import (
    DependencyResolverInput,
    MaterializedLockPackage,
)
from intergrax.agent_distribution.resolver import (
    CallableDependencyResolver,
    DependencyResolver,
    ResolvedDependencyClosure,
)


def identity_dependency_closure(
    resolver_input: DependencyResolverInput,
) -> ResolvedDependencyClosure:
    """Map pinned agent packages to deterministic lock packages for tests."""
    packages = tuple(
        MaterializedLockPackage(
            distribution_name=package.distribution_package_id,
            version="1.0.0",
            package_digest=package.package_digest,
        )
        for package in resolver_input.specification.agent_packages
    )
    return ResolvedDependencyClosure(
        resolver_algorithm_id=resolver_input.resolver_algorithm_id,
        resolver_algorithm_version=resolver_input.resolver_algorithm_version,
        python_version="3.12",
        packages=packages,
    )


def make_identity_dependency_resolver() -> DependencyResolver:
    """Explicit injected resolver for admin/build acceptance tests."""
    return CallableDependencyResolver(identity_dependency_closure)
