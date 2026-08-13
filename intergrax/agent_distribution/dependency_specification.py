# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic L1 + L2 → L3 candidate dependency specification builder (AP-6)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    InstalledAgentRequirementSet,
    PolicyDependencyConstraint,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.errors import DependencySpecificationError


def _canonicalize_platform_extras(platform_extras: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in platform_extras if item.strip()}))


def _canonicalize_policy_constraints(
    policy_constraints: Sequence[PolicyDependencyConstraint],
) -> tuple[PolicyDependencyConstraint, ...]:
    return tuple(
        sorted(
            policy_constraints,
            key=lambda item: (item.constraint_kind, item.constraint_value),
        )
    )


def build_candidate_dependency_specification(
    *,
    repository_declaration: RepositoryDependencyDeclaration,
    installed_agent_requirement_set: InstalledAgentRequirementSet,
    platform_version: str,
    platform_extras: Sequence[str] = (),
    policy_constraints: Sequence[PolicyDependencyConstraint] = (),
    repository_lock_hint_ref: str | None = None,
) -> CandidateDependencySpecification:
    """Merge L1 repository declaration with L2 installed agent requirements (§15.3)."""
    normalized_platform_version = platform_version.strip()
    if not normalized_platform_version:
        raise DependencySpecificationError("platform_version must be non-empty")

    return CandidateDependencySpecification(
        application_release_id=repository_declaration.application_release_id,
        platform_version=normalized_platform_version,
        repository_declaration=repository_declaration,
        agent_packages=installed_agent_requirement_set.agent_packages,
        platform_extras=_canonicalize_platform_extras(platform_extras),
        policy_constraints=_canonicalize_policy_constraints(policy_constraints),
        repository_lock_hint_ref=(
            repository_lock_hint_ref.strip()
            if repository_lock_hint_ref is not None and repository_lock_hint_ref.strip()
            else None
        ),
    )
