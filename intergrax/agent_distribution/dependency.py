# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dependency declaration and materialized runtime lock contracts (§15–§16)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution._digest import (
    content_digest_for_model,
    normalize_optional_package_digest,
    normalize_package_digest,
)

_NON_EMPTY = Field(min_length=1)

SCHEMA_REPOSITORY_DEPENDENCY_DECLARATION_V1: Final = (
    "repository_dependency_declaration.v1"
)
SCHEMA_INSTALLED_AGENT_REQUIREMENT_SET_V1: Final = "installed_agent_requirement_set.v1"
SCHEMA_CANDIDATE_DEPENDENCY_SPECIFICATION_V1: Final = (
    "candidate_dependency_specification.v1"
)
SCHEMA_DEPENDENCY_RESOLVER_INPUT_V1: Final = "dependency_resolver_input.v1"
SCHEMA_MATERIALIZED_RUNTIME_LOCK_V1: Final = "materialized_runtime_lock.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class RepositoryDependencyDeclaration(BaseModel):
    """L1 — application release dependency baseline (§15.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_REPOSITORY_DEPENDENCY_DECLARATION_V1
    application_release_id: str = _NON_EMPTY
    direct_dependencies: tuple[str, ...] = ()
    python_requires: str | None = None

    @field_validator("application_release_id", "python_requires")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class InstalledAgentPackageRequirement(BaseModel):
    """One digest-pinned agent package requirement from installation metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_package_id: str = _NON_EMPTY
    package_digest: str = _NON_EMPTY
    agent_project_metadata_ref: str = _NON_EMPTY

    @field_validator("distribution_package_id", "agent_project_metadata_ref")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)


class InstalledAgentRequirementSet(BaseModel):
    """L2 — digest-pinned agent requirements from effective roster (§15.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_INSTALLED_AGENT_REQUIREMENT_SET_V1
    effective_roster_revision_id: str = _NON_EMPTY
    agent_packages: tuple[InstalledAgentPackageRequirement, ...]

    @field_validator("effective_roster_revision_id")
    @classmethod
    def _strip_revision(cls, value: str) -> str:
        return _strip_required(value)


class PolicyDependencyConstraint(BaseModel):
    """Resolver policy constraint — deny/pin/python version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_kind: str = _NON_EMPTY
    constraint_value: str = _NON_EMPTY

    @field_validator("constraint_kind", "constraint_value")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class CandidateDependencySpecification(BaseModel):
    """L3 — deterministic merge input for resolver (§15.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CANDIDATE_DEPENDENCY_SPECIFICATION_V1
    application_release_id: str = _NON_EMPTY
    platform_version: str = _NON_EMPTY
    repository_declaration: RepositoryDependencyDeclaration
    agent_packages: tuple[InstalledAgentPackageRequirement, ...]
    platform_extras: tuple[str, ...] = ()
    policy_constraints: tuple[PolicyDependencyConstraint, ...] = ()
    repository_lock_hint_ref: str | None = None

    @field_validator(
        "application_release_id", "platform_version", "repository_lock_hint_ref"
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @model_validator(mode="after")
    def _validate_application_release_alignment(
        self,
    ) -> CandidateDependencySpecification:
        if (
            self.application_release_id
            != self.repository_declaration.application_release_id
        ):
            raise ValueError(
                "application_release_id must match repository_declaration.application_release_id"
            )
        return self


class DependencyResolverInput(BaseModel):
    """L4 — resolver boundary input (§15.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DEPENDENCY_RESOLVER_INPUT_V1
    specification: CandidateDependencySpecification
    resolver_algorithm_id: str = _NON_EMPTY
    resolver_algorithm_version: str = _NON_EMPTY
    lock_policy_ref: str | None = None

    @field_validator(
        "resolver_algorithm_id", "resolver_algorithm_version", "lock_policy_ref"
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    def inputs_digest(self) -> str:
        return content_digest_for_model(self)


class LockPackageRole(StrEnum):
    DIRECT = "direct"
    TRANSITIVE = "transitive"


class MaterializedLockPackage(BaseModel):
    """One resolved package in the immutable closure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_name: str = _NON_EMPTY
    version: str = _NON_EMPTY
    package_digest: str | None = None
    dependency_of: str | None = None

    @field_validator("distribution_name", "version", "dependency_of")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_optional_package_digest(cls, value: str | None) -> str | None:
        return normalize_optional_package_digest(value)


class MaterializedAgentClosureEntry(BaseModel):
    """Agent closure edge in the lock artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_package_id: str = _NON_EMPTY
    package_digest: str = _NON_EMPTY
    role: LockPackageRole

    @field_validator("distribution_package_id")
    @classmethod
    def _strip_distribution_package_id(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)


class MaterializedLockReproducibilityEvidence(BaseModel):
    """Audit pointers — no secrets or live catalog data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolver_log_ref: str | None = None
    input_snapshot_ref: str | None = None

    @field_validator("resolver_log_ref", "input_snapshot_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class MaterializedLockRollbackEvidence(BaseModel):
    """Rollback lineage metadata on lock artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supersedes_lock_id: str | None = None
    rollback_eligible: bool = False

    @field_validator("supersedes_lock_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class MaterializedRuntimeLock(BaseModel):
    """L5 — immutable dependency closure artifact (§16.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MATERIALIZED_RUNTIME_LOCK_V1
    lock_id: str | None = None
    lock_digest: str | None = None
    resolver_algorithm_id: str = _NON_EMPTY
    resolver_algorithm_version: str = _NON_EMPTY
    created_at: datetime | None = None
    inputs_digest: str = _NON_EMPTY
    intergrax_version: str = _NON_EMPTY
    python_version: str = _NON_EMPTY
    platform_extras: tuple[str, ...] = ()
    packages: tuple[MaterializedLockPackage, ...]
    agent_closure: tuple[MaterializedAgentClosureEntry, ...]
    repository_lock_hint_digest: str | None = None
    reproducibility_evidence: MaterializedLockReproducibilityEvidence | None = None
    rollback_evidence: MaterializedLockRollbackEvidence | None = None

    @field_validator(
        "lock_id",
        "lock_digest",
        "resolver_algorithm_id",
        "resolver_algorithm_version",
        "inputs_digest",
        "intergrax_version",
        "python_version",
        "repository_lock_hint_digest",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    def compute_lock_digest(self) -> str:
        payload = self.model_copy(
            update={"lock_id": None, "lock_digest": None, "created_at": None}
        )
        return content_digest_for_model(payload)

    def with_content_identity(self) -> MaterializedRuntimeLock:
        digest = self.compute_lock_digest()
        return self.model_copy(update={"lock_id": digest, "lock_digest": digest})

    @model_validator(mode="after")
    def _validate_content_identity(self) -> MaterializedRuntimeLock:
        computed = self.compute_lock_digest()
        has_lock_id = self.lock_id is not None
        has_lock_digest = self.lock_digest is not None
        if has_lock_id != has_lock_digest:
            raise ValueError("lock_id and lock_digest must both be set or both absent")
        if has_lock_id:
            if self.lock_id != self.lock_digest:
                raise ValueError("lock_id and lock_digest must match")
            if self.lock_id != computed:
                raise ValueError(
                    "claimed lock identity does not match semantic content digest"
                )
        return self
