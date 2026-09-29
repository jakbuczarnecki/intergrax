# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed scoped integration adaptation contracts (AW-7C-P2/P3)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.base import IntegrationCategory


class ScopedIntegrationAdaptationFailureReason(StrEnum):
    STRATEGY_UNAVAILABLE = "STRATEGY_UNAVAILABLE"
    STRATEGY_AMBIGUOUS = "STRATEGY_AMBIGUOUS"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    SCOPE_WIDENING = "SCOPE_WIDENING"
    OPERATION_WIDENING = "OPERATION_WIDENING"
    NETWORK_WIDENING = "NETWORK_WIDENING"
    EXPIRY_WIDENING = "EXPIRY_WIDENING"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    FINGERPRINT_MISMATCH = "FINGERPRINT_MISMATCH"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    ADAPTATION_FAILED = "ADAPTATION_FAILED"


class ScopedIntegrationAdaptationError(Exception):
    def __init__(
        self,
        reason: ScopedIntegrationAdaptationFailureReason,
        *,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.detail = detail
        message = reason.value if not detail else f"{reason.value}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ScopedIntegrationAdaptationOperationId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            _require_non_empty(self.value, "operation_id"),
        )


def scoped_integration_adaptation_operation_id(
    value: str,
) -> ScopedIntegrationAdaptationOperationId:
    return ScopedIntegrationAdaptationOperationId(value=value)


def parse_scoped_integration_adaptation_operation_ids(
    values: tuple[str, ...],
) -> tuple[ScopedIntegrationAdaptationOperationId, ...]:
    if not values:
        raise ValueError("permitted operations must not be empty")
    return tuple(scoped_integration_adaptation_operation_id(value) for value in values)


@runtime_checkable
class ScopedIntegrationAdaptationSpecification(Protocol):
    @property
    def specification_type(self) -> str: ...

    @property
    def specification_version(self) -> str: ...

    @property
    def specification_fingerprint(self) -> str: ...


def _require_non_empty(value: str, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _sorted_operations(
    operations: tuple[ScopedIntegrationAdaptationOperationId, ...],
) -> tuple[ScopedIntegrationAdaptationOperationId, ...]:
    if not operations:
        raise ValueError("permitted operations must not be empty")
    return tuple(sorted(operations, key=lambda op: op.value))


@dataclass(frozen=True, slots=True)
class ScopedIntegrationAdaptationScope:
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    permitted_operations: tuple[ScopedIntegrationAdaptationOperationId, ...]
    network_allowlist: NetworkEgressAllowlist
    credential_grant_ref: str
    expires_at: datetime
    candidate_id: str
    candidate_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _require_non_empty(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "resource_scope",
            _require_non_empty(self.resource_scope, "resource_scope"),
        )
        object.__setattr__(
            self,
            "credential_grant_ref",
            _require_non_empty(self.credential_grant_ref, "credential_grant_ref"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            _require_non_empty(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "candidate_revision",
            _require_non_empty(self.candidate_revision, "candidate_revision"),
        )
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        object.__setattr__(
            self,
            "permitted_operations",
            _sorted_operations(self.permitted_operations),
        )


def derive_scoped_integration_adaptation_scope_fingerprint(
    scope: ScopedIntegrationAdaptationScope,
) -> str:
    op_part = ",".join(op.value for op in scope.permitted_operations)
    parts = [
        f"tenant_id={scope.tenant_id}",
        f"integration_category={scope.integration_category.value}",
        f"provider_id={scope.provider_id}",
        f"resource_scope={scope.resource_scope}",
        f"operations={op_part}",
        f"network={scope.network_allowlist.fingerprint()}",
        f"credential_grant_ref={scope.credential_grant_ref}",
        f"expires_at={scope.expires_at.isoformat()}",
        f"candidate_id={scope.candidate_id}",
        f"candidate_revision={scope.candidate_revision}",
    ]
    canonical = "\n".join(parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class ScopedIntegrationAdaptationTarget:
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    current_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _require_non_empty(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "resource_scope",
            _require_non_empty(self.resource_scope, "resource_scope"),
        )
        object.__setattr__(
            self,
            "current_revision",
            _require_non_empty(self.current_revision, "current_revision"),
        )


@dataclass(frozen=True, slots=True)
class ScopedIntegrationAdaptationRequest:
    request_id: str
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    scope: ScopedIntegrationAdaptationScope
    correlation_id: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _require_non_empty(self.request_id, "request_id"),
        )
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, "tenant_id"),
        )
        scope = self.scope
        if scope.tenant_id != self.tenant_id:
            raise ValueError("request tenant must match scope tenant")
        if scope.integration_category != self.integration_category:
            raise ValueError("request integration_category must match scope")
        if scope.provider_id != self.provider_id:
            raise ValueError("request provider_id must match scope")
        if scope.resource_scope != self.resource_scope:
            raise ValueError("request resource_scope must match scope")


@dataclass(frozen=True, slots=True)
class ScopedIntegrationAdaptationArtifact:
    artifact_id: str
    artifact_fingerprint: str
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    strategy_id: str
    candidate_id: str
    candidate_revision: str
    scope: ScopedIntegrationAdaptationScope
    scope_fingerprint: str
    specification: ScopedIntegrationAdaptationSpecification
    evidence_refs: tuple[str, ...] = ()
    correlation_id: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _require_non_empty(self.artifact_id, "artifact_id"),
        )
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _require_non_empty(self.artifact_fingerprint, "artifact_fingerprint"),
        )
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _require_non_empty(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "resource_scope",
            _require_non_empty(self.resource_scope, "resource_scope"),
        )
        object.__setattr__(
            self,
            "strategy_id",
            _require_non_empty(self.strategy_id, "strategy_id"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            _require_non_empty(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "candidate_revision",
            _require_non_empty(self.candidate_revision, "candidate_revision"),
        )
        art_scope = self.scope
        if art_scope.tenant_id != self.tenant_id:
            raise ValueError("artifact tenant must match scope tenant")
        if art_scope.integration_category != self.integration_category:
            raise ValueError("artifact integration_category must match scope")
        if art_scope.provider_id != self.provider_id:
            raise ValueError("artifact provider_id must match scope")
        if art_scope.resource_scope != self.resource_scope:
            raise ValueError("artifact resource_scope must match scope")
        if art_scope.candidate_id != self.candidate_id:
            raise ValueError("artifact candidate_id must match scope")
        if art_scope.candidate_revision != self.candidate_revision:
            raise ValueError("artifact candidate_revision must match scope")
        expected_scope_fp = derive_scoped_integration_adaptation_scope_fingerprint(
            art_scope,
        )
        if self.scope_fingerprint != expected_scope_fp:
            raise ValueError("scope_fingerprint must match derived scope fingerprint")
        expected_artifact_fp = self.derived_artifact_fingerprint()
        if self.artifact_fingerprint != expected_artifact_fp:
            raise ValueError("artifact_fingerprint must match derived fingerprint")

    def derived_artifact_fingerprint(self) -> str:
        return derive_scoped_integration_adaptation_artifact_fingerprint(
            artifact_id=self.artifact_id,
            strategy_id=self.strategy_id,
            integration_category=self.integration_category,
            provider_id=self.provider_id,
            resource_scope=self.resource_scope,
            candidate_id=self.candidate_id,
            candidate_revision=self.candidate_revision,
            scope_fingerprint=self.scope_fingerprint,
            specification=self.specification,
        )


def derive_scoped_integration_adaptation_artifact_fingerprint(
    *,
    artifact_id: str,
    strategy_id: str,
    integration_category: IntegrationCategory,
    provider_id: str,
    resource_scope: str,
    candidate_id: str,
    candidate_revision: str,
    scope_fingerprint: str,
    specification: ScopedIntegrationAdaptationSpecification,
) -> str:
    parts = [
        f"artifact_id={artifact_id}",
        f"strategy_id={strategy_id}",
        f"integration_category={integration_category.value}",
        f"provider_id={provider_id}",
        f"resource_scope={resource_scope}",
        f"candidate_id={candidate_id}",
        f"candidate_revision={candidate_revision}",
        f"scope_fingerprint={scope_fingerprint}",
        f"spec_type={specification.specification_type}",
        f"spec_version={specification.specification_version}",
        f"spec_fp={specification.specification_fingerprint}",
    ]
    canonical = "\n".join(parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@runtime_checkable
class ScopedIntegrationAdaptationStrategy(Protocol):
    @property
    def strategy_id(self) -> str: ...

    def supports(
        self,
        request: ScopedIntegrationAdaptationRequest,
        target: ScopedIntegrationAdaptationTarget,
    ) -> bool: ...

    def adapt(
        self,
        request: ScopedIntegrationAdaptationRequest,
        target: ScopedIntegrationAdaptationTarget,
    ) -> ScopedIntegrationAdaptationArtifact: ...


@dataclass(frozen=True, slots=True)
class ScopedIntegrationAdaptationTargetLookupKey:
    """Integrations-owned lookup identity — not request-echo target truth."""

    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _require_non_empty(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "resource_scope",
            _require_non_empty(self.resource_scope, "resource_scope"),
        )


@runtime_checkable
class ScopedIntegrationAdaptationTargetSource(Protocol):
    """Authoritative target truth — resolved independently of request claims."""

    def resolve(
        self,
        key: ScopedIntegrationAdaptationTargetLookupKey,
    ) -> ScopedIntegrationAdaptationTarget: ...


@runtime_checkable
class ScopedIntegrationAdaptationTargetResolver(Protocol):
    def resolve_target(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationTarget: ...


@runtime_checkable
class ScopedIntegrationAdaptationPort(Protocol):
    def adapt(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationArtifact: ...


@dataclass(frozen=True, slots=True)
class ScopedAdaptedIntegrationOperationEvidence:
    """Typed adapted-operation facts — no secret material."""

    evidence_ref: str
    tenant_id: str
    execution_id: str
    artifact_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_ref",
            _require_non_empty(self.evidence_ref, "evidence_ref"),
        )
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "execution_id",
            _require_non_empty(self.execution_id, "execution_id"),
        )
        object.__setattr__(
            self,
            "artifact_id",
            _require_non_empty(self.artifact_id, "artifact_id"),
        )


@runtime_checkable
class ScopedAdaptedIntegrationOperationPort(Protocol):
    """Execution-bound adapted integration operation — Integrations-owned seam."""

    def execute(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence: ...


SCOPE_FINGERPRINT_PREFIX: Final = "sha256:"


def build_scoped_integration_adaptation_artifact(
    *,
    artifact_id: str,
    tenant_id: str,
    integration_category: IntegrationCategory,
    provider_id: str,
    resource_scope: str,
    strategy_id: str,
    candidate_id: str,
    candidate_revision: str,
    scope: ScopedIntegrationAdaptationScope,
    specification: ScopedIntegrationAdaptationSpecification,
    evidence_refs: tuple[str, ...] = (),
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> ScopedIntegrationAdaptationArtifact:
    scope_fingerprint = derive_scoped_integration_adaptation_scope_fingerprint(scope)
    artifact_fingerprint = derive_scoped_integration_adaptation_artifact_fingerprint(
        artifact_id=artifact_id,
        strategy_id=strategy_id,
        integration_category=integration_category,
        provider_id=provider_id,
        resource_scope=resource_scope,
        candidate_id=candidate_id,
        candidate_revision=candidate_revision,
        scope_fingerprint=scope_fingerprint,
        specification=specification,
    )
    return ScopedIntegrationAdaptationArtifact(
        artifact_id=artifact_id,
        artifact_fingerprint=artifact_fingerprint,
        tenant_id=tenant_id,
        integration_category=integration_category,
        provider_id=provider_id,
        resource_scope=resource_scope,
        strategy_id=strategy_id,
        candidate_id=candidate_id,
        candidate_revision=candidate_revision,
        scope=scope,
        scope_fingerprint=scope_fingerprint,
        specification=specification,
        evidence_refs=evidence_refs,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


__all__ = [
    "build_scoped_integration_adaptation_artifact",
    "SCOPE_FINGERPRINT_PREFIX",
    "ScopedIntegrationAdaptationArtifact",
    "ScopedIntegrationAdaptationError",
    "ScopedIntegrationAdaptationFailureReason",
    "ScopedIntegrationAdaptationOperationId",
    "ScopedIntegrationAdaptationPort",
    "ScopedIntegrationAdaptationRequest",
    "ScopedIntegrationAdaptationScope",
    "ScopedIntegrationAdaptationSpecification",
    "ScopedIntegrationAdaptationStrategy",
    "ScopedAdaptedIntegrationOperationEvidence",
    "ScopedAdaptedIntegrationOperationPort",
    "ScopedIntegrationAdaptationTarget",
    "ScopedIntegrationAdaptationTargetLookupKey",
    "ScopedIntegrationAdaptationTargetResolver",
    "ScopedIntegrationAdaptationTargetSource",
    "derive_scoped_integration_adaptation_artifact_fingerprint",
    "derive_scoped_integration_adaptation_scope_fingerprint",
    "parse_scoped_integration_adaptation_operation_ids",
    "scoped_integration_adaptation_operation_id",
]
