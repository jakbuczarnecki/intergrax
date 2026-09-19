# © Artur Czarnecki. All rights reserved.

"""Collaborative Activity ingestion policy contracts (Multiplayer MP-6C).

Trusted publisher context, replaceable ingestion policy, and typed admission
decisions — deterministic only; no persistence or source-domain semantics.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityPublication,
)

SCHEMA_COLLABORATIVE_ACTIVITY_PUBLISHER_CONTEXT_V1: Final = (
    "collaborative_activity_publisher_context.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_INGESTION_REQUEST_V1: Final = (
    "collaborative_activity_ingestion_request.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_INGESTION_DECISION_V1: Final = (
    "collaborative_activity_ingestion_decision.v1"
)
SCHEMA_DEFAULT_COLLABORATIVE_ACTIVITY_INGESTION_POLICY_CONFIG_V1: Final = (
    "default_collaborative_activity_ingestion_policy_config.v1"
)

DEFAULT_COLLABORATIVE_ACTIVITY_INGESTION_POLICY_ID: Final = (
    "collaborative_work.collaborative_activity.ingestion.default"
)

_RESERVED_PUBLISHER_NAMESPACES: Final = frozenset({"intergrax", "platform"})

_NON_EMPTY = Field(min_length=1)


class CollaborativeActivityPublisherKind(StrEnum):
    """Trusted ingress classification — not the semantic activity actor."""

    PLATFORM = "platform"
    PLUGIN = "plugin"


class CollaborativeActivityPublisherContext(BaseModel):
    """Authenticated producer identity supplied by the trusted composition boundary.

    Must not be derived from ``CollaborativeActivityPublication`` fields.
    ``producer_principal_id`` is the authenticated caller; ``publication.actor`` is
    the semantic subject of the activity only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_publisher_context.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PUBLISHER_CONTEXT_V1
    )
    tenant_id: str = _NON_EMPTY
    producer_principal_id: str = _NON_EMPTY
    kind: CollaborativeActivityPublisherKind
    owned_namespace: str | None = Field(
        default=None,
        description="Required for PLUGIN — sole source/type namespace this producer may claim",
    )
    allowed_workspace_ids: tuple[str, ...] = Field(
        default=(),
        description="When non-empty, publication workspace_id must be listed",
    )

    @field_validator("tenant_id", "producer_principal_id", "owned_namespace")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _plugin_namespace_required(self) -> CollaborativeActivityPublisherContext:
        if self.kind is CollaborativeActivityPublisherKind.PLUGIN:
            if self.owned_namespace is None:
                raise ValueError("owned_namespace is required for PLUGIN publishers")
            owned = self.owned_namespace.strip().lower()
            if owned in _RESERVED_PUBLISHER_NAMESPACES:
                raise ValueError("PLUGIN owned_namespace cannot be a reserved namespace")
        elif self.owned_namespace is not None:
            raise ValueError("owned_namespace must be omitted for PLATFORM publishers")
        return self


class CollaborativeActivityIngestionOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class CollaborativeActivityIngestionDenialReason(StrEnum):
    """Stable machine-readable admission denial codes."""

    TENANT_MISMATCH = "tenant_mismatch"
    WORKSPACE_RESTRICTED = "workspace_restricted"
    RESERVED_NAMESPACE_SPOOF = "reserved_namespace_spoof"
    SOURCE_NAMESPACE_UNAUTHORIZED = "source_namespace_unauthorized"
    TYPE_NAMESPACE_UNAUTHORIZED = "type_namespace_unauthorized"
    SOURCE_TYPE_NAMESPACE_MISMATCH = "source_type_namespace_mismatch"
    PLATFORM_TYPE_REQUIRED = "platform_type_required"
    CORRECTION_PUBLISHER_UNAUTHORIZED = "correction_publisher_unauthorized"
    POLICY_AMBIGUITY = "policy_ambiguity"


class CollaborativeActivityIngestionRequest(BaseModel):
    """Policy input — publication plus trusted publisher context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_ingestion_request.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_INGESTION_REQUEST_V1
    )
    publication: CollaborativeActivityPublication
    publisher_context: CollaborativeActivityPublisherContext


class CollaborativeActivityIngestionDecision(BaseModel):
    """Typed ingestion policy outcome — authoritative for append intent materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_ingestion_decision.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_INGESTION_DECISION_V1
    )
    outcome: CollaborativeActivityIngestionOutcome
    policy_id: str = _NON_EMPTY
    effective_durability_class: CollaborativeActivityDurabilityClass | None = None
    denial_reason: CollaborativeActivityIngestionDenialReason | None = None

    @model_validator(mode="after")
    def _align_outcome_fields(self) -> CollaborativeActivityIngestionDecision:
        if self.outcome is CollaborativeActivityIngestionOutcome.ALLOW:
            if self.denial_reason is not None:
                raise ValueError("denial_reason must be omitted when outcome is allow")
            if self.effective_durability_class is None:
                raise ValueError("effective_durability_class required when outcome is allow")
        else:
            if self.denial_reason is None:
                raise ValueError("denial_reason required when outcome is deny")
            if self.effective_durability_class is not None:
                raise ValueError("effective_durability_class must be omitted when outcome is deny")
        return self


class DefaultCollaborativeActivityIngestionPolicyConfig(BaseModel):
    """Immutable configuration for the platform default ingestion policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["default_collaborative_activity_ingestion_policy_config.v1"] = (
        SCHEMA_DEFAULT_COLLABORATIVE_ACTIVITY_INGESTION_POLICY_CONFIG_V1
    )
    policy_id: str = DEFAULT_COLLABORATIVE_ACTIVITY_INGESTION_POLICY_ID


def fail_closed_collaborative_activity_ingestion_decision(
    *,
    policy_id: str,
    denial_reason: CollaborativeActivityIngestionDenialReason,
) -> CollaborativeActivityIngestionDecision:
    return CollaborativeActivityIngestionDecision(
        outcome=CollaborativeActivityIngestionOutcome.DENY,
        policy_id=policy_id,
        denial_reason=denial_reason,
    )


@runtime_checkable
class CollaborativeActivityIngestionPolicy(Protocol):
    """Replaceable deterministic ingestion admission and durability resolution."""

    @property
    def policy_id(self) -> str:
        """Stable policy identity for audit."""
        ...

    def evaluate(
        self,
        request: CollaborativeActivityIngestionRequest,
    ) -> CollaborativeActivityIngestionDecision:
        """Admission, namespace authorization, and effective durability resolution."""
        ...


class CollaborativeActivityAdmissionRejected(RuntimeError):
    """Expected policy denial at the publication ingress boundary."""

    def __init__(
        self,
        *,
        denial_reason: CollaborativeActivityIngestionDenialReason,
        policy_id: str,
    ) -> None:
        self.denial_reason = denial_reason
        self.policy_id = policy_id
        super().__init__(f"{policy_id}: {denial_reason.value}")


class CollaborativeActivityIngestionPolicyError(RuntimeError):
    """Unexpected policy evaluation failure — ingestion fails closed."""


class CollaborativeActivityIngestionAppendError(RuntimeError):
    """Append store failure after successful admission."""
