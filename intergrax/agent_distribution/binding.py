# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Application agent binding contracts (AGENT_DISTRIBUTION §12)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from intergrax.agent_distribution._config_validation import validate_non_secret_distribution_config
from intergrax.agent_distribution._immutable_json import (
    DistributionJsonValue,
    distribution_json_to_plain,
    freeze_distribution_json_object,
)

_NON_EMPTY = Field(min_length=1)

SCHEMA_APPLICATION_AGENT_BINDING_V1: Final = "application_agent_binding.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentBindingPolicyOverrides(BaseModel):
    """Tool/budget policy overrides — durable binding authority only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_allowlist: tuple[str, ...] = ()
    tool_denylist: tuple[str, ...] = ()
    budget_override_ref: str | None = None

    @field_validator("budget_override_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentBindingFactoryReference(BaseModel):
    """Typed factory/builder reference representation — strings only, no imports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    builder_key: str | None = None
    factory_path: str | None = None

    @field_validator("builder_key", "factory_path")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @model_validator(mode="after")
    def _require_one_reference(self) -> AgentBindingFactoryReference:
        if self.builder_key is None and self.factory_path is None:
            raise ValueError("factory reference requires builder_key or factory_path")
        return self


class ApplicationAgentBinding(BaseModel):
    """Durable application binding anchored on installation_slot_id (§12.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_APPLICATION_AGENT_BINDING_V1
    application_binding_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    logical_agent_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    active_installation_id: str | None = None
    builtin_package_ref: str | None = None
    enablement: bool = False
    default_agent: bool | None = None
    config: Mapping[str, DistributionJsonValue] = Field(default_factory=dict)
    secret_refs: tuple[str, ...] = ()
    policy_overrides: AgentBindingPolicyOverrides | None = None
    factory_reference: AgentBindingFactoryReference | None = None
    manifest_origin_ref: str | None = None
    tombstone: bool = False
    binding_revision: int = Field(ge=0)

    @field_validator(
        "application_binding_id",
        "application_id",
        "application_environment_id",
        "logical_agent_id",
        "installation_slot_id",
        "active_installation_id",
        "builtin_package_ref",
        "manifest_origin_ref",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("secret_refs")
    @classmethod
    def _validate_secret_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)

    @field_validator("config", mode="before")
    @classmethod
    def _validate_config_raw(cls, value: object) -> dict[str, DistributionJsonValue]:
        if not isinstance(value, Mapping):
            raise ValueError("config must be a mapping")
        return validate_non_secret_distribution_config(
            value,
            context_label="binding config",
        )

    @field_validator("config", mode="after")
    @classmethod
    def _freeze_config(
        cls,
        value: dict[str, DistributionJsonValue],
    ) -> Mapping[str, DistributionJsonValue]:
        return freeze_distribution_json_object(value)

    @field_serializer("config")
    def _serialize_config(self, value: Mapping[str, DistributionJsonValue]) -> dict[str, DistributionJsonValue]:
        return distribution_json_to_plain(value)

    @model_validator(mode="after")
    def _validate_binding_target(self) -> ApplicationAgentBinding:
        if self.tombstone and self.enablement:
            raise ValueError("tombstoned bindings cannot be enabled")
        return self

    def survives_installation_upgrade(
        self,
        *,
        prior_active_installation_id: str,
        next_active_installation_id: str,
    ) -> bool:
        """Binding identity is slot-anchored; only active_installation_id may change."""
        if self.installation_slot_id == "":
            return False
        if self.active_installation_id != prior_active_installation_id:
            return False
        return next_active_installation_id != prior_active_installation_id
