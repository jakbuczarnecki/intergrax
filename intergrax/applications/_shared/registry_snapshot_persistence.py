# © Artur Czarnecki. All rights reserved.

"""Typed durable envelope for registry snapshot persistence (COMPAT-X-R2-R1)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_REGISTRY_SNAPSHOT_V1: Final = "registry_snapshot.v1"


class RegistrySnapshotPersistedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["registry_snapshot.v1"] = SCHEMA_REGISTRY_SNAPSHOT_V1
    tool_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    prompt_ids: tuple[str, ...] = ()
    agent_contract_ids: tuple[str, ...] = ()
    evaluation_registry_ids: tuple[str, ...] = ()

    def to_legacy_projection(self) -> dict[str, object]:
        return {
            "tool_ids": list(self.tool_ids),
            "skill_ids": list(self.skill_ids),
            "prompt_ids": list(self.prompt_ids),
            "agent_contract_ids": list(self.agent_contract_ids),
            "evaluation_registry_ids": list(self.evaluation_registry_ids),
        }


class LegacyRegistrySnapshotWire(BaseModel):
    """Historical writer shape before ``schema_version`` (evidence @ e5ae6a3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    prompt_ids: tuple[str, ...] = ()
    agent_contract_ids: tuple[str, ...] = ()
    evaluation_registry_ids: tuple[str, ...] = ()


LEGACY_UNVERSIONED_PRE_V1: Final = "LEGACY_UNVERSIONED_PRE_V1"


def parse_registry_snapshot_persisted(raw: dict[str, object]) -> RegistrySnapshotPersistedPayload:
    schema_version = raw.get("schema_version")
    if schema_version == SCHEMA_REGISTRY_SNAPSHOT_V1:
        return RegistrySnapshotPersistedPayload.model_validate(raw)
    if "schema_version" not in raw:
        legacy = LegacyRegistrySnapshotWire.model_validate(raw)
        return RegistrySnapshotPersistedPayload(
            schema_version=SCHEMA_REGISTRY_SNAPSHOT_V1,
            tool_ids=legacy.tool_ids,
            skill_ids=legacy.skill_ids,
            prompt_ids=legacy.prompt_ids,
            agent_contract_ids=legacy.agent_contract_ids,
            evaluation_registry_ids=legacy.evaluation_registry_ids,
        )
    raise ValueError(
        f"unsupported registry snapshot schema version {schema_version!r}; "
        f"expected {SCHEMA_REGISTRY_SNAPSHOT_V1!r} or {LEGACY_UNVERSIONED_PRE_V1}"
    )
