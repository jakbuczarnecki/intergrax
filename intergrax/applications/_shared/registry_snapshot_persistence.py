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
