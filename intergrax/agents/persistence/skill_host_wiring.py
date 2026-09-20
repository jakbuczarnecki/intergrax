# © Artur Czarnecki. All rights reserved.

"""Host wiring for application-owned skill registries (ME-16 / SK-BRIDGE parity)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.skills.execution_binding import SkillExecutionPinningStore
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry


@dataclass(frozen=True, slots=True)
class HostSkillCatalogWiring:
    skill_profile: SkillProfile
    skill_registry: SkillRegistry
    skill_pinning_store: SkillExecutionPinningStore | None = None


def attach_skill_host_wiring_metadata(
    metadata: dict[str, Any],
    wiring: HostSkillCatalogWiring | None,
) -> dict[str, Any]:
    wired = dict(metadata)
    if wiring is not None:
        wired[AcpMetadataKey.SKILL_HOST_WIRING] = wiring
    return wired


def resolve_skill_host_wiring_from_metadata(
    metadata: dict[str, Any],
) -> HostSkillCatalogWiring | None:
    raw = metadata.get(AcpMetadataKey.SKILL_HOST_WIRING)
    if raw is None:
        return None
    if not isinstance(raw, HostSkillCatalogWiring):
        raise TypeError("skill host wiring metadata has unexpected type")
    return raw


def inject_acp_skill_host_wiring_metadata(
    metadata: dict[str, Any],
    wiring: HostSkillCatalogWiring | None,
) -> None:
    if wiring is None:
        return
    metadata[AcpMetadataKey.SKILL_HOST_WIRING] = wiring


__all__ = [
    "HostSkillCatalogWiring",
    "attach_skill_host_wiring_metadata",
    "inject_acp_skill_host_wiring_metadata",
    "resolve_skill_host_wiring_from_metadata",
]
