# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for inbound interaction adapters (§18, Phase H.2)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.adapters.chained_adapter import ChainedInteractionAdapter
from intergrax.runtime.interactions.adapters.lab_json_adapter import LabJsonInteractionAdapter
from intergrax.runtime.interactions.adapters.slash_command_adapter import SlashCommandInteractionAdapter
from intergrax.runtime.interactions.adapters.teams_activity_adapter import TeamsActivityInteractionAdapter
from intergrax.runtime.task.task import Task

ENV_INTERACTION_SURFACE = "INTERGRAX_INTERACTION_SURFACE"

InteractionAdapterFactory = Callable[[], InteractionAdapter]

_DEFAULT_CHAIN = (
    SlashCommandInteractionAdapter(),
    TeamsActivityInteractionAdapter(),
    LabJsonInteractionAdapter(),
)


class InteractionSurface(str, Enum):
    AUTO = "auto"
    LAB = "lab"
    SLASH_COMMAND = "slash_command"
    TEAMS = "teams"


@dataclass(frozen=True)
class InteractionSettings:
    surface: InteractionSurface = InteractionSurface.AUTO


def resolve_interaction_settings(*, surface: Optional[str] = None) -> InteractionSettings:
    raw = (surface or os.environ.get(ENV_INTERACTION_SURFACE, InteractionSurface.AUTO.value)).strip().lower()
    try:
        resolved = InteractionSurface(raw)
    except ValueError:
        resolved = InteractionSurface.AUTO
    return InteractionSettings(surface=resolved)


def create_interaction_adapter(
    settings: Optional[InteractionSettings] = None,
    *,
    implementation: Optional[InteractionAdapter] = None,
    factory: Optional[InteractionAdapterFactory] = None,
) -> InteractionAdapter:
    """
    Build an inbound interaction adapter.

    Priority: explicit ``implementation`` > ``factory`` > ``settings``/env defaults.
    """
    if implementation is not None:
        return implementation
    if factory is not None:
        return factory()

    resolved = settings or resolve_interaction_settings()
    if resolved.surface == InteractionSurface.LAB:
        return LabJsonInteractionAdapter()
    if resolved.surface == InteractionSurface.SLASH_COMMAND:
        return SlashCommandInteractionAdapter()
    if resolved.surface == InteractionSurface.TEAMS:
        return TeamsActivityInteractionAdapter()
    return ChainedInteractionAdapter(_DEFAULT_CHAIN)


def intake_payload_to_task(
    payload: Dict[str, object],
    *,
    tenant_id: str,
    user_id: Optional[str] = None,
    adapter: Optional[InteractionAdapter] = None,
) -> Task:
    """Convenience entrypoint for HTTP handlers, workers, and lab scripts."""
    resolved = adapter or create_interaction_adapter()
    if not isinstance(payload, dict):
        raise TypeError("interaction payload must be a dict")
    return resolved.to_task(payload, tenant_id=tenant_id, user_id=user_id)
