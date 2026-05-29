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
from intergrax.runtime.task.task import Task

ENV_INTERACTION_SURFACE = "INTERGRAX_INTERACTION_SURFACE"

InteractionAdapterFactory = Callable[[], InteractionAdapter]


def _slack_interaction_adapter() -> InteractionAdapter:
    from intergrax.integrations.providers.slack.adapter import SlackInteractionAdapter

    return SlackInteractionAdapter()


def _teams_interaction_adapter() -> InteractionAdapter:
    from intergrax.integrations.providers.teams.bundle import create_teams_interaction_surface

    return create_teams_interaction_surface()


def _lab_json_interaction_adapter() -> InteractionAdapter:
    from intergrax.integrations.providers.lab_json.bundle import create_lab_json_interaction_surface

    return create_lab_json_interaction_surface()


def _default_interaction_chain() -> tuple[InteractionAdapter, ...]:
    return (
        _slack_interaction_adapter(),
        _teams_interaction_adapter(),
        _lab_json_interaction_adapter(),
    )


class InteractionSurface(str, Enum):
    AUTO = "auto"
    LAB = "lab"
    LAB_JSON = "lab_json"
    SLACK = "slack"
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
    if resolved.surface in (InteractionSurface.LAB, InteractionSurface.LAB_JSON):
        from intergrax.integrations.providers.lab_json.bundle import create_lab_json_interaction_surface

        return create_lab_json_interaction_surface()
    if resolved.surface in (InteractionSurface.SLACK, InteractionSurface.SLASH_COMMAND):
        from intergrax.integrations.providers.slack.bundle import create_slack_interaction_surface

        return create_slack_interaction_surface()
    if resolved.surface == InteractionSurface.TEAMS:
        from intergrax.integrations.providers.teams.bundle import create_teams_interaction_surface

        return create_teams_interaction_surface()
    return ChainedInteractionAdapter(_default_interaction_chain())


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
