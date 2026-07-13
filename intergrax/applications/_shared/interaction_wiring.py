# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production Slack / Teams / lab interaction intake for Tier-3 hosts (§18, B.12)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.runtime.interactions.factory import (
    create_interaction_adapter,
    resolve_interaction_settings,
)
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.interactions.task_executor import TaskExecutor
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def wire_interaction_intake_service(
    nexus_loop: NexusLoop | None = None,
    *,
    interaction_surface: str = "auto",
    task_executor: TaskExecutor | None = None,
    task_enricher: Callable[..., object] | None = None,
) -> InteractionIntakeService:
    """
    Build :class:`InteractionIntakeService` for ``POST /v1/interactions/intake``.

    ``interaction_surface`` follows ``LAB_INTERACTION_SURFACE`` semantics:
    ``auto`` | ``slack`` | ``teams`` | ``lab`` | ``lab_json``.
    """
    settings = resolve_interaction_settings(surface=interaction_surface)
    adapter = create_interaction_adapter(settings)
    verifier = create_inbound_verifier()
    return InteractionIntakeService(
        nexus_loop=nexus_loop,
        task_executor=task_executor,
        adapter=adapter,
        verifier=verifier,
        task_enricher=task_enricher,
    )
