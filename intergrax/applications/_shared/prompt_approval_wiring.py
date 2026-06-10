# © Artur Czarnecki. All rights reserved.

"""Prompt approval workflow wiring for product hosts (AUDIT-IDEAL-17.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.prompts.registry.prompt_approval import PromptApprovalQueue


@dataclass(frozen=True, slots=True)
class PromptApprovalWiring:
    enabled: bool
    queue: PromptApprovalQueue | None


def resolve_prompt_approval_wiring(env: ApplicationEnvironmentProfile) -> PromptApprovalWiring:
    """Product hosts require managed prompt approval before use."""
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    enabled = is_product and env.prompt_profile.approval_required
    return PromptApprovalWiring(
        enabled=enabled,
        queue=PromptApprovalQueue() if enabled else None,
    )
