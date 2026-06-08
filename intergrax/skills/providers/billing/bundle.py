# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.billing.plugin import BillingSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_billing_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(BillingSkillPlugin, override=override)
