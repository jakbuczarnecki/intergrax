# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.cloud_platform.plugin import CloudPlatformSkillPlugin
from intergrax.skills.registry.plugin_register import register_skill_plugin


def register_cloud_platform_skill_bundle(*, override: bool = False) -> None:
    register_skill_plugin(CloudPlatformSkillPlugin, override=override)
