# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register skill bundles from manifests or plugin classes."""

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.core.plugin import SkillPlugin, skill_bundle_manifest_for_plugin
from intergrax.skills.registry.catalog import SkillBundleEntry, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry


def register_from_skill_manifest(
    manifest: SkillBundleManifest,
    register_fn,
    *,
    override: bool = False,
) -> SkillBundleManifest:
    register_skill_bundle(
        SkillBundleEntry(
            bundle_id=manifest.bundle_id,
            skill_ids=manifest.skill_ids,
            register=register_fn,
            status=manifest.status,
            description=manifest.description,
        ),
        override=override,
    )
    return manifest


def register_skill_plugin(
    plugin: type[SkillPlugin],
    *,
    override: bool = False,
) -> SkillBundleManifest:
    manifest = skill_bundle_manifest_for_plugin(plugin)

    def _register(registry: SkillRegistry) -> None:
        plugin.register_skills(registry)

    return register_from_skill_manifest(manifest, _register, override=override)
