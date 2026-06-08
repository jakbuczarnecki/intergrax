# © Artur Czarnecki. All rights reserved.
"""Regenerate plugin.py for bundles extended by SK-EXP4."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "intergrax" / "skills" / "providers"

EXTENDED = (
    "browser",
    "cache",
    "knowledge",
    "data",
    "eval",
    "harness",
    "health",
    "ops",
    "memory",
    "notify",
    "platform",
    "storage",
    "vector_store",
    "modality",
    "research",
)


def _const_names(manifests_path: Path) -> list[str]:
    text = manifests_path.read_text(encoding="utf-8")
    return re.findall(r"^([A-Z][A-Z0-9_]+) = SkillManifest\(", text, re.MULTILINE)


def _bundle_title(bundle_id: str) -> str:
    return "".join(p.capitalize() for p in bundle_id.split("_"))


def _regenerate_plugin(bundle_id: str) -> None:
    manifests_path = ROOT / bundle_id / "manifests.py"
    consts = _const_names(manifests_path)
    if not consts:
        raise ValueError(f"No manifests in {bundle_id}")
    class_name = _bundle_title(bundle_id) + "SkillPlugin"
    tuple_name = f"_{bundle_id.upper()}_MANIFESTS"
    imports = ",\n    ".join(consts)
    plugin_path = ROOT / bundle_id / "plugin.py"
    existing = plugin_path.read_text(encoding="utf-8")
    desc_match = re.search(r'description="([^"]+)"', existing)
    desc = desc_match.group(1) if desc_match else f"{bundle_id} skill packs"
    if "SK-EXP" not in desc:
        desc = f"{desc} (SK-EXP4)"

    content = f'''# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.{bundle_id}.manifests import (
    {imports},
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

{tuple_name} = (
    {",\n    ".join(consts)},
)


class {class_name}:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="{bundle_id}",
            skill_ids=tuple(m.skill_id for m in {tuple_name}),
            status=SkillBundleStatus.STABLE,
            description="{desc}",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return {tuple_name}

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in {tuple_name}:
            registry.register(manifest)
'''
    plugin_path.write_text(content, encoding="utf-8")
    print(f"Patched {bundle_id}/plugin.py ({len(consts)} manifests)")


def main() -> None:
    for bundle_id in EXTENDED:
        _regenerate_plugin(bundle_id)


if __name__ == "__main__":
    main()
