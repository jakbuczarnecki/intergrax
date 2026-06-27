#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""SK-MAINT-02 — promoted skill bundles must match catalog STABLE status."""

from __future__ import annotations

import sys


def main() -> int:
    from intergrax.skills.providers.knowledge.plugin import KnowledgeSkillPlugin
    from intergrax.skills.registry.catalog import SkillBundleStatus

    manifest = KnowledgeSkillPlugin.skill_bundle_manifest()
    if manifest.status is not SkillBundleStatus.STABLE:
        print(f"skill bundle maturity audit failed: knowledge is {manifest.status.value}, expected stable")
        return 1
    if manifest.bundle_id != "knowledge":
        print("skill bundle maturity audit failed: unexpected knowledge bundle_id")
        return 1
    print("skill bundle maturity audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
