#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-12.1 — LangGraph-compatible skill pack import path."""

from __future__ import annotations

import sys

from intergrax.applications._shared.skill_import_wiring import (
    import_langgraph_skill_pack,
    resolve_langgraph_skill_import_enabled,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.skills.importers.langgraph_skill_pack import LangGraphSkillPackImporter


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    if not resolve_langgraph_skill_import_enabled(env):
        print("product host must allow LangGraph skill import", file=sys.stderr)
        return 1

    manifest = LangGraphSkillPackImporter().import_payload(
        {
            "skill_id": "demo.langgraph",
            "description": "LangGraph-compatible demo pack",
            "version": "1.0.0",
            "tools": ["rag.retrieve"],
            "graph": {
                "nodes": [{"id": "retrieve"}, {"id": "summarize"}],
                "edges": [{"source": "retrieve", "target": "summarize"}],
            },
        }
    )
    if "langgraph_pack" not in manifest.tags:
        print("imported manifest must be tagged langgraph_pack", file=sys.stderr)
        return 1
    _ = import_langgraph_skill_pack  # wiring export present

    print("OK: LangGraph skill pack import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
