#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ensure every routed skill exists on disk."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "governance" / "agent-control-plane" / "skill_routing_registry.yaml"


def main() -> int:
    doc = yaml.safe_load(_REGISTRY.read_text(encoding="utf-8"))
    skills = doc.get("skills", {})
    missing = []
    for skill_id, meta in skills.items():
        path = _REPO / str(meta.get("path", ""))
        if not path.is_file():
            missing.append(f"{skill_id} -> {path.relative_to(_REPO)}")
    if missing:
        print("[ERROR] verify_agent_skills: missing SKILL.md files:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"[INFO] verify_agent_skills: OK ({len(skills)} skills)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
