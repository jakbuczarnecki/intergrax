#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Cross-check role_registry vs skill_routing_registry."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_ACP = _REPO / "governance" / "agent-control-plane"


def main() -> int:
    roles_doc = yaml.safe_load((_ACP / "role_registry.yaml").read_text(encoding="utf-8"))
    skills_doc = yaml.safe_load((_ACP / "skill_routing_registry.yaml").read_text(encoding="utf-8"))
    roles = roles_doc.get("roles", {})
    skills = skills_doc.get("skills", {})
    errors: list[str] = []
    for skill_id, meta in skills.items():
        if not isinstance(meta, dict):
            errors.append(f"{skill_id}: invalid entry")
            continue
        role = meta.get("role")
        if role not in roles:
            errors.append(f"{skill_id}: role {role!r} not in role_registry")
        rel = meta.get("path")
        if not rel or not (_REPO / str(rel)).is_file():
            errors.append(f"{skill_id}: missing file {rel}")
    for role_name, role_meta in roles.items():
        allowed = role_meta.get("allowed_skills", [])
        if isinstance(allowed, list):
            for skill_id in allowed:
                if skill_id not in skills:
                    errors.append(f"role {role_name}: unknown skill {skill_id}")
    if errors:
        print("[ERROR] control plane consistency failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("[INFO] verify_control_plane_consistency: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
