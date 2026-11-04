#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Validate agent control plane YAML structure."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_ACP = _REPO / "governance" / "agent-control-plane"


def main() -> int:
    role_path = _ACP / "role_registry.yaml"
    data = yaml.safe_load(role_path.read_text(encoding="utf-8"))
    roles = data.get("roles") if isinstance(data, dict) else None
    if not isinstance(roles, dict) or len(roles) != 7:
        print("[ERROR] role_registry: expected exactly 7 roles", file=sys.stderr)
        return 1
    if data.get("default_role") != "orchestrator":
        print("[ERROR] role_registry: default_role must be orchestrator", file=sys.stderr)
        return 1
    sub = data.get("subagent_role_map")
    if not isinstance(sub, dict) or not sub:
        print("[ERROR] role_registry: missing subagent_role_map", file=sys.stderr)
        return 1
    print("[INFO] verify_agent_control_plane: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
