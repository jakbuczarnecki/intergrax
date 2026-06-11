#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — Nexus routes by capability token, not agent class names (ACP-CON-6)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROUTING_KEYS = (
    "agent_class",
    "agent_import_path",
    "agent_module",
)


def _forbidden_routing_literals(path: Path) -> list[str]:
    """Reject class-name routing keys in Nexus selection paths (not registry bootstrap)."""
    hits: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel.endswith("registry/bootstrap.py"):
        return hits
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for key in FORBIDDEN_ROUTING_KEYS:
            if f'"{key}"' in stripped or f"'{key}'" in stripped:
                if "forbidden" in stripped.lower() or "TaskRouting" in stripped:
                    continue
                hits.append(f"{rel}:{line_no}: routing key {key!r}")
    return hits


def main() -> int:
    violations: list[str] = []

    router = REPO_ROOT / "intergrax" / "runtime" / "nexus" / "agent_router.py"
    router_text = router.read_text(encoding="utf-8")
    if "validate_task_for_capability_routing" not in router_text:
        violations.append("agent_router.py must call validate_task_for_capability_routing")
    if "select_best_capability_match" not in router_text:
        violations.append("agent_router.py must use select_best_capability_match for capability path")

    routing_module = REPO_ROOT / "intergrax" / "runtime" / "registry" / "capability_routing.py"
    if not routing_module.is_file():
        violations.append("missing intergrax/runtime/registry/capability_routing.py")

    nexus_root = REPO_ROOT / "intergrax" / "runtime" / "nexus"
    for path in sorted(nexus_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        violations.extend(_forbidden_routing_literals(path))

    task_routing = REPO_ROOT / "intergrax" / "contracts" / "task_routing.py"
    if not task_routing.is_file():
        violations.append("missing intergrax/contracts/task_routing.py")
    else:
        tree = ast.parse(task_routing.read_text(encoding="utf-8"))
        class_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.endswith("Error")
        }
        if "TaskRoutingViolationError" not in class_names:
            violations.append("task_routing.py must define TaskRoutingViolationError")

    if violations:
        print("Capability routing gate violations:")
        print("\n".join(violations))
        return 1

    print("Capability routing gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
