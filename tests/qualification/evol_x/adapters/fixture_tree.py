# © Artur Czarnecki. All rights reserved.

"""Build closed-world inventories over plain directory trees (synthetic fixtures; no git)."""

from __future__ import annotations

from pathlib import Path

from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory, build_closed_world_inventory

FIXTURE_ROOTS = ("intergrax/", "agents/", "applications/")


def write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return root


def inventory_for(root: Path, *, sha: str = "fixture") -> ClosedWorldInventory:
    return build_closed_world_inventory(source_commit_sha=sha, roots=FIXTURE_ROOTS, fixture_root=root)
