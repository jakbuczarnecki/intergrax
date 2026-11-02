# © Artur Czarnecki. All rights reserved.

"""Topological check schedule with depends_on validation."""

from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import ManifestCheck


def validate_schedule(checks: tuple[ManifestCheck, ...]) -> None:
    known = {c.check_id for c in checks}
    for check in checks:
        orphans = check.depends_on - known
        if orphans:
            raise ValueError(f"depends_on references unknown check_id(s): {orphans} for {check.check_id}")


def topological_order(checks: tuple[ManifestCheck, ...]) -> tuple[ManifestCheck, ...]:
    validate_schedule(checks)
    by_id = {c.check_id: c for c in checks}
    ordered_ids: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(check_id: str) -> None:
        if check_id in visited:
            return
        if check_id in visiting:
            raise ValueError(f"cycle detected in depends_on at check_id={check_id}")
        visiting.add(check_id)
        for dep in by_id[check_id].depends_on:
            visit(dep)
        visiting.remove(check_id)
        visited.add(check_id)
        ordered_ids.append(check_id)

    for check_id in by_id:
        visit(check_id)
    return tuple(by_id[cid] for cid in ordered_ids)
