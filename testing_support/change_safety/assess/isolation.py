# © Artur Czarnecki. All rights reserved.

"""W4-01 capability blast-radius / application isolation assessment (passive, read-only)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.contracts.impact import AffectedNode, ImpactAssessmentResult

_APP_PREFIX = "module:applications."


def application_of_node(node_id: str) -> str | None:
    """``module:applications.<app>...`` -> ``<app>``; anything else -> None."""
    if not node_id.startswith(_APP_PREFIX):
        return None
    rest = node_id[len(_APP_PREFIX) :]
    app = rest.split(".", 1)[0]
    return app or None


def application_of_path(path: str) -> str | None:
    norm = path.replace("\\", "/")
    if not norm.startswith("applications/"):
        return None
    parts = norm.split("/")
    return parts[1] if len(parts) > 2 else None


def _through_shared(witness: tuple[str, ...], shared_prefixes: frozenset[str]) -> bool:
    for step in witness:
        mod = step[len("module:") :] if step.startswith("module:") else step
        for prefix in shared_prefixes:
            if mod == prefix or mod.startswith(prefix + "."):
                return True
    return False


@dataclass(frozen=True)
class IsolationReport:
    origin_apps: tuple[str, ...]
    blast_apps: tuple[str, ...]
    escapes: tuple[str, ...]  # affected node ids that crossed an application boundary undeclared
    core_nodes: tuple[str, ...]  # affected modules outside applications/ (platform/agents), informative
    isolated: bool


def assess_isolation(
    impact: ImpactAssessmentResult,
    *,
    changed_paths: frozenset[str],
    declared_shared: frozenset[str] = frozenset(),
) -> IsolationReport:
    """Blast radius of ``changed_paths`` restricted to applications; undeclared cross-app reach = escape.

    ``declared_shared``: dotted module prefixes (e.g. ``applications.shared``) through which
    cross-application consumption is explicitly declared and therefore not an escape.
    """
    origin = tuple(sorted({a for a in (application_of_path(p) for p in changed_paths) if a}))
    blast: set[str] = set()
    escapes: list[str] = []
    core: list[str] = []
    for node in impact.affected:
        app = application_of_node(node.node_id)
        if app is None:
            if node.node_id.startswith("module:"):
                core.append(node.node_id)
            continue
        blast.add(app)
        if origin and app not in origin and not _through_shared(node.witness_path, declared_shared):
            escapes.append(node.node_id)
    return IsolationReport(
        origin_apps=origin,
        blast_apps=tuple(sorted(blast)),
        escapes=tuple(sorted(escapes)),
        core_nodes=tuple(sorted(core)),
        isolated=not escapes,
    )


def detect_cross_application_blast(
    affected: tuple[AffectedNode, ...],
    *,
    declared_shared: frozenset[str],
) -> tuple[str, ...]:
    """Backward-compatible helper: affected application nodes outside ``declared_shared`` paths."""
    hits: list[str] = []
    for node in affected:
        path = node.witness_path[0] if node.witness_path else node.node_id
        if path.startswith("applications/") and path not in declared_shared:
            hits.append(node.node_id)
    return tuple(sorted(hits))
