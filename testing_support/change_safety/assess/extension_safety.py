# © Artur Czarnecki. All rights reserved.

"""W4-05 extension safety: frozen-core touch classifier (passive; never grants authorization)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory

_CLASS_RANK = {"A": 1, "B": 2, "C": 3}


class ExtensionVerdict(StrEnum):
    EXTENSION_ONLY = "EXTENSION_ONLY"
    CORE_TOUCH_AUTHORIZED = "CORE_TOUCH_AUTHORIZED"
    CORE_TOUCH_UNAUTHORIZED = "CORE_TOUCH_UNAUTHORIZED"
    UNCLASSIFIED = "UNCLASSIFIED"  # path in no registry row: governance fail-safe => treated as Class C


@dataclass(frozen=True)
class FrozenCoreTouch:
    path: str
    surface_id: str
    governance_surface: str
    change_class_default: str
    authorization_required: bool


@dataclass(frozen=True)
class ExtensionSafetyReport:
    verdict: ExtensionVerdict
    touches: tuple[FrozenCoreTouch, ...]
    unauthorized_surfaces: tuple[str, ...]
    unclassified_paths: tuple[str, ...]
    effective_class: str  # A|B|C (fail-safe: C when anything unclassified or unauthorized)


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _match_surface(path: str, surfaces: tuple[dict[str, object], ...]) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for row in surfaces:
        prefix = str(row["path_prefix"])
        if path == prefix or (prefix.endswith("/") and path.startswith(prefix)):
            if best is None or len(prefix) > len(str(best["path_prefix"])):
                best = row
    return best


def classify_frozen_core_touches(changed_paths: frozenset[str]) -> tuple[FrozenCoreTouch, ...]:
    registry = load_evol_x_data_document("frozen_core_registry.v1.json")
    surfaces = tuple(registry["surfaces"])
    out: list[FrozenCoreTouch] = []
    for path in sorted(_norm(p) for p in changed_paths):
        row = _match_surface(path, surfaces)
        if row is not None:
            out.append(
                FrozenCoreTouch(
                    path=path,
                    surface_id=str(row["surface_id"]),
                    governance_surface=str(row["governance_surface"]),
                    change_class_default=str(row["change_class_default"]),
                    authorization_required=bool(row["authorization_required"]),
                )
            )
    return tuple(out)


def assess_extension_safety(
    *, changed_paths: frozenset[str], authorized_surfaces: frozenset[str] = frozenset()
) -> ExtensionSafetyReport:
    """``authorized_surfaces``: surface ids for which an operator-signed reopen/authorization record exists."""
    registry = load_evol_x_data_document("frozen_core_registry.v1.json")
    ext_prefixes = tuple(str(r["path_prefix"]) for r in registry["extension_points"])
    paths = frozenset(_norm(p) for p in changed_paths)
    touches = classify_frozen_core_touches(paths)
    touched_paths = {t.path for t in touches}
    unclassified = tuple(
        sorted(
            p
            for p in paths
            if p not in touched_paths
            and not p.endswith((".md", ".json"))
            and not p.startswith(ext_prefixes + ("tests/", "docs/"))
        )
    )
    unauthorized = tuple(sorted({t.surface_id for t in touches if t.authorization_required} - authorized_surfaces))
    if unauthorized:
        verdict = ExtensionVerdict.CORE_TOUCH_UNAUTHORIZED
    elif touches:
        verdict = ExtensionVerdict.CORE_TOUCH_AUTHORIZED
    elif unclassified:
        verdict = ExtensionVerdict.UNCLASSIFIED
    else:
        verdict = ExtensionVerdict.EXTENSION_ONLY
    rank = max([_CLASS_RANK[t.change_class_default] for t in touches] or [1])
    if unclassified or unauthorized:
        rank = 3
    return ExtensionSafetyReport(
        verdict=verdict,
        touches=touches,
        unauthorized_surfaces=unauthorized,
        unclassified_paths=unclassified,
        effective_class={v: k for k, v in _CLASS_RANK.items()}[rank],
    )


def assess_frozen_core_touch(*, changed_paths: frozenset[str]) -> bool:
    return bool(classify_frozen_core_touches(changed_paths))


def swap_rewrite_set(
    inventory: ClosedWorldInventory,
    *,
    provider_module: str,
    composition_modules: frozenset[str],
) -> tuple[str, ...]:
    """S4/W3-03: consumers of ``provider_module`` that are NOT composition roots => consumer rewrite required."""
    from testing_support.change_safety.assess.consumers import consumers_of_targets

    consumers = consumers_of_targets(inventory, frozenset({provider_module}))
    return tuple(sorted(c for c in consumers - composition_modules if c.startswith("module:")))
