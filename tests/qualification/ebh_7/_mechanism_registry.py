# © Artur Czarnecki. All rights reserved.

"""EBH-7 primary mechanism registry (238 disjoint MECH IDs over the tracked tree)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from tests.qualification.dup_x._closed_world_inventory import (
    DUP_X_RESPONSIBILITY_FAMILIES,
    discover_normative_composition_root_union,
)
from tests.qualification.dup_x._findings_register import DUP_CLASSES
from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG
from tests.qualification.ebh_7._constants import EBH7_MECHANISM_UNIQUE_COUNT


@dataclass(frozen=True, slots=True)
class MechanismRow:
    mechanism_id: str
    domain: str
    semantic_owner: str
    production_path: str


_TIER_FAMILIES: Final[tuple[tuple[str, str], ...]] = (
    ("MECH-TIER-0", "intergrax/"),
    ("MECH-TIER-1", "intergrax/runtime/"),
    ("MECH-TIER-2", "agents/"),
    ("MECH-TIER-3", "applications/"),
)


@lru_cache(maxsize=1)
def primary_mechanism_registry() -> tuple[MechanismRow, ...]:
    rows: list[MechanismRow] = []
    for fam in DUP_X_RESPONSIBILITY_FAMILIES:
        rows.append(
            MechanismRow(
                f"MECH-DUP-RF-{fam.family_id}",
                "duplicate-responsibility",
                fam.canonical_owner_doc,
                fam.implementation_owner_path,
            )
        )
    for dup_id in DUP_CLASSES:
        rows.append(
            MechanismRow(
                f"MECH-DUP-CLS-{dup_id}",
                "duplicate-scan",
                "DUP-X",
                dup_id,
            )
        )
    for fam in EBH6_FAMILY_CATALOG:
        rows.append(
            MechanismRow(
                f"MECH-FAM-{fam.family_id}",
                fam.ecap_theme,
                fam.authority,
                fam.scope_paths[0] if fam.scope_paths else "",
            )
        )
    for concern in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        rows.append(
            MechanismRow(
                f"MECH-VBN-{concern.concern_id}",
                concern.domain.value,
                concern.semantic_owner,
                concern.vendor_code_location,
            )
        )
    for path in sorted(discover_normative_composition_root_union()):
        rows.append(
            MechanismRow(
                f"MECH-CMP-{path.replace('/', '_')[:120]}",
                "composition-root",
                "TRACE-X/CONFIG-X union",
                path,
            )
        )
    from tests.qualification.dup_x._closed_world_inventory import discover_application_host_factory_paths

    for path in sorted(discover_application_host_factory_paths()):
        rows.append(
            MechanismRow(
                f"MECH-HOST-{path.replace('/', '_')}",
                "tier3-host",
                "TIER3_APPLICATION",
                path,
            )
        )
    for tier_id, root in _TIER_FAMILIES:
        rows.append(MechanismRow(tier_id, "tier-executive", "PLATFORM_FOUNDATION", root))
    return tuple(rows)


def assert_ebh7_mechanism_registry_integrity() -> None:
    reg = primary_mechanism_registry()
    assert len(reg) == EBH7_MECHANISM_UNIQUE_COUNT
    assert all(r.mechanism_id and r.semantic_owner for r in reg)
