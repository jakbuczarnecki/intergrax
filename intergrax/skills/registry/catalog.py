# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Sequence

from intergrax.skills.registry.runtime import SkillRegistry


class SkillBundleStatus(str, Enum):
    STABLE = "stable"
    BETA = "beta"


class UnknownSkillBundleError(KeyError):
    """Raised when a skill bundle id is not in the catalog."""


SkillRegisterFn = Callable[[SkillRegistry], None]


@dataclass(frozen=True)
class SkillBundleEntry:
    bundle_id: str
    skill_ids: tuple[str, ...]
    register: SkillRegisterFn
    status: SkillBundleStatus = SkillBundleStatus.STABLE
    description: str = ""

    def register_bundle(self, registry: SkillRegistry) -> None:
        self.register(registry)


_CATALOG: dict[str, SkillBundleEntry] = {}


def register_skill_bundle(entry: SkillBundleEntry, *, override: bool = False) -> None:
    bundle_id = entry.bundle_id.strip().lower()
    if bundle_id in _CATALOG and not override:
        raise ValueError(f"Skill bundle '{bundle_id}' is already registered.")
    _CATALOG[bundle_id] = SkillBundleEntry(
        bundle_id=bundle_id,
        skill_ids=entry.skill_ids,
        register=entry.register,
        status=entry.status,
        description=entry.description,
    )


def clear_skill_catalog() -> None:
    _CATALOG.clear()


def get_bundle(bundle_id: str) -> SkillBundleEntry:
    normalized = bundle_id.strip().lower()
    try:
        return _CATALOG[normalized]
    except KeyError as exc:
        raise UnknownSkillBundleError(normalized) from exc


def iter_bundles() -> Iterator[SkillBundleEntry]:
    yield from _CATALOG.values()


def list_catalog_skill_ids() -> list[str]:
    ids: set[str] = set()
    for entry in _CATALOG.values():
        ids.update(entry.skill_ids)
    return sorted(ids)
