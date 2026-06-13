# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool bundle catalog (Phase O.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Sequence

from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


class ToolBundleStatus(str, Enum):
    STABLE = "stable"
    BETA = "beta"
    DEPRECATED = "deprecated"


class UnknownToolBundleError(KeyError):
    """Raised when a tool bundle id is not in the catalog."""


ToolRegisterFn = Callable[[ToolRegistry, ToolWiringContext], None]


@dataclass(frozen=True)
class ToolBundleMetadata:
    bundle_id: str
    tool_ids: Sequence[str]
    status: ToolBundleStatus
    description: str


@dataclass(frozen=True)
class ToolBundleEntry:
    """Catalog row: bundle_id → registration function for one tool family."""

    bundle_id: str
    tool_ids: tuple[str, ...]
    register: ToolRegisterFn
    status: ToolBundleStatus = ToolBundleStatus.STABLE
    description: str = ""

    @property
    def metadata(self) -> ToolBundleMetadata:
        return ToolBundleMetadata(
            bundle_id=self.bundle_id,
            tool_ids=self.tool_ids,
            status=self.status,
            description=self.description,
        )


_CATALOG: dict[str, ToolBundleEntry] = {}


def register_tool_bundle(entry: ToolBundleEntry, *, override: bool = False) -> None:
    bundle_id = entry.bundle_id.strip().lower()
    if bundle_id in _CATALOG and not override:
        raise ValueError(f"Tool bundle '{bundle_id}' is already registered.")
    normalized = ToolBundleEntry(
        bundle_id=bundle_id,
        tool_ids=entry.tool_ids,
        register=entry.register,
        status=entry.status,
        description=entry.description,
    )
    _CATALOG[bundle_id] = normalized


def unregister_tool_bundle(bundle_id: str) -> None:
    _CATALOG.pop(bundle_id.strip().lower(), None)


def clear_tool_catalog() -> None:
    """Test helper — reset catalog to empty."""
    _CATALOG.clear()


def get_bundle(bundle_id: str) -> ToolBundleEntry:
    normalized = bundle_id.strip().lower()
    try:
        return _CATALOG[normalized]
    except KeyError as exc:
        raise UnknownToolBundleError(normalized) from exc


def iter_bundles() -> Iterator[ToolBundleEntry]:
    yield from _CATALOG.values()


def list_bundle_ids() -> list[str]:
    return sorted(_CATALOG)


def is_tool_bundle_registered(bundle_id: str) -> bool:
    """Return whether a bundle id is already present in the catalog."""
    return bundle_id.strip().lower() in _CATALOG


def list_catalog_tool_ids() -> list[str]:
    ids: set[str] = set()
    for entry in _CATALOG.values():
        ids.update(entry.tool_ids)
    return sorted(ids)


def metadata_for_bundle(bundle_id: str) -> ToolBundleMetadata:
    return get_bundle(bundle_id).metadata


def catalog_snapshot() -> dict[str, ToolBundleEntry]:
    return dict(_CATALOG)
