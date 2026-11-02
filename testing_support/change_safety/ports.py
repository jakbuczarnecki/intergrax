# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChangeSet:
    """Git change set. ``paths`` = every touched path (old **and** new for renames); ``statuses`` = path -> A/M/D/R."""

    paths: frozenset[str]
    base_sha: str
    head_sha: str
    statuses: tuple[tuple[str, str], ...] = ()
    renames: tuple[tuple[str, str], ...] = ()


class ChangeSetPort(Protocol):
    def load(self, *, base_sha: str, head_sha: str) -> ChangeSet: ...


class SurfaceSourcePort(Protocol):
    def list_surfaces(self) -> frozenset[str]: ...


class SuiteCatalogPort(Protocol):
    def collectable_node_ids(self) -> frozenset[str]: ...

    def mandatory_node_ids(self) -> frozenset[str]: ...
