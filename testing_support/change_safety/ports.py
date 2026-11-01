# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChangeSet:
    paths: frozenset[str]
    base_sha: str
    head_sha: str


class ChangeSetPort(Protocol):
    def load(self, *, base_sha: str, head_sha: str) -> ChangeSet: ...


class SurfaceSourcePort(Protocol):
    def list_surfaces(self) -> frozenset[str]: ...


class SuiteCatalogPort(Protocol):
    def collectable_node_ids(self) -> frozenset[str]: ...

    def mandatory_node_ids(self) -> frozenset[str]: ...
