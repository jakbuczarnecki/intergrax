# © Artur Czarnecki. All rights reserved.

"""LKW HTTP/execution readiness projection contract (APP-HOST-8B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LocalWorkspaceComponentReadiness:
    name: str
    enabled: bool
    required: bool
    healthy: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class LocalWorkspaceReadinessSnapshot:
    ready: bool
    accepts_new_work: bool
    state: str
    detail: str
    rejection_error_id: str
    components: tuple[LocalWorkspaceComponentReadiness, ...] = ()


@runtime_checkable
class LocalWorkspaceReadinessProvider(Protocol):
    def readiness_snapshot(self) -> LocalWorkspaceReadinessSnapshot:
        ...
