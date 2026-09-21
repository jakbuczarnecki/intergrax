# © Artur Czarnecki. All rights reserved.

"""Shared-context capability port (ACP W2 seam · implementation-neutral)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.shared_context import SharedContextView


class SharedContextAccessPort(Protocol):
    """Minimal load / persist / project capability for ACP shared-context."""

    def load(self) -> SharedContextView | None: ...

    def persist(self, view: SharedContextView) -> None: ...

    def project(self, *, task_id: str) -> SharedContextView: ...
