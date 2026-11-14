# © Artur Czarnecki. All rights reserved.

"""Helpers shared by the dispatching-wrapper pins (T-IGX-040-A)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from intergrax.integrations.registry.catalog import get_entry, register_integration


def wrapper_depth(factory: Callable[..., object], *, module_name: str) -> int:
    """Number of nested dispatching wrappers defined in ``module_name`` around ``factory``."""
    depth = 0
    current: object = factory
    while getattr(current, "__module__", None) == module_name:
        depth += 1
        cells = getattr(current, "__closure__", None) or ()
        inner = [cell.cell_contents for cell in cells if callable(cell.cell_contents)]
        if not inner:
            break
        current = inner[0]
    return depth


class RecordingFactory:
    """Stand-in owning factory: records every call and returns a prepared integration."""

    def __init__(self, result: Callable[[], object]) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self._result()


def install_recording_owning_factory(slug: str, recorder: RecordingFactory) -> None:
    """Replace the factory of an already registered owning row (before the embedding contribution)."""
    entry = get_entry(slug)
    register_integration(replace(entry, factory=recorder), override=True)


__all__ = [
    "RecordingFactory",
    "install_recording_owning_factory",
    "wrapper_depth",
]
