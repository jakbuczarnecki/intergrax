# © Artur Czarnecki. All rights reserved.

"""Lazy module export helpers without getattr/setattr."""

from __future__ import annotations

import importlib
from types import ModuleType


def export_from_module(module: ModuleType, attr: str) -> object:
    try:
        return module.__dict__[attr]
    except KeyError as exc:
        raise AttributeError(f"module {module.__name__!r} has no attribute {attr!r}") from exc


def export_from_bundle(bundle: ModuleType, name: str, allowed: frozenset[str]) -> object:
    if name not in allowed:
        raise AttributeError(f"module {bundle.__name__!r} has no attribute {name!r}")
    return export_from_module(bundle, name)


def export_from_import_path(module_path: str, attr: str) -> object:
    return export_from_module(importlib.import_module(module_path), attr)
