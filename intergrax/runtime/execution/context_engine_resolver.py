# © Artur Czarnecki. All rights reserved.

"""Resolve custom ContextEngine classes from dotted refs (CE-ENG-REF) — EE owner zone."""

from __future__ import annotations

import importlib

from intergrax.context.protocols import ContextEngine
from intergrax.context.registry import ContextPluginRegistry
from intergrax.utils import attribute_access


class ContextEngineImportError(ImportError):
    """Raised when ``engine_ref`` cannot be resolved."""


def load_context_engine(
    engine_ref: str,
    *,
    registry: ContextPluginRegistry,
) -> ContextEngine:
    """Instantiate a ``ContextEngine`` from ``package.module.Class``."""
    module_path, _, class_name = engine_ref.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"Invalid engine_ref: {engine_ref!r}")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ContextEngineImportError(
            f"Cannot import module {module_path!r} for engine_ref {engine_ref!r}",
        ) from exc

    try:
        engine_cls = attribute_access.optional(module, class_name)
    except AttributeError as exc:
        raise ContextEngineImportError(
            f"Module {module_path!r} has no attribute {class_name!r}",
        ) from exc

    if not isinstance(engine_cls, type):
        raise ContextEngineImportError(f"engine_ref {engine_ref!r} is not a class")

    instance = engine_cls(registry=registry)
    if not isinstance(instance, ContextEngine):
        raise ContextEngineImportError(
            f"engine_ref {engine_ref!r} must resolve to ContextEngine implementation",
        )
    return instance


__all__ = ["ContextEngineImportError", "load_context_engine"]
