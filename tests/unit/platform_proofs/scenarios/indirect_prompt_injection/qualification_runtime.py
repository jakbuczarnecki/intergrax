"""Shared helpers for IPI deterministic qualification tests."""

from __future__ import annotations


def execution_engine_importable() -> bool:
    try:
        from intergrax.applications._shared import scenario_runtime_baseline  # noqa: F401

        return True
    except TypeError:
        return False
