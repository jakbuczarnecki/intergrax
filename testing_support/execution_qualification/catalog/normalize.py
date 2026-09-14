# © Artur Czarnecki. All rights reserved.

"""Normalization helpers for canonical qualification catalog."""

from __future__ import annotations


def normalize_pytest_arguments(targets: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(target.replace("\\", "/") for target in targets)
