# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed outputs for catalog context-injection tools (Phase Q+-T.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContextInjectionOutput(Protocol):
    used: bool
    context_text: str
