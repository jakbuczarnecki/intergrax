# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic subject reference port surface for Problem persistence indexes (HARDENING-8)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProblemGroupingSubjectRef(Protocol):
    """Stable identity for one diagnostic subject in a grouping invocation."""

    @property
    def tenant_id(self) -> str: ...

    @property
    def index_token(self) -> str: ...


__all__ = ["ProblemGroupingSubjectRef"]
