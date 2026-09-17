# © Artur Czarnecki. All rights reserved.

"""UE-9D / MEM-XINT-4-R — iterative bounded tool loop context authority policy."""

from __future__ import annotations

from intergrax.context.protocols import ContextEngine


class ContextEngineRequiredForIterativeToolLoopError(RuntimeError):
    """Raised when iterative tool feedback cannot be composed without Context Engineering."""


class SyncIterativeBoundedToolLoopForbiddenError(RuntimeError):
    """Raised when sync bounded tool loop is used for multi-round ReAct (CE is async-only)."""


def require_context_engine_for_iterative_bounded_tool_loop(
    *,
    max_iterations: int,
    context_engine: ContextEngine | None,
) -> None:
    if max_iterations <= 1:
        return
    if context_engine is not None:
        return
    raise ContextEngineRequiredForIterativeToolLoopError(
        "context_engine is required for iterative bounded tool loops; "
        "wire ContextEngine on RuntimeConfig and use run_bounded_tool_loop_async."
    )


def reject_sync_iterative_bounded_tool_loop(max_iterations: int) -> None:
    if max_iterations <= 1:
        return
    raise SyncIterativeBoundedToolLoopForbiddenError(
        "Iterative bounded tool loops must use run_bounded_tool_loop_async with a wired "
        "context_engine; sync BoundedReactPattern cannot compose tool feedback."
    )
