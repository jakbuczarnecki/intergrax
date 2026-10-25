# © Artur Czarnecki. All rights reserved.

"""Catalog tool child execution port contract (TOOL-LIFE-X R3.1 / OD-CHILD-ARCH)."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from pydantic import BaseModel

WorkT = TypeVar("WorkT", contravariant=True)


@runtime_checkable
class CatalogToolChildExecutionPort(Protocol[WorkT]):
    """Host-facing port for canonical in-process catalog child admission."""

    async def execute_catalog_child_invoke(
        self,
        work: WorkT,
        *,
        admission_hooks: tuple[object, ...] = (),
    ) -> ToolExecutionResult[BaseModel]: ...


__all__ = ["CatalogToolChildExecutionPort"]
