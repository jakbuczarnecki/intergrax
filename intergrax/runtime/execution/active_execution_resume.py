# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped active resume plan binding for checkpoint lineage."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionTreeResumePlan,
)


@dataclass(frozen=True, slots=True)
class ActiveExecutionResumePlan:
    """Immutable execution-scoped carrier for one resume plan."""

    plan: ExecutionTreeResumePlan


_active_execution_resume_plan: ContextVar[ActiveExecutionResumePlan | None] = ContextVar(
    "active_execution_resume_plan",
    default=None,
)


def bind_active_execution_resume_plan(binding: ActiveExecutionResumePlan) -> Token:
    return _active_execution_resume_plan.set(binding)


def reset_active_execution_resume_plan(token: Token) -> None:
    _active_execution_resume_plan.reset(token)


def peek_active_execution_resume_plan() -> ActiveExecutionResumePlan | None:
    return _active_execution_resume_plan.get()
