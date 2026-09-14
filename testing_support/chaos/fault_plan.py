# © Artur Czarnecki. All rights reserved.

"""Typed deterministic fault plans for chaos qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, TypeVar

T = TypeVar("T")


class FaultInjectionPoint(StrEnum):
    EXECUTION_WORKER = "execution_worker"
    DEPENDENCY_TOOL = "dependency_tool"
    PERSISTENCE = "persistence"
    EVIDENCE = "evidence"
    CHECKPOINT = "checkpoint"
    CAPACITY = "capacity"
    CHILD_EXECUTION = "child_execution"
    RECOVERY = "recovery"
    OBSERVABILITY_EXPORT = "observability_export"
    CANCELLATION = "cancellation"
    TIMEOUT = "timeout"
    COMPOUND = "compound"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class FailOnCall:
    """Raise on the N-th invocation (1-based)."""

    call_number: int
    message: str = "deterministic_fault"

    def should_fail(self, invocation_index: int) -> bool:
        return invocation_index == self.call_number


def call_counter() -> Callable[[], int]:
    state = {"n": 0}

    def bump() -> int:
        state["n"] += 1
        return state["n"]

    return bump


def raise_on_call(
    plan: FailOnCall,
    *,
    counter: Callable[[], int],
    exc_factory: Callable[[str], Exception] | None = None,
) -> None:
    index = counter()
    if plan.should_fail(index):
        factory = exc_factory or (lambda msg: RuntimeError(msg))
        raise factory(plan.message)
