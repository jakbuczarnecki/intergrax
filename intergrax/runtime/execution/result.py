# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution result envelope (UE-5B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

OutputT = TypeVar("OutputT")


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ExecutionResult(Generic[OutputT]):
    status: ExecutionStatus
    output: OutputT
