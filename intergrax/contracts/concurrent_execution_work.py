# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed concurrency policy for canonical concurrent Execution work (W1)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MAX_CONCURRENT_EXECUTION_WORK: Final = 64


class ConcurrentExecutionWorkPolicy(BaseModel):
    """Explicit upper bound on in-flight concurrent work units per call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_concurrency: int = Field(ge=1, le=MAX_CONCURRENT_EXECUTION_WORK)


DEFAULT_CONCURRENT_EXECUTION_WORK_POLICY = ConcurrentExecutionWorkPolicy(
    max_concurrency=MAX_CONCURRENT_EXECUTION_WORK,
)
