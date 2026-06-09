# © Artur Czarnecki. All rights reserved.

"""Typed partial-result block for TaskResult (IDEAL-22.4)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PartialResultContract(BaseModel):
    """Structured partial completion payload attached to TaskResult."""

    model_config = ConfigDict(extra="forbid")

    completed_steps: tuple[str, ...] = ()
    failed_step: str | None = None
    partial_answer: str = ""
    recoverable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
