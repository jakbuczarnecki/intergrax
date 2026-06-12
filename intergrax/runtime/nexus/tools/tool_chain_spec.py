# © Artur Czarnecki. All rights reserved.

"""Deterministic tool chain specification (TOOL-ENG-20)."""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field

USER_QUERY_SOURCE = "__user_query__"
ChainInputSource = Union[str, "FieldRef"]


class FieldRef(BaseModel):
    """Reference a field from a prior chain step output."""

    step: int = Field(ge=0)
    field: str = Field(min_length=1)


class ChainStep(BaseModel):
    """One tool invocation in a fixed pipeline."""

    tool_id: str = Field(min_length=1)
    step_id: str = ""
    input_mappings: dict[str, ChainInputSource] = Field(
        default_factory=dict,
        description="Input field name → literal, user query sentinel, or FieldRef.",
    )


class ToolChainSpec(BaseModel):
    """Ordered deterministic tool pipeline — no LLM between steps."""

    steps: list[ChainStep] = Field(min_length=1)
