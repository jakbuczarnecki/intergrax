# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    step_id: str
    tool_id: str
    input: BaseModel  # already validated by planner schema


@dataclass(frozen=True, slots=True)
class ToolCallPlan:
    calls: List[PlannedToolCall]
