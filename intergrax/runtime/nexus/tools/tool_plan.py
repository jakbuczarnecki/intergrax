# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel

from intergrax.runtime.nexus.planning.stepplan_models import StepId


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    step_id: StepId
    tool_id: str
    input: BaseModel  # already validated by planner schema


@dataclass(frozen=True, slots=True)
class ToolCallPlan:
    calls: List[PlannedToolCall]
