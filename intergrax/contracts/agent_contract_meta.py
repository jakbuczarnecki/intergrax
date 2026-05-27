# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentExecutionMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class AgentRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentContract(BaseModel):
    """
    Declarative agent metadata (canonical architecture §12).

    Describes what an agent can do; Nexus decides when to invoke it.
    """

    id: str
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    allowed_tools: List[str] = Field(default_factory=list)
    required_adapters: List[str] = Field(default_factory=list)
    execution_mode: AgentExecutionMode = AgentExecutionMode.ASYNC
    max_steps: Optional[int] = None
    max_duration_seconds: Optional[float] = None
    max_cost: Optional[float] = None
    risk_level: AgentRiskLevel = AgentRiskLevel.MEDIUM
    validation_rules: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
