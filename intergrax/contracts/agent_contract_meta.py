# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.skills.core.contracts import SkillManifest
from intergrax.tools.core.contracts import ToolContract


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

    Authors declare ``skills`` (catalog manifests) and ``extra_tools`` (``ToolContract``).
    ``allowed_tools`` is populated at ``AgentRegistry.register`` after skill/tool resolution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    skills: List[SkillManifest] = Field(
        default_factory=list,
        description="Composable skill packs from the skill catalog (§7.1.8).",
    )
    extra_tools: List[ToolContract] = Field(
        default_factory=list,
        description="Additional catalog tools beyond the skill union (ToolContract references).",
    )
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Resolved tool allow-list (skills + extra_tools); set by AgentRegistry.register.",
    )
    required_adapters: List[str] = Field(default_factory=list)
    execution_mode: AgentExecutionMode = AgentExecutionMode.ASYNC
    max_steps: Optional[int] = None
    max_duration_seconds: Optional[float] = None
    max_cost: Optional[float] = None
    risk_level: AgentRiskLevel = AgentRiskLevel.MEDIUM
    lifecycle_state: AgentLifecycleState = AgentLifecycleState.PRODUCTION
    production_eligible: bool = False
    owner_team: Optional[str] = None
    owner_contact: Optional[str] = None
    on_call_contact: Optional[str] = Field(
        default=None,
        description="On-call contact for certified/production agents (AUDIT-IDEAL-31.1).",
    )
    runbook_ref: Optional[str] = None
    prompt_binding_id: Optional[str] = Field(
        default=None,
        description="Primary prompt registry id bound to this agent (IDEAL-17.5).",
    )
    modality_profile_id: Optional[str] = Field(
        default=None,
        description="Optional modality profile id for media/tool plane filtering (IDEAL-18.2).",
    )
    validation_rules: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
