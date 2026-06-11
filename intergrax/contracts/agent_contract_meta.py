# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.memory_scope import MemoryScope
from intergrax.contracts.agent_contract_section12 import (
    DEFAULT_FAILURE_MODES,
    DEFAULT_INPUT_SCHEMA,
    DEFAULT_OUTPUT_SCHEMA,
    DEFAULT_VALIDATION_RULES,
)
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
    input_schema: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_INPUT_SCHEMA))
    output_schema: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_OUTPUT_SCHEMA))
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
    memory_scope: MemoryScope = Field(
        default=MemoryScope.USER,
        description="Default memory namespace scope (architecture §30.9).",
    )
    memory_namespace_template: Optional[str] = Field(
        default=None,
        description="Template for custom scope; placeholders §30.9.",
    )
    default_rag_collection: Optional[str] = Field(
        default=None,
        description="Default RAG collection id for this agent.",
    )
    prompt_binding_id: Optional[str] = Field(
        default=None,
        description="Primary prompt registry id bound to this agent (IDEAL-17.5).",
    )
    modality_profile_id: Optional[str] = Field(
        default=None,
        description="Optional modality profile id for media/tool plane filtering (IDEAL-18.2).",
    )
    validation_rules: List[str] = Field(
        default_factory=lambda: list(DEFAULT_VALIDATION_RULES),
    )
    failure_modes: List[str] = Field(
        default_factory=lambda: list(DEFAULT_FAILURE_MODES),
    )
