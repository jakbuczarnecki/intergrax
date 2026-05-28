# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context assembly models for Nexus ContextManager v2 (§28)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions

__all__ = [
    "ContextProvenance",
    "ContextSourceType",
    "ContextSummaryTier",
    "PriorOutputRecord",
    "TaskContextAssemblyOptions",
]


class ContextSourceType(str, Enum):
    TASK_MESSAGE = "task_message"
    DEPENDENCY_OUTPUT = "dependency_output"
    SHARED_CONTEXT = "shared_context"
    HANDOFF = "handoff"
    ARTIFACT = "artifact"


class ContextProvenance(BaseModel):
    """Traceable origin of a context fragment passed to an agent."""

    source_type: ContextSourceType
    source_id: str
    agent_id: Optional[str] = None
    shared_version: Optional[int] = None
    schema_version: str = "context_provenance.v1"


class PriorOutputRecord(BaseModel):
    """Structured prior node output with provenance (evidence vs interpretation split)."""

    node_id: str
    agent_id: str
    summary: str
    evidence: str = ""
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    provenance: ContextProvenance
    schema_version: str = "prior_output_record.v1"
