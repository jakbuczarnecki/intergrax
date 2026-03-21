# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.llm.messages import AttachmentRef
from intergrax.runtime.nexus.engine.contracts.agent_state import AgentState


# -------------------------
# LOW-LEVEL DOMAIN MODELS
# -------------------------

class Clause(BaseModel):
    id: str
    text: str
    category: Optional[str] = None
    is_sensitive: bool = False


class SensitiveFlag(BaseModel):
    clause_id: str
    reason: str


class ComplianceResult(BaseModel):
    clause_id: str
    compliant: bool
    details: Optional[str] = None


class LegalCheck(BaseModel):
    clause_id: str
    valid: bool
    source: Optional[str] = None
    details: Optional[str] = None


class Uncertainty(BaseModel):
    area: str
    description: str


class LegalOpinion(BaseModel):
    summary: str
    risk_level: str  # e.g. LOW / MEDIUM / HIGH
    recommendations: List[str] = Field(default_factory=list)


# -------------------------
# MAIN AGENT STATE
# -------------------------

class LegalAgentState(AgentState, BaseModel):
    """
    Typed state shared across LegalAgent steps.
    """
    
    config: LegalAgentConfig
    
    attachment_refs: List[AttachmentRef] = Field(default_factory=list)

    clauses: List[Clause] = Field(default_factory=list)
    sensitive_flags: List[SensitiveFlag] = Field(default_factory=list)

    compliance_results: List[ComplianceResult] = Field(default_factory=list)
    legal_checks: List[LegalCheck] = Field(default_factory=list)

    uncertainties: List[Uncertainty] = Field(default_factory=list)

    final_opinion: Optional[LegalOpinion] = None    