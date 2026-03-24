# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_tool_plan import LegalToolPlan
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
    reason: str = Field(
        default="",
        description="Why the clause is sensitive; LLM may send null — coerced to empty.",
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _coerce_sensitive_reason(cls, v: object) -> str:
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)


PolicyViolationSeverity = Literal["LOW", "MEDIUM", "HIGH"]


class PolicyViolation(BaseModel):
    clause_id: str
    policy_rule: str
    violation: str
    suggested_fix: str
    severity: PolicyViolationSeverity


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


LegalRecommendationAction = Literal["modify", "remove", "add", "review"]
LegalRecommendationPriority = Literal["LOW", "MEDIUM", "HIGH"]


class LegalRecommendation(BaseModel):
    clause_id: str
    action: LegalRecommendationAction
    priority: LegalRecommendationPriority
    recommendation: str
    suggested_text: Optional[str] = None


DecisionStatus = Literal["APPROVE", "REJECT", "CONDITIONAL", "ESCALATE"]


class LegalDecision(BaseModel):
    status: DecisionStatus
    confidence: float = Field(ge=0.0, le=1.0)
    blocking_issues: List[str] = Field(default_factory=list)
    summary: str = Field(
        default="",
        description="Short rationale; LLMs often omit — default empty.",
    )

    @field_validator("summary", mode="before")
    @classmethod
    def _coerce_summary(cls, v: object) -> str:
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)


# -------------------------
# MAIN AGENT STATE
# -------------------------

class LegalAgentState(AgentState, BaseModel):
    """
    Typed state shared across LegalAgent steps.
    """
    
    config: LegalAgentConfig
    
    clauses: List[Clause] = Field(
        default_factory=list,
        description=(
            "Filled by LegalExtractClausesStep; optionally merged/deduped by "
            "LegalNormalizeClausesStep before risk and downstream steps."
        ),
    )
    sensitive_flags: List[SensitiveFlag] = Field(default_factory=list)

    compliance_results: List[ComplianceResult] = Field(default_factory=list)
    legal_checks: List[LegalCheck] = Field(default_factory=list)

    uncertainties: List[Uncertainty] = Field(default_factory=list)

    policy_violations: Optional[List[PolicyViolation]] = Field(
        default=None,
        description="Filled by LegalPolicyComplianceStep; None until that step runs.",
    )

    recommendations: List[LegalRecommendation] = Field(
        default_factory=list,
        description="Filled by LegalRecommendationStep.",
    )

    decision: Optional[LegalDecision] = Field(
        default=None,
        description="Filled by LegalDecisionStep; may be tightened by LegalDecisionEnforcementStep.",
    )

    decision_pre_enforcement_status: Optional[DecisionStatus] = Field(
        default=None,
        description="Snapshot of decision.status before enforcement rules (set by LegalDecisionEnforcementStep).",
    )
    decision_enforcement_modified: bool = Field(
        default=False,
        description="True if LegalDecisionEnforcementStep changed decision.status.",
    )

    final_opinion: Optional[LegalOpinion] = None

    legal_stages_completed_this_run: List[str] = Field(
        default_factory=list,
        description=(
            "Stage flag names already executed in the current LegalDynamicPipeline run "
            "(e.g. 'run_extract'); used to avoid duplicate execution across loop iterations."
        ),
    )

    last_legal_tool_plan: Optional[LegalToolPlan] = Field(
        default=None,
        description="Latest Tier-2 tool/RAG/websearch intent before legal stage routing.",
    )

    legal_tool_runtime_feedback_json: str = Field(
        default="{}",
        description=(
            "JSON summary of Nexus RAG/websearch/tools usage after the tool bridge "
            "(feeds legal routing / replan metrics)."
        ),
    )