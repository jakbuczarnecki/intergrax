# © Artur Czarnecki. All rights reserved.

"""Compliance profile templates per regulated domain class (AUDIT-IDEAL-5.2)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.task_envelope import TaskRiskTier
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy


class ComplianceDomainClass(str, Enum):
    REGULATED = "regulated"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"


class ComplianceProfileTemplate(BaseModel):
    """Typed compliance posture for a regulated domain class."""

    model_config = ConfigDict(extra="forbid")

    domain_class: ComplianceDomainClass
    require_human_on_critical: bool = True
    data_compliance: DataCompliancePolicy = Field(default_factory=DataCompliancePolicy)
    retention_days: int = Field(default=90, ge=1, le=3650)
    audit_trail_required: bool = True


_COMPLIANCE_TEMPLATES: dict[ComplianceDomainClass, ComplianceProfileTemplate] = {
    ComplianceDomainClass.REGULATED: ComplianceProfileTemplate(
        domain_class=ComplianceDomainClass.REGULATED,
        require_human_on_critical=True,
        data_compliance=DataCompliancePolicy(api_trace_export="redacted", redact_tool_calls_in_api=True),
        retention_days=180,
        audit_trail_required=True,
    ),
    ComplianceDomainClass.HEALTHCARE: ComplianceProfileTemplate(
        domain_class=ComplianceDomainClass.HEALTHCARE,
        require_human_on_critical=True,
        data_compliance=DataCompliancePolicy(api_trace_export="none", redact_tool_calls_in_api=True),
        retention_days=365,
        audit_trail_required=True,
    ),
    ComplianceDomainClass.FINANCIAL: ComplianceProfileTemplate(
        domain_class=ComplianceDomainClass.FINANCIAL,
        require_human_on_critical=True,
        data_compliance=DataCompliancePolicy(api_trace_export="redacted", redact_tool_calls_in_api=True),
        retention_days=2555,
        audit_trail_required=True,
    ),
}


def resolve_compliance_template(domain_class: ComplianceDomainClass) -> ComplianceProfileTemplate:
    return _COMPLIANCE_TEMPLATES[domain_class]


def compliance_template_for_risk_tier(risk_tier: TaskRiskTier) -> ComplianceProfileTemplate | None:
    if risk_tier is TaskRiskTier.REGULATED:
        return resolve_compliance_template(ComplianceDomainClass.REGULATED)
    return None


def compliance_domain_fragments(template: ComplianceProfileTemplate) -> dict[str, Any]:
    return {
        "compliance_profile": {
            "domain_class": template.domain_class.value,
            "retention_days": template.retention_days,
            "audit_trail_required": template.audit_trail_required,
        },
        "data_compliance": {
            "api_trace_export": template.data_compliance.api_trace_export,
            "redact_tool_calls_in_api": template.data_compliance.redact_tool_calls_in_api,
        },
    }
