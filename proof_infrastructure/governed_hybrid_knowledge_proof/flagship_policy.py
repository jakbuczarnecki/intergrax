# © Artur Czarnecki. All rights reserved.

"""Policy-derived obligation fixtures for COMM-5 F3-F flagship proof."""

from __future__ import annotations

from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
)
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
)
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    MaxAgeTemporalConstraintV1,
    RequireLiveEvidencePolicyRuleV1,
    RequireLiveEvidenceRuleParametersV1,
    ResolvedPolicyRuleV1,
    TypedCapabilityRequestEntryV1,
    ValidAtTemporalConstraintV1,
)
from intergrax.runtime.vendor_knowledge.live.change_approval import (
    CHANGE_APPROVAL_READ_CAPABILITY_ID,
)
from intergrax.runtime.vendor_knowledge.live.governance_approval import (
    GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
)
from intergrax.runtime.vendor_knowledge.live.project_status import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
)
from intergrax.runtime.vendor_knowledge.live.security_status import (
    SECURITY_STATUS_READ_CAPABILITY_ID,
)
from proof_infrastructure.controlled_change_approval_service.seed import (
    ORION_FIXTURE_CHANGE_ID,
)
from proof_infrastructure.controlled_governance_approval_service.seed import (
    ORION_FIXTURE_SUBJECT_ID,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_PROJECT_ID,
)

FLAGSHIP_QUESTION = "Can ORION be deployed to production tonight?"
FLAGSHIP_TENANT_ID = "flagship-proof"
FLAGSHIP_WORKSPACE_ID = "orion-flagship-workspace"
FLAGSHIP_POLICY_REV_17 = "17"
FLAGSHIP_POLICY_REV_18 = "18"

FLAGSHIP_CONN_READINESS = "conn.flagship.project-status"
FLAGSHIP_CONN_SECURITY = "conn.flagship.security-status"
FLAGSHIP_CONN_CHANGE = "conn.flagship.change-approval"
FLAGSHIP_CONN_GOVERNANCE = "conn.flagship.governance-approval"

FLAGSHIP_BINDING_READINESS = "binding-flagship-readiness"
FLAGSHIP_BINDING_SECURITY = "binding-flagship-security"
FLAGSHIP_BINDING_CHANGE = "binding-flagship-change"
FLAGSHIP_BINDING_GOVERNANCE = "binding-flagship-governance"

_REV17_SECURITY_MAX_AGE_SECONDS = 86_400
_REV18_SECURITY_MAX_AGE_SECONDS = 3_600


def _live_rule(
    *,
    policy_document_id: str,
    revision_id: str,
    rule_id: str,
    requirement_key: str,
    semantic_role: str,
    capability_id: str,
    live_access_binding_id: str,
    live_call_descriptor_ref: str,
    typed_request: tuple[TypedCapabilityRequestEntryV1, ...],
    temporal_constraint: MaxAgeTemporalConstraintV1 | ValidAtTemporalConstraintV1 | None = None,
) -> RequireLiveEvidencePolicyRuleV1:
    return RequireLiveEvidencePolicyRuleV1(
        policy_document_id=policy_document_id,
        revision_id=revision_id,
        rule_id=rule_id,
        parameters=RequireLiveEvidenceRuleParametersV1(
            semantic_role=semantic_role,
            requirement_key=requirement_key,
            capability_id=capability_id,
            live_access_binding_id=live_access_binding_id,
            live_call_descriptor_ref=live_call_descriptor_ref,
            typed_capability_request=typed_request,
            temporal_constraint=temporal_constraint,
        ),
    )


def build_flagship_deployment_policy_rules(
    *,
    policy_revision: str,
) -> tuple[ResolvedPolicyRuleV1, ...]:
    security_max_age = (
        _REV17_SECURITY_MAX_AGE_SECONDS
        if policy_revision == FLAGSHIP_POLICY_REV_17
        else _REV18_SECURITY_MAX_AGE_SECONDS
    )
    return (
        _live_rule(
            policy_document_id="deployment-policy",
            revision_id=policy_revision,
            rule_id="RULE-READINESS",
            requirement_key="readiness",
            semantic_role="Project readiness status",
            capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
            live_access_binding_id=FLAGSHIP_BINDING_READINESS,
            live_call_descriptor_ref="readiness-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="project_id",
                    value=ORION_FIXTURE_PROJECT_ID,
                ),
            ),
        ),
        _live_rule(
            policy_document_id="security-policy",
            revision_id=policy_revision,
            rule_id="RULE-SECURITY",
            requirement_key="security",
            semantic_role="Security blocker status",
            capability_id=SECURITY_STATUS_READ_CAPABILITY_ID,
            live_access_binding_id=FLAGSHIP_BINDING_SECURITY,
            live_call_descriptor_ref="security-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="project_id",
                    value=ORION_FIXTURE_PROJECT_ID,
                ),
            ),
            temporal_constraint=MaxAgeTemporalConstraintV1(
                max_age_seconds=security_max_age,
            ),
        ),
        _live_rule(
            policy_document_id="change-policy",
            revision_id=policy_revision,
            rule_id="RULE-CHANGE",
            requirement_key="change",
            semantic_role="Change approval status",
            capability_id=CHANGE_APPROVAL_READ_CAPABILITY_ID,
            live_access_binding_id=FLAGSHIP_BINDING_CHANGE,
            live_call_descriptor_ref="change-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="change_id",
                    value=ORION_FIXTURE_CHANGE_ID,
                ),
            ),
        ),
        _live_rule(
            policy_document_id="architecture-policy",
            revision_id=policy_revision,
            rule_id="RULE-ARCH",
            requirement_key="architecture",
            semantic_role="Architecture approval status",
            capability_id=GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
            live_access_binding_id=FLAGSHIP_BINDING_GOVERNANCE,
            live_call_descriptor_ref="architecture-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="subject_id",
                    value=ORION_FIXTURE_SUBJECT_ID,
                ),
            ),
            temporal_constraint=ValidAtTemporalConstraintV1(),
        ),
    )


FLAGSHIP_PROVIDER_IDS: tuple[str, ...] = (
    PROJECT_STATUS_PROVIDER_ID,
    SECURITY_STATUS_PROVIDER_ID,
    CHANGE_APPROVAL_PROVIDER_ID,
    GOVERNANCE_APPROVAL_PROVIDER_ID,
)

FLAGSHIP_CONNECTION_REFS: tuple[str, ...] = (
    FLAGSHIP_CONN_READINESS,
    FLAGSHIP_CONN_SECURITY,
    FLAGSHIP_CONN_CHANGE,
    FLAGSHIP_CONN_GOVERNANCE,
)

FLAGSHIP_CAPABILITY_IDS: tuple[str, ...] = (
    PROJECT_STATUS_READ_CAPABILITY_ID,
    SECURITY_STATUS_READ_CAPABILITY_ID,
    CHANGE_APPROVAL_READ_CAPABILITY_ID,
    GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
)
