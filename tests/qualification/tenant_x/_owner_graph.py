# © Artur Czarnecki. All rights reserved.

"""TENANT-X identity / isolation ownership graph (qualification non-authoritative)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._types import TenantOwnerAssignment

TENANT_OWNER_GRAPH: Final[tuple[TenantOwnerAssignment, ...]] = (
    TenantOwnerAssignment(
        responsibility="tenant_identity_ownership",
        owner="CANDIDATE_OWNER: EXECUTION (RuntimeRequest / TaskEnvelope admission)",
        surfaces=(
            "intergrax/runtime/execution/agent_runtime_io.py::RuntimeRequest",
            "intergrax/contracts/runtime_execution_admission.py::RootExecutionAuthorityAdmission",
        ),
    ),
    TenantOwnerAssignment(
        responsibility="tenant_validation_ownership",
        owner="CANDIDATE_OWNER: GOVERNANCE + contract validators",
        surfaces=(
            "intergrax/contracts/runtime_inspection/query.py::RuntimeInspectionQuery",
            "intergrax/contracts/agent_runtime_governance.py",
        ),
    ),
    TenantOwnerAssignment(
        responsibility="propagation_responsibility",
        owner="CANDIDATE_OWNER: EXECUTION orchestration + Governance admission bridge",
        surfaces=("intergrax/runtime/execution/", "intergrax/governance/"),
    ),
    TenantOwnerAssignment(
        responsibility="state_isolation_responsibility",
        owner="CANDIDATE_OWNER: STATE_RECOVERY (runtime schema + checkpoint stores)",
        surfaces=("intergrax/runtime/schema/", "intergrax/runtime/checkpoint/", "agents/persistence/"),
    ),
    TenantOwnerAssignment(
        responsibility="provider_config_tenant_binding",
        owner="CANDIDATE_OWNER: INTEGRATIONS (Integration Catalog + CONFIG realization)",
        surfaces=("intergrax/integrations/", "intergrax/integrations/_shared/p3/configs.py"),
    ),
    TenantOwnerAssignment(
        responsibility="credential_resolution_tenant_binding",
        owner="CANDIDATE_OWNER: INTEGRATIONS_CREDENTIALS (CredentialBroker)",
        surfaces=("intergrax/integrations/credentials/broker.py",),
    ),
    TenantOwnerAssignment(
        responsibility="recovery_tenant_continuity",
        owner="CANDIDATE_OWNER: STATE_RECOVERY (resume/replay/checkpoint coordinators)",
        surfaces=("intergrax/runtime/recovery/", "intergrax/runtime/replay/"),
    ),
    TenantOwnerAssignment(
        responsibility="evidence_tenant_continuity",
        owner="CANDIDATE_OWNER: TRACEABILITY (functional evidence + runtime events)",
        surfaces=(
            "intergrax/contracts/functional_evidence/tenant.py",
            "intergrax/contracts/runtime_event.py",
        ),
    ),
    TenantOwnerAssignment(
        responsibility="cross_tenant_authority",
        owner="CANDIDATE_OWNER: GOVERNANCE (explicit policy only)",
        surfaces=("LEGAL_CROSS_TENANT_BOUNDARY",),
    ),
)
