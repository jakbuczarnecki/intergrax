# © Artur Czarnecki. All rights reserved.

"""TENANT-X identity / isolation ownership graph (no duplicate authority)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._types import TenantOwnerAssignment

TENANT_OWNER_GRAPH: Final[tuple[TenantOwnerAssignment, ...]] = (
    TenantOwnerAssignment(
        responsibility="tenant_identity_ownership",
        owner="EXECUTION (RuntimeRequest / TaskEnvelope admission)",
        surfaces=(
            "intergrax/runtime/execution/agent_runtime_io.py::RuntimeRequest",
            "intergrax/contracts/runtime_execution_admission.py::RuntimeExecutionAdmission",
        ),
    ),
    TenantOwnerAssignment(
        responsibility="tenant_validation_ownership",
        owner="GOVERNANCE + contract validators (fail-closed on missing/mismatch)",
        surfaces=(
            "intergrax/contracts/runtime_inspection/query.py::RuntimeInspectionQuery",
            "intergrax/contracts/agent_runtime_governance.py",
        ),
    ),
    TenantOwnerAssignment(
        responsibility="propagation_responsibility",
        owner="EXECUTION orchestration + Governance admission bridge",
        surfaces=("intergrax/runtime/execution/", "intergrax/governance/"),
    ),
    TenantOwnerAssignment(
        responsibility="state_isolation_responsibility",
        owner="STATE_RECOVERY (runtime schema + checkpoint stores)",
        surfaces=("intergrax/runtime/schema/", "intergrax/runtime/checkpoint/", "agents/persistence/"),
    ),
    TenantOwnerAssignment(
        responsibility="provider_config_tenant_binding",
        owner="INTEGRATIONS (Integration Catalog + CONFIG realization)",
        surfaces=("intergrax/integrations/", "intergrax/integrations/_shared/p3/configs.py"),
    ),
    TenantOwnerAssignment(
        responsibility="credential_resolution_tenant_binding",
        owner="INTEGRATIONS_CREDENTIALS (CredentialBroker)",
        surfaces=("intergrax/integrations/credentials/broker.py",),
    ),
    TenantOwnerAssignment(
        responsibility="recovery_tenant_continuity",
        owner="STATE_RECOVERY (resume/replay/checkpoint coordinators)",
        surfaces=("intergrax/runtime/recovery/", "intergrax/runtime/replay/"),
    ),
    TenantOwnerAssignment(
        responsibility="evidence_tenant_continuity",
        owner="TRACEABILITY (functional evidence + runtime events)",
        surfaces=(
            "intergrax/contracts/functional_evidence/tenant.py",
            "intergrax/contracts/runtime_event.py",
        ),
    ),
    TenantOwnerAssignment(
        responsibility="cross_tenant_authority",
        owner="GOVERNANCE (explicit policy only; no implicit admin bypass)",
        surfaces=("LEGAL_CROSS_TENANT_BOUNDARY",),
    ),
)
