# © Artur Czarnecki. All rights reserved.

"""Deterministic ORION deployment-policy fixtures for the flagship proof."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from intergrax.runtime.vendor_knowledge.live.project_status.project import (
    ProjectStatusReadLiveRequestV1,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
)

PROOF_TENANT_ID = "governed-hybrid-proof"
PROOF_WORKSPACE_ID = "orion-workspace"
PROOF_CONNECTION_REF = "conn.orion.project-status"
PROOF_BINDING_ID = "binding-orion-project-status"
PROOF_INDEXED_BINDING_ID = "binding-deployment-policy"
PROOF_INDEXED_SOURCE_ID = "source-deployment-policy"
PROOF_LIVE_CALL_ID = "orion-status-read"
PROOF_CREDENTIAL_REF = "secrets/governed-hybrid-proof/project-status"
PROOF_NOW = datetime(2026, 8, 19, 14, 0, tzinfo=UTC)
PROOF_DISABLE_IDEMPOTENCY_HASH = "b" * 64

ORION_DEPLOYMENT_QUESTION = "Is ORION ready for deployment?"

DEPLOYMENT_POLICY_FILENAME = "deployment-policy-approved.txt"
DEPLOYMENT_POLICY_CONTENT = """Deployment Policy — Approved

A project is ready for deployment only when:
1. readiness score is at least 90
2. no security blocker is OPEN
""".strip()


def deployment_policy_content_hash() -> str:
    return sha256(DEPLOYMENT_POLICY_CONTENT.encode("utf-8")).hexdigest()


def deployment_policy_file_digest() -> str:
    return f"sha256:{deployment_policy_content_hash()}"


def deployment_policy_logical_path(policy_path: Path) -> str:
    from local_workspace_application.workspaces.idempotency import normalize_source_path

    return normalize_source_path(policy_path)


def deployment_policy_document_id(
    *,
    tenant_id: str = PROOF_TENANT_ID,
    workspace_id: str = PROOF_WORKSPACE_ID,
    source_id: str = PROOF_INDEXED_SOURCE_ID,
    normalized_source_path: str,
    content_hash: str,
) -> str:
    from local_workspace_application.workspaces.idempotency import logical_document_id

    return logical_document_id(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        normalized_source_path=normalized_source_path,
        content_hash=content_hash,
        materialization_scope=None,
    )


def orion_provider_request() -> ProjectStatusReadLiveRequestV1:
    return ProjectStatusReadLiveRequestV1(project_id=ORION_FIXTURE_PROJECT_ID)


def orion_blocker_label(*, status: str) -> str:
    return f"{ORION_FIXTURE_BLOCKER_ID} {status}"
