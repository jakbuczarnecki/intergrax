# © Artur Czarnecki. All rights reserved.

"""Deterministic ORION deployment-policy fixtures for the flagship proof."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

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
PROOF_DOCUMENT_ID = "document-deployment-policy"
PROOF_NOW = datetime(2026, 8, 19, 14, 0, tzinfo=UTC)

ORION_DEPLOYMENT_QUESTION = "Is ORION ready for deployment?"

DEPLOYMENT_POLICY_FILENAME = "deployment-policy-approved.txt"
DEPLOYMENT_POLICY_CONTENT = """Deployment Policy — Approved

A project is ready for deployment only when:
1. readiness score is at least 90
2. no security blocker is OPEN
""".strip()


def deployment_policy_content_hash() -> str:
    return sha256(DEPLOYMENT_POLICY_CONTENT.encode("utf-8")).hexdigest()


def orion_provider_request() -> dict[str, str]:
    return {"project_id": ORION_FIXTURE_PROJECT_ID}


def orion_blocker_label(*, status: str) -> str:
    return f"{ORION_FIXTURE_BLOCKER_ID} {status}"
