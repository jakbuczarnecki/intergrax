# © Artur Czarnecki. All rights reserved.

"""Shared HTTP helpers for workspace knowledge configuration mutations."""

from __future__ import annotations

import hashlib
import re

from fastapi import HTTPException, Request, status

from intergrax.fastapi_core.context import get_request_context
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
)

_IF_MATCH_RE = re.compile(r"^WKC/(\d+)$")


def resolve_knowledge_configuration_tenant_id(
    request: Request,
    *,
    header_tenant_id: str | None,
) -> str:
    try:
        context = get_request_context(request)
    except RuntimeError:
        context = None
    if context is not None and context.tenant_id:
        return str(context.tenant_id)
    if header_tenant_id and header_tenant_id.strip():
        return header_tenant_id.strip()
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="tenant_id_required",
    )


def parse_knowledge_configuration_if_match(value: str | None) -> int:
    if value is None or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="knowledge_configuration_if_match_required",
        )
    match = _IF_MATCH_RE.fullmatch(value.strip())
    if match is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="knowledge_configuration_if_match_invalid",
        )
    return int(match.group(1))


def require_knowledge_configuration_idempotency_key(value: str | None) -> str:
    if value is None or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="knowledge_configuration_idempotency_key_required",
        )
    return value.strip()


def hash_knowledge_configuration_idempotency_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def map_knowledge_configuration_mutation_error(
    exc: WorkspaceKnowledgeConfigurationMutationError,
) -> HTTPException:
    if exc.error_code in {
        "configuration_revision_conflict",
        "configuration_idempotency_conflict",
    }:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.error_code,
        )
    if exc.error_code in {
        "configuration_recovery_required",
        "configuration_mutation_cleanup_failed",
        "configuration_mutation_conditional_write_failed",
    }:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.error_code,
        )
    if exc.error_code.endswith("_unavailable"):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.error_code,
        )
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=exc.error_code,
    )
