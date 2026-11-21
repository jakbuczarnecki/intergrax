# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64
import hashlib
from urllib.parse import quote

from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.tools.providers.storage.contracts import (
    StorageDeleteInput,
    StorageDeleteOutput,
    StorageExistsInput,
    StorageExistsOutput,
    StorageGetInput,
    StorageGetOutput,
    StoragePresignedUrlInput,
    StoragePresignedUrlOutput,
    StoragePutInput,
    StoragePutOutput,
)
from intergrax.tools.registry.wiring import (
    StorageAccessAuditEvent,
    StorageAccessOutcome,
    ToolWiringContext,
)

STORAGE_GET_TOOL_ID = "storage.get"
STORAGE_PUT_TOOL_ID = "storage.put"
STORAGE_PRESIGNED_URL_TOOL_ID = "storage.presigned_url"
STORAGE_DELETE_TOOL_ID = "storage.delete"
STORAGE_EXISTS_TOOL_ID = "storage.exists"

TENANT_OBJECT_KEY_PREFIX = "tenants/"


#: Namespace of Collaborative Work artifact content (every key version). It is reachable only through
#: the governed artifact port; the raw tools below refuse it (T-IGX-041-H, F-D8-03). A drift gate keeps
#: it equal to the prefix of ``collaborative_work.content_storage``.
GOVERNED_CONTENT_KEY_PREFIX = "intergrax/collaborative-work/artifact-content/"


class GovernedStorageKeyError(PermissionError):
    """The key addresses artifact content that is reachable only through the governed artifact port."""


class TenantScopeRequiredError(PermissionError):
    """Raw storage tools require a non-blank tenant on the wiring context."""


class UnsafeTenantSegmentError(PermissionError):
    """The wiring context tenant is not a single safe path segment."""


class TenantNamespaceKeyError(PermissionError):
    """The logical key addresses the tenant namespace root directly."""


def _refuse_governed_content_key(key: str) -> None:
    # Each segment is compared case-insensitively and without trailing dots and spaces: case-insensitive
    # and dot-stripping stores (Windows, macOS) address the same physical key through such spellings.
    segments: list[str] = []
    for part in key.strip().replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise GovernedStorageKeyError("storage keys with '..' segments are refused")
        folded = part.rstrip(". ").casefold()
        if folded:
            segments.append(folded)
    joined = "/".join(segments)
    prefix = GOVERNED_CONTENT_KEY_PREFIX.casefold()
    if joined == prefix.rstrip("/") or joined.startswith(prefix):
        raise GovernedStorageKeyError(
            "this key space holds governed artifact content: use the governed artifact read port"
        )


def refuse_governed_content_key(key: str) -> None:
    """Shared refusal for every agent-callable tool that takes a caller-chosen object storage key."""
    _refuse_governed_content_key(key)


def _logical_key_digest(logical_key: str) -> str:
    return hashlib.sha256(logical_key.strip().encode("utf-8")).hexdigest()


def _require_audit_sink(ctx: ToolWiringContext):
    sink = ctx.storage_audit_sink
    if sink is None:
        raise RuntimeError("storage_audit_sink_not_configured")
    return sink


def _record_storage_access(
    ctx: ToolWiringContext,
    *,
    tool_id: str,
    tenant_id: str,
    outcome: StorageAccessOutcome,
    logical_key: str,
    reason_code: str,
) -> None:
    _require_audit_sink(ctx).record(
        StorageAccessAuditEvent(
            tenant_id=tenant_id,
            tool_id=tool_id,
            outcome=outcome,
            key_digest=_logical_key_digest(logical_key),
            reason_code=reason_code,
        )
    )


def _validate_tenant_segment(tenant_id: str) -> None:
    if tenant_id in ("", ".", "..") or "/" in tenant_id or "\\" in tenant_id:
        raise UnsafeTenantSegmentError("tenant_id must be one safe path segment")


def _scoped_key(ctx: ToolWiringContext, logical_key: str) -> str:
    tenant_raw = (ctx.tenant_id or "").strip()
    if not tenant_raw:
        raise TenantScopeRequiredError("tool wiring context requires a non-blank tenant_id")
    _validate_tenant_segment(tenant_raw)
    logical = logical_key.strip()
    if logical.casefold().startswith(TENANT_OBJECT_KEY_PREFIX.casefold()):
        raise TenantNamespaceKeyError("logical storage keys must not address the tenant namespace root")
    encoded_tenant = quote(tenant_raw, safe="")
    return f"{TENANT_OBJECT_KEY_PREFIX}{encoded_tenant}/{logical}"


def _resolve_scoped_storage_key(
    ctx: ToolWiringContext,
    *,
    tool_id: str,
    logical_key: str,
) -> str:
    logical = logical_key.strip()
    tenant_for_audit = (ctx.tenant_id or "").strip()
    try:
        _refuse_governed_content_key(logical)
    except GovernedStorageKeyError:
        _record_storage_access(
            ctx,
            tool_id=tool_id,
            tenant_id=tenant_for_audit,
            outcome=StorageAccessOutcome.DENIED,
            logical_key=logical,
            reason_code="governed_content_key",
        )
        raise
    try:
        physical_key = _scoped_key(ctx, logical)
    except TenantScopeRequiredError:
        _record_storage_access(
            ctx,
            tool_id=tool_id,
            tenant_id=tenant_for_audit,
            outcome=StorageAccessOutcome.DENIED,
            logical_key=logical,
            reason_code="tenant_scope_required",
        )
        raise
    except UnsafeTenantSegmentError:
        _record_storage_access(
            ctx,
            tool_id=tool_id,
            tenant_id=tenant_for_audit,
            outcome=StorageAccessOutcome.DENIED,
            logical_key=logical,
            reason_code="unsafe_tenant_segment",
        )
        raise
    except TenantNamespaceKeyError:
        _record_storage_access(
            ctx,
            tool_id=tool_id,
            tenant_id=tenant_for_audit,
            outcome=StorageAccessOutcome.DENIED,
            logical_key=logical,
            reason_code="tenant_namespace_key",
        )
        raise
    tenant_for_audit = (ctx.tenant_id or "").strip()
    _record_storage_access(
        ctx,
        tool_id=tool_id,
        tenant_id=tenant_for_audit,
        outcome=StorageAccessOutcome.ALLOWED,
        logical_key=logical,
        reason_code="ok",
    )
    return physical_key


def _require_storage(ctx: ToolWiringContext) -> ObjectStorage:
    storage = ctx.object_storage
    if storage is None:
        raise RuntimeError("object_storage_not_configured")
    return storage


def storage_get(ctx: ToolWiringContext, params: StorageGetInput) -> StorageGetOutput:
    physical_key = _resolve_scoped_storage_key(ctx, tool_id=STORAGE_GET_TOOL_ID, logical_key=params.key)
    stored = _require_storage(ctx).get(physical_key)
    logical = params.key.strip()
    if stored is None:
        return StorageGetOutput(key=logical, found=False)
    return StorageGetOutput(
        key=logical,
        found=True,
        body_base64=base64.b64encode(stored.body).decode("ascii"),
        content_type=stored.content_type,
        size_bytes=stored.size_bytes or len(stored.body),
    )


def storage_put(ctx: ToolWiringContext, params: StoragePutInput) -> StoragePutOutput:
    physical_key = _resolve_scoped_storage_key(ctx, tool_id=STORAGE_PUT_TOOL_ID, logical_key=params.key)
    body = base64.b64decode(params.body_base64)
    _require_storage(ctx).put(
        physical_key,
        body,
        content_type=params.content_type,
    )
    return StoragePutOutput(key=params.key.strip(), stored=True, size_bytes=len(body))


class GovernedPresignRequiredError(RuntimeError):
    """Presigned URLs are issued only through the governed artifact read port."""


def storage_presigned_url(ctx: ToolWiringContext, params: StoragePresignedUrlInput) -> StoragePresignedUrlOutput:
    """Fail closed (PDC-D8-R010, OD-PDC-SCOPE-04 = A).

    This tool used to sign any key for any caller, with no tenant scope, authorization or audit.
    A presigned URL is now issued only by ``GovernedArtifactReadPort.issue_download_url``
    (tenant check, authorization, audit event, short-lived GET). The tool stays registered so
    catalogs and manifests keep resolving, but it never calls the storage provider.
    """
    logical = params.key.strip()
    tenant_for_audit = (ctx.tenant_id or "").strip()
    _record_storage_access(
        ctx,
        tool_id=STORAGE_PRESIGNED_URL_TOOL_ID,
        tenant_id=tenant_for_audit,
        outcome=StorageAccessOutcome.DENIED,
        logical_key=logical,
        reason_code="governed_presign_required",
    )
    raise GovernedPresignRequiredError(
        "storage.presigned_url is disabled: request a download through the governed artifact read port"
    )


def storage_delete(ctx: ToolWiringContext, params: StorageDeleteInput) -> StorageDeleteOutput:
    physical_key = _resolve_scoped_storage_key(ctx, tool_id=STORAGE_DELETE_TOOL_ID, logical_key=params.key)
    _require_storage(ctx).delete(physical_key)
    return StorageDeleteOutput(key=params.key.strip(), deleted=True)


def storage_exists(ctx: ToolWiringContext, params: StorageExistsInput) -> StorageExistsOutput:
    physical_key = _resolve_scoped_storage_key(ctx, tool_id=STORAGE_EXISTS_TOOL_ID, logical_key=params.key)
    stored = _require_storage(ctx).get(physical_key)
    logical = params.key.strip()
    if stored is None:
        return StorageExistsOutput(key=logical, exists=False)
    return StorageExistsOutput(
        key=logical,
        exists=True,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes or len(stored.body),
    )
