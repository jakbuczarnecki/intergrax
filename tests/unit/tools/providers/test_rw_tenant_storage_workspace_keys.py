# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part B: tenant tool context, scoped object keys, storage audit."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest
from intergrax.integrations.contracts.object_storage import StoredObject
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.providers.storage import service as storage_service
from intergrax.tools.providers.storage.contracts import (
    StorageGetInput,
    StoragePutInput,
)
from intergrax.tools.providers.storage.service import (
    GOVERNED_CONTENT_KEY_PREFIX,
    TenantNamespaceKeyError,
    TenantScopeRequiredError,
    UnsafeTenantSegmentError,
    storage_get,
    storage_put,
)
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceExportArtifactInput,
    WorkspaceImportArtifactInput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.providers.workspace.service import (
    workspace_export_artifact,
    workspace_import_artifact,
    workspace_write_file,
)
from intergrax.tools.registry.wiring import (
    ListStorageAccessAuditSink,
    StorageAccessOutcome,
    ToolWiringContext,
)

pytestmark = pytest.mark.unit


class SharedMemoryStorage:
    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        del metadata
        self._objects[key] = StoredObject(
            key=key,
            body=body,
            content_type=content_type,
            size_bytes=len(body),
        )

    def get(self, key: str) -> StoredObject | None:
        return self._objects.get(key)

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def presigned_url(self, key: str, **kwargs: Any) -> str:
        raise AssertionError("presigned_url must not be called")

    def close(self) -> None:
        return None


def _ctx(
    storage: SharedMemoryStorage,
    *,
    tenant: str,
    sink: ListStorageAccessAuditSink | None = None,
    shadow_workspace: ShadowWorkspace | None = None,
) -> ToolWiringContext:
    return ToolWiringContext(
        object_storage=storage,
        tenant_id=tenant,
        storage_audit_sink=sink if sink is not None else ListStorageAccessAuditSink(),
        shadow_workspace=shadow_workspace,
    )


def test_ten_02a_tool_context_carries_tenant() -> None:
    fields = {field.name for field in dataclasses.fields(ToolWiringContext)}
    assert "tenant_id" in fields


def test_ten_02b_object_written_for_a_not_readable_as_b() -> None:
    storage = SharedMemoryStorage()
    sink_a = ListStorageAccessAuditSink()
    sink_b = ListStorageAccessAuditSink()
    ctx_a = _ctx(storage, tenant="tenant-a", sink=sink_a)
    ctx_b = _ctx(storage, tenant="tenant-b", sink=sink_b)
    body = b"secret-payload"
    storage_put(
        ctx_a,
        StoragePutInput(key="docs/obj.bin", body_base64=base64.b64encode(body).decode("ascii")),
    )
    missing = storage_get(ctx_b, StorageGetInput(key="docs/obj.bin"))
    assert missing.found is False
    own = storage_get(ctx_a, StorageGetInput(key="docs/obj.bin"))
    assert own.found is True
    assert base64.b64decode(own.body_base64) == body


def test_ten_02c_each_raw_call_emits_audit_event() -> None:
    storage = SharedMemoryStorage()
    sink = ListStorageAccessAuditSink()
    ctx = _ctx(storage, tenant="tenant-audit", sink=sink)
    storage_put(
        ctx,
        StoragePutInput(key="k1", body_base64=base64.b64encode(b"x").decode("ascii")),
    )
    storage_get(ctx, StorageGetInput(key="k1"))
    assert len(sink.events) == 2
    for event in sink.events:
        assert event.tenant_id == "tenant-audit"
        assert event.tool_id.startswith("storage.")
        assert len(event.key_digest) == 64


def test_ten_02d_missing_tenant_or_sink_refused() -> None:
    storage = SharedMemoryStorage()
    sink = ListStorageAccessAuditSink()
    with pytest.raises(TenantScopeRequiredError):
        storage_get(
            ToolWiringContext(object_storage=storage, storage_audit_sink=sink),
            StorageGetInput(key="k"),
        )
    with pytest.raises(RuntimeError, match="storage_audit_sink_not_configured"):
        storage_get(
            ToolWiringContext(object_storage=storage, tenant_id="tenant-a"),
            StorageGetInput(key="k"),
        )


def test_ten_03a_foreign_namespace_key_refused() -> None:
    storage = SharedMemoryStorage()
    sink = ListStorageAccessAuditSink()
    ctx = _ctx(storage, tenant="tenant-a", sink=sink)
    with pytest.raises(TenantNamespaceKeyError):
        storage_get(ctx, StorageGetInput(key="tenants/tenant-b/docs/x"))
    assert sink.events[-1].outcome == StorageAccessOutcome.DENIED


def test_ten_03b_workspace_export_import_tenant_bound(tmp_path: Path) -> None:
    storage = SharedMemoryStorage()
    ws_a = ShadowWorkspace.create(tmp_path / "a", tenant_id="tenant-a", task_id="task-a")
    ws_b = ShadowWorkspace.create(tmp_path / "b", tenant_id="tenant-b", task_id="task-b")
    write_ctx = _ctx(storage, tenant="tenant-a", shadow_workspace=ws_a)
    workspace_write_file(write_ctx, WorkspaceWriteFileInput(path="x.txt", content="hi"))
    export = workspace_export_artifact(
        _ctx(storage, tenant="tenant-a", shadow_workspace=ws_a),
        WorkspaceExportArtifactInput(path="x.txt", storage_key="blob.bin"),
    )
    assert export.exported is True
    imported = workspace_import_artifact(
        _ctx(storage, tenant="tenant-b", shadow_workspace=ws_b),
        WorkspaceImportArtifactInput(path="y.txt", storage_key="blob.bin"),
    )
    assert imported.imported is False
    assert imported.reason == "object_not_found"


@pytest.mark.parametrize("tenant", ["", ".", "..", "a/b", r"a\b"])
def test_ten_03c_unsafe_tenant_segment_refused(tenant: str) -> None:
    storage = SharedMemoryStorage()
    ctx = _ctx(storage, tenant=tenant)
    with pytest.raises((TenantScopeRequiredError, UnsafeTenantSegmentError)):
        storage_get(ctx, StorageGetInput(key="safe"))


def test_ten_03d_audit_event_carries_no_content() -> None:
    storage = SharedMemoryStorage()
    sink = ListStorageAccessAuditSink()
    logical = "tenants/other/x"
    ctx = _ctx(storage, tenant="tenant-a", sink=sink)
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()
    with pytest.raises(TenantNamespaceKeyError):
        storage_get(ctx, StorageGetInput(key=logical))
    event = sink.events[-1]
    dumped = dataclasses.asdict(event)
    assert event.key_digest == digest
    assert logical not in str(dumped)
    assert "other/x" not in str(dumped)


def test_ten_03e_governed_prefix_refusal_unchanged() -> None:
    storage = SharedMemoryStorage()
    ctx = _ctx(storage, tenant="tenant-a")
    key = GOVERNED_CONTENT_KEY_PREFIX + "v1/x"
    with pytest.raises(storage_service.GovernedStorageKeyError):
        storage_get(ctx, StorageGetInput(key=key))
    assert not storage._objects
