# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R3 (G-03): workspace artifact tools and the filesystem store respect the governed key space."""

from __future__ import annotations

import ast
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from intergrax.collaborative_work import content_storage
from intergrax.contracts.collaborative_work import ArtifactContentRef
from intergrax.integrations._shared.p3.clients import FilesystemBlobClient
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.providers.storage import service as storage_service
from intergrax.tools.providers.storage.contracts import (
    StorageDeleteInput,
    StorageExistsInput,
    StorageGetInput,
    StoragePutInput,
)
from intergrax.tools.providers.workspace import service as workspace_service
from intergrax.tools.providers.workspace.contracts import WorkspaceExportArtifactInput, WorkspaceImportArtifactInput
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_SERVICE = REPO_ROOT / "intergrax" / "tools" / "providers" / "workspace" / "service.py"
_DIGEST = "ab" * 32


class _RecordingStorage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> Any:
        self.calls.append(("get", key))
        body = self.objects.get(key)
        return None if body is None else SimpleNamespace(body=body, content_type="text/plain")

    def put(self, key: str, body: bytes, **kwargs: Any) -> None:
        self.calls.append(("put", key))
        self.objects[key] = body

    def delete(self, key: str) -> None:
        self.calls.append(("delete", key))

    def presigned_url(self, key: str, **kwargs: Any) -> str:
        raise AssertionError("never called")

    def close(self) -> None:
        return None


def _real_key() -> str:
    ref = ArtifactContentRef(
        content_ref=f"artifact-content://sha256/{_DIGEST}",
        integrity_digest=f"sha256:{_DIGEST}",
        media_type="text/plain",
        size_bytes=1,
    )
    return content_storage.artifact_content_object_key("tenant-victim", "ws-1", ref)


def _governed_variants() -> list[str]:
    real = _real_key()
    return [
        real,
        "intergrax/collaborative-work/artifact-content/v1/",
        "/" + real,
        "intergrax//collaborative-work///artifact-content/v1/x",
        real.replace("/", "\\"),
        "docs/../" + real,
    ]


def _alias_variants() -> list[str]:
    """Spellings that address the same physical keys on case-insensitive or dot-stripping stores."""
    return [
        "Intergrax/Collaborative-Work/artifact-content/v1/abc/sha256/dead",
        "INTERGRAX/COLLABORATIVE-WORK/ARTIFACT-CONTENT/v1/x",
        "intergrax/collaborative-work./artifact-content/v1/abc/sha256/dead",
        "intergrax/collaborative-work/artifact-content./v1/abc/sha256/dead",
        "intergrax/collaborative-work /artifact-content/v1/abc/sha256/dead",
        "intergrax./collaborative-work/artifact-content/v1/x",
    ]


def _workspace_ctx(tmp_path: Path, storage: _RecordingStorage) -> ToolWiringContext:
    workspace = ShadowWorkspace.create(tmp_path, tenant_id="tenant-1", task_id="task-1")
    workspace.write_text("out/report.txt", "payload", content_type="text/plain")
    return ToolWiringContext(shadow_workspace=workspace, object_storage=storage)


@pytest.mark.parametrize("key", _governed_variants() + _alias_variants())
def test_workspace_export_and_import_refuse_governed_keys(tmp_path: Path, key: str) -> None:
    storage = _RecordingStorage()
    ctx = _workspace_ctx(tmp_path, storage)

    with pytest.raises(storage_service.GovernedStorageKeyError):
        workspace_service.workspace_export_artifact(
            ctx, WorkspaceExportArtifactInput(path="out/report.txt", storage_key=key)
        )
    with pytest.raises(storage_service.GovernedStorageKeyError):
        workspace_service.workspace_import_artifact(
            ctx, WorkspaceImportArtifactInput(path="in/victim.txt", storage_key=key)
        )

    assert storage.calls == []  # nothing reached the object storage
    assert [a.relative_path for a in ctx.shadow_workspace.list_artifacts()] == ["out/report.txt"]  # type: ignore[union-attr]


def test_workspace_tools_still_round_trip_ordinary_keys(tmp_path: Path) -> None:
    storage = _RecordingStorage()
    ctx = _workspace_ctx(tmp_path, storage)

    exported = workspace_service.workspace_export_artifact(
        ctx, WorkspaceExportArtifactInput(path="out/report.txt", storage_key="exports/report.txt")
    )
    imported = workspace_service.workspace_import_artifact(
        ctx, WorkspaceImportArtifactInput(path="in/back.txt", storage_key="exports/report.txt")
    )

    assert exported.exported is True and imported.imported is True
    assert [name for name, _ in storage.calls] == ["put", "get"]


@pytest.mark.parametrize("key", _alias_variants())
def test_the_shared_refusal_and_the_raw_tools_cover_case_and_dot_aliases(key: str) -> None:
    with pytest.raises(storage_service.GovernedStorageKeyError):
        storage_service.refuse_governed_content_key(key)  # type: ignore[attr-defined]
    storage = _RecordingStorage()
    ctx = ToolWiringContext(object_storage=storage)
    body = base64.b64encode(b"x").decode("ascii")
    for call in (
        lambda: storage_service.storage_get(ctx, StorageGetInput(key=key)),
        lambda: storage_service.storage_put(ctx, StoragePutInput(key=key, body_base64=body)),
        lambda: storage_service.storage_delete(ctx, StorageDeleteInput(key=key)),
        lambda: storage_service.storage_exists(ctx, StorageExistsInput(key=key)),
    ):
        with pytest.raises(storage_service.GovernedStorageKeyError):
            call()
    assert storage.calls == []


@pytest.mark.parametrize(
    "key",
    [
        "docs/a.txt",
        "intergrax/other/x",
        "intergrax/collaborative-work/other/x",
        "collaborative-work/artifact-content/x",
        "intergrax/collaborative-workspace/artifact-content/x",
    ],
)
def test_the_shared_refusal_keeps_accepting_unrelated_keys(key: str) -> None:
    storage_service.refuse_governed_content_key(key)  # type: ignore[attr-defined]


def _unordered_guard_calls(source: str) -> list[str]:
    """Functions that read ``storage_key`` but touch workspace or storage before the shared refusal."""
    offenders: list[str] = []
    for fn in (n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)):
        if not any(isinstance(n, ast.Attribute) and n.attr == "storage_key" for n in ast.walk(fn)):
            continue
        guard_at = next(
            (
                index
                for index, stmt in enumerate(fn.body)
                if any(
                    isinstance(n, ast.Call) and getattr(n.func, "id", "") == "refuse_governed_content_key"
                    for n in ast.walk(stmt)
                )
            ),
            None,
        )
        if guard_at is None:
            offenders.append(f"{fn.name}: no refusal")
            continue
        for stmt in fn.body[:guard_at]:
            if any(
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id in {"workspace", "object_storage"}
                for n in ast.walk(stmt)
            ):
                offenders.append(f"{fn.name}: workspace or storage access before the refusal")
    return offenders


def test_workspace_tools_refuse_before_any_workspace_or_storage_access() -> None:
    assert _unordered_guard_calls(WORKSPACE_SERVICE.read_text(encoding="utf-8")) == []


def test_the_order_gate_detects_a_missing_or_late_refusal() -> None:
    missing = (
        "def tool(ctx, params):\n    storage = ctx.object_storage\n    return object_storage.get(params.storage_key)\n"
    )
    late = (
        "def tool(ctx, params):\n"
        "    body = workspace.read_artifact_bytes(params.path)\n"
        "    refuse_governed_content_key(params.storage_key)\n"
    )
    ok = (
        "def tool(ctx, params):\n    refuse_governed_content_key(params.storage_key)\n"
        "    return object_storage.get(1)\n"
    )
    assert _unordered_guard_calls(missing) == ["tool: no refusal"]
    assert _unordered_guard_calls(late) == ["tool: workspace or storage access before the refusal"]
    assert _unordered_guard_calls(ok) == []


def _escaping_keys(tmp_path: Path) -> list[str]:
    outside = tmp_path / "outside" / "escaped.txt"
    return [str(outside), "../escaped.txt", "a/../../escaped.txt", "C:/escaped.txt", "C:\\escaped.txt"]


@pytest.mark.parametrize("index", range(5))
def test_filesystem_blob_client_refuses_keys_that_escape_its_root(tmp_path: Path, index: int) -> None:
    key = _escaping_keys(tmp_path)[index]
    root = tmp_path / "store"
    client = FilesystemBlobClient(root)

    for call in (
        lambda: client.put_object(Key=key, Body=b"x", ContentType="text/plain"),
        lambda: client.get_object(Key=key),
        lambda: client.delete_object(Key=key),
    ):
        with pytest.raises(ValueError):
            call()

    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "escaped.txt").exists()
    assert [p for p in root.rglob("*") if p.is_file()] == []


def test_filesystem_blob_client_still_stores_and_reads_ordinary_keys(tmp_path: Path) -> None:
    client = FilesystemBlobClient(tmp_path / "store")
    client.put_object(Key="docs/a.txt", Body=b"hello", ContentType="text/plain", Metadata={"k": "v"})

    stored = client.get_object(Key="docs/a.txt")

    assert stored["Body"].read() == b"hello"
    assert stored["ContentType"] == "text/plain"
    client.delete_object(Key="docs/a.txt")
    with pytest.raises(FileNotFoundError):
        client.get_object(Key="docs/a.txt")
