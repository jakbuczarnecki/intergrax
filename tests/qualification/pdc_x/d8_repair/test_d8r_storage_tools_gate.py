# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-H (F-D8-03): the raw ``storage.*`` tools cannot address Collaborative Work artifact content."""

from __future__ import annotations

import ast
import base64
from pathlib import Path
from typing import Any

import pytest
from intergrax.collaborative_work import content_storage
from intergrax.contracts.collaborative_work import ArtifactContentRef
from intergrax.tools.providers.storage import service
from intergrax.tools.providers.storage.contracts import (
    StorageDeleteInput,
    StorageExistsInput,
    StorageGetInput,
    StoragePutInput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
GUARDED_PREFIX = "intergrax/collaborative-work/artifact-content/"
_DIGEST = "ab" * 32


class _RecordingStorage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, key: str) -> Any:
        self.calls.append(("get", key))
        return None

    def put(self, key: str, body: bytes, **kwargs: Any) -> None:
        self.calls.append(("put", key))

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


def _calls(ctx: ToolWiringContext, key: str) -> list[Any]:
    body = base64.b64encode(b"x").decode("ascii")
    return [
        lambda: service.storage_get(ctx, StorageGetInput(key=key)),
        lambda: service.storage_put(ctx, StoragePutInput(key=key, body_base64=body)),
        lambda: service.storage_delete(ctx, StorageDeleteInput(key=key)),
        lambda: service.storage_exists(ctx, StorageExistsInput(key=key)),
    ]


def _variants() -> list[str]:
    real = _real_key()
    return [
        real,
        "intergrax/collaborative-work/artifact-content/v1/",
        "intergrax/collaborative-work/artifact-content/v2/anything",
        "/" + real,
        "intergrax//collaborative-work///artifact-content/v1/x",
        "./" + real,
        "intergrax/./collaborative-work/artifact-content/v1/x",
        real.replace("/", "\\"),
        "docs/../" + real,
        "  " + real + "  ",
    ]


@pytest.mark.parametrize("key", _variants())
def test_every_raw_storage_tool_refuses_collaborative_work_content_keys(key: str) -> None:
    storage = _RecordingStorage()
    ctx = ToolWiringContext(object_storage=storage)
    for call in _calls(ctx, key):
        with pytest.raises(service.GovernedStorageKeyError):
            call()
    assert storage.calls == []  # nothing reached the object storage


def test_ordinary_keys_are_unaffected() -> None:
    storage = _RecordingStorage()
    ctx = ToolWiringContext(object_storage=storage)
    for call in _calls(ctx, "docs/a.txt"):
        call()
    assert [name for name, _ in storage.calls] == ["get", "put", "delete", "get"]
    for key in ("intergrax/other/x", "intergrax/collaborative-work/other/x", "collaborative-work/artifact-content/x"):
        service.storage_get(ctx, StorageGetInput(key=key))


def test_the_refused_prefix_tracks_the_physical_content_key_prefix() -> None:
    assert content_storage._PHYSICAL_KEY_PREFIX.startswith(GUARDED_PREFIX)
    assert _real_key().startswith(GUARDED_PREFIX)


def test_each_raw_tool_calls_the_guard_first() -> None:
    tree = ast.parse((REPO_ROOT / "intergrax/tools/providers/storage/service.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("storage_get", "storage_put", "storage_delete", "storage_exists"):
        first = functions[name].body[0]
        assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), name
        assert getattr(first.value.func, "id", "") == "_refuse_governed_content_key", name
