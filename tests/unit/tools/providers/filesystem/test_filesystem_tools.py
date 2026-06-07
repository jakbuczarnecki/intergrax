# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.tools.providers.filesystem.contracts import (
    FilesystemGlobInput,
    FilesystemListInput,
    FilesystemReadTextInput,
    FilesystemStatInput,
    FilesystemWriteTextInput,
)
from intergrax.tools.providers.filesystem.service import (
    filesystem_glob,
    filesystem_list,
    filesystem_read_text,
    filesystem_stat,
    filesystem_write_text,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def allowlisted_ctx(tmp_path: Path) -> ToolWiringContext:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("hello world", encoding="utf-8")
    return ToolWiringContext(read_allowlist_roots=frozenset({str(tmp_path.resolve())}))


def test_filesystem_list_returns_entries(allowlisted_ctx: ToolWiringContext, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    out = filesystem_list(allowlisted_ctx, FilesystemListInput(path=str(docs.resolve())))
    assert out.total == 1
    assert out.entries[0].name == "note.txt"


def test_filesystem_glob_finds_txt(allowlisted_ctx: ToolWiringContext, tmp_path: Path) -> None:
    out = filesystem_glob(
        allowlisted_ctx,
        FilesystemGlobInput(root=str(tmp_path.resolve()), pattern="**/*.txt"),
    )
    assert out.total == 1
    assert out.paths[0].endswith("note.txt")


def test_filesystem_read_text_reads_file(allowlisted_ctx: ToolWiringContext, tmp_path: Path) -> None:
    file_path = tmp_path / "docs" / "note.txt"
    out = filesystem_read_text(allowlisted_ctx, FilesystemReadTextInput(path=str(file_path.resolve())))
    assert out.text == "hello world"


def test_filesystem_stat_reports_metadata(allowlisted_ctx: ToolWiringContext, tmp_path: Path) -> None:
    file_path = tmp_path / "docs" / "note.txt"
    out = filesystem_stat(allowlisted_ctx, FilesystemStatInput(path=str(file_path.resolve())))
    assert out.exists is True
    assert out.is_file is True


def test_filesystem_rejects_path_outside_allowlist(tmp_path: Path) -> None:
    ctx = ToolWiringContext(read_allowlist_roots=frozenset({str(tmp_path.resolve())}))
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(RuntimeError, match="path_not_in_allowlist"):
        filesystem_read_text(ctx, FilesystemReadTextInput(path=str(outside.resolve())))


def test_filesystem_write_text_writes_allowlisted_file(allowlisted_ctx: ToolWiringContext, tmp_path: Path) -> None:
    target = tmp_path / "docs" / "output.txt"
    out = filesystem_write_text(
        allowlisted_ctx,
        FilesystemWriteTextInput(path=str(target.resolve()), content="saved"),
    )
    assert out.written is True
    assert target.read_text(encoding="utf-8") == "saved"
