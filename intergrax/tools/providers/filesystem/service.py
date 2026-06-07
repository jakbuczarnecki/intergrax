# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from intergrax.tools.providers.filesystem.allowlist import require_read_allowlist_roots, resolve_allowed_path
from intergrax.tools.providers.filesystem.contracts import (
    FilesystemEntryOutput,
    FilesystemGlobInput,
    FilesystemGlobOutput,
    FilesystemListInput,
    FilesystemListOutput,
    FilesystemReadTextInput,
    FilesystemReadTextOutput,
    FilesystemStatInput,
    FilesystemStatOutput,
    FilesystemWriteTextInput,
    FilesystemWriteTextOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

FILESYSTEM_LIST_TOOL_ID = "filesystem.list"
FILESYSTEM_GLOB_TOOL_ID = "filesystem.glob"
FILESYSTEM_READ_TEXT_TOOL_ID = "filesystem.read_text"
FILESYSTEM_STAT_TOOL_ID = "filesystem.stat"
FILESYSTEM_WRITE_TEXT_TOOL_ID = "filesystem.write_text"


def _roots(ctx: ToolWiringContext) -> frozenset[str]:
    return require_read_allowlist_roots(ctx.read_allowlist_roots)


def filesystem_list(ctx: ToolWiringContext, params: FilesystemListInput) -> FilesystemListOutput:
    directory = resolve_allowed_path(params.path, _roots(ctx))
    if not directory.is_dir():
        raise RuntimeError("path_not_a_directory")
    entries: list[FilesystemEntryOutput] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if len(entries) >= params.limit:
            break
        entries.append(
            FilesystemEntryOutput(
                name=child.name,
                path=str(child.resolve()),
                is_dir=child.is_dir(),
            )
        )
    return FilesystemListOutput(path=str(directory), entries=entries, total=len(entries))


def filesystem_glob(ctx: ToolWiringContext, params: FilesystemGlobInput) -> FilesystemGlobOutput:
    root = resolve_allowed_path(params.root, _roots(ctx))
    if not root.is_dir():
        raise RuntimeError("root_not_a_directory")
    matches: list[str] = []
    for match in sorted(root.glob(params.pattern)):
        resolved = match.resolve()
        try:
            resolve_allowed_path(str(resolved), _roots(ctx))
        except RuntimeError:
            continue
        matches.append(str(resolved))
        if len(matches) >= params.limit:
            break
    return FilesystemGlobOutput(
        root=str(root),
        pattern=params.pattern.strip(),
        paths=matches,
        total=len(matches),
    )


def filesystem_read_text(ctx: ToolWiringContext, params: FilesystemReadTextInput) -> FilesystemReadTextOutput:
    file_path = resolve_allowed_path(params.path, _roots(ctx))
    if not file_path.is_file():
        raise RuntimeError("path_not_a_file")
    size_bytes = file_path.stat().st_size
    truncated = size_bytes > params.max_bytes
    read_size = params.max_bytes if truncated else size_bytes
    data = file_path.read_bytes()[:read_size]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("file_not_utf8_text") from exc
    return FilesystemReadTextOutput(
        path=str(file_path),
        text=text,
        truncated=truncated,
        size_bytes=size_bytes,
    )


def filesystem_stat(ctx: ToolWiringContext, params: FilesystemStatInput) -> FilesystemStatOutput:
    target = resolve_allowed_path(params.path, _roots(ctx))
    if not target.exists():
        return FilesystemStatOutput(path=str(target), exists=False)
    stat = target.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return FilesystemStatOutput(
        path=str(target),
        exists=True,
        is_dir=target.is_dir(),
        is_file=target.is_file(),
        size_bytes=int(stat.st_size),
        modified_at_utc=modified,
    )


def filesystem_write_text(ctx: ToolWiringContext, params: FilesystemWriteTextInput) -> FilesystemWriteTextOutput:
    file_path = resolve_allowed_path(params.path, _roots(ctx))
    encoded = params.content.encode("utf-8")
    if len(encoded) > params.max_bytes:
        raise RuntimeError("content_exceeds_max_bytes")
    created = not file_path.exists()
    if params.create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(encoded)
    return FilesystemWriteTextOutput(
        path=str(file_path),
        written=True,
        size_bytes=len(encoded),
        created=created,
    )
