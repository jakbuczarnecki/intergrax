# © Artur Czarnecki. All rights reserved.

"""Workspace index for codebase context preset (CE-7.1)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable


@dataclass(frozen=True, slots=True)
class WorkspaceChunk:
    path: str
    start_line: int
    end_line: int
    content_hash: str
    content: str = ""


@dataclass(frozen=True, slots=True)
class WorkspaceIndexResult:
    root_merkle: str
    chunks: tuple[WorkspaceChunk, ...]


ChunkHook = Callable[[str, str], list[tuple[int, int, str]]]


def _default_chunk_hook(path: str, content: str) -> list[tuple[int, int, str]]:
    lines = content.splitlines() or [""]
    return [(1, len(lines), content)]


def build_workspace_index(
    files: dict[str, str],
    *,
    chunk_hook: ChunkHook | None = None,
) -> WorkspaceIndexResult:
    """Deterministic Merkle root over workspace chunks (promoted from MEM spike)."""
    hook = chunk_hook or _default_chunk_hook
    chunks: list[WorkspaceChunk] = []
    leaf_hashes: list[str] = []
    for path in sorted(files):
        content = files[path]
        for start, end, chunk_text in hook(path, content):
            content_hash = sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append(
                WorkspaceChunk(
                    path=path,
                    start_line=start,
                    end_line=end,
                    content_hash=content_hash,
                    content=chunk_text,
                )
            )
            leaf_hashes.append(sha256(f"{path}:{content_hash}".encode()).hexdigest())
    if not leaf_hashes:
        root = sha256(b"").hexdigest()
    else:
        root = sha256("".join(leaf_hashes).encode()).hexdigest()
    return WorkspaceIndexResult(root_merkle=root, chunks=tuple(chunks))
