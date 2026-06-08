# © Artur Czarnecki. All rights reserved.

"""Workspace incremental index spike — Merkle + AST chunking RFC (Phase MEM-DEPTH-5.5)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import List


@dataclass(frozen=True, slots=True)
class WorkspaceChunk:
    path: str
    start_line: int
    end_line: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class WorkspaceIndexSpikeResult:
    root_merkle: str
    chunks: List[WorkspaceChunk]


def build_workspace_index_spike(files: dict[str, str]) -> WorkspaceIndexSpikeResult:
    """
    RFC spike — deterministic Merkle root over path/content hashes.

    Not wired to Tier-3 hosts by default; see plan MEM-DEPTH-5.5.
    """
    chunks: List[WorkspaceChunk] = []
    leaf_hashes: List[str] = []
    for path in sorted(files):
        content = files[path]
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        line_count = max(1, content.count("\n") + 1)
        chunks.append(
            WorkspaceChunk(
                path=path,
                start_line=1,
                end_line=line_count,
                content_hash=content_hash,
            )
        )
        leaf_hashes.append(sha256(f"{path}:{content_hash}".encode()).hexdigest())

    if not leaf_hashes:
        root = sha256(b"").hexdigest()
    else:
        combined = "".join(leaf_hashes)
        root = sha256(combined.encode()).hexdigest()

    return WorkspaceIndexSpikeResult(root_merkle=root, chunks=chunks)
