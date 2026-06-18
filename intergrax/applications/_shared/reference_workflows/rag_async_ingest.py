# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Reference async ingest shard planner for Tier-3 workflow workers (Proposal D / M-RAG.67).

Harness contract: ``rag.schedule_ingest_job`` triggers ``workflow_orchestrator``.
This module provides a **reference** shard plan that workers can call before
``rag.ingest_document`` per shard — not a Tier-0 ingest implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from intergrax.rag.ingest.ingest_policy import build_ingest_idempotency_key


@dataclass(frozen=True)
class IngestShardPlan:
    source_path: str
    shard_index: int
    shard_count: int
    idempotency_key: str
    workflow_parameters: dict[str, str]


def iter_directory_shards(
    directory: Path,
    *,
    files_per_shard: int = 10,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
    glob: str = "**/*",
) -> Iterator[IngestShardPlan]:
    """
    Yield shard plans for a directory corpus.

    Each shard is a batch of file paths that a workflow worker can ingest via
    ``rag.ingest_document`` (one call per file or batched by product policy).
    """
    if files_per_shard <= 0:
        raise ValueError("files_per_shard must be positive")

    files = sorted(p for p in directory.glob(glob) if p.is_file())
    if not files:
        return

    shard_count = (len(files) + files_per_shard - 1) // files_per_shard
    for shard_index in range(shard_count):
        start = shard_index * files_per_shard
        batch = files[start : start + files_per_shard]
        primary = batch[0]
        idempotency_key = build_ingest_idempotency_key(
            source_path=str(directory),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            explicit_key=f"shard-{shard_index}-of-{shard_count}",
        )
        params: dict[str, str] = {
            "job_type": "rag.ingest",
            "shard_index": str(shard_index),
            "shard_count": str(shard_count),
            "idempotency_key": idempotency_key,
        }
        if tenant_id:
            params["tenant_id"] = tenant_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        for idx, path in enumerate(batch):
            params[f"file_{idx}"] = str(path)
        yield IngestShardPlan(
            source_path=str(primary),
            shard_index=shard_index,
            shard_count=shard_count,
            idempotency_key=idempotency_key,
            workflow_parameters=params,
        )


def shard_file_paths(
    paths: Sequence[Path | str],
    *,
    files_per_shard: int = 10,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
) -> list[IngestShardPlan]:
    """Build shard plans from an explicit file list."""
    normalized = [Path(p) for p in paths]
    if not normalized:
        return []

    shard_count = (len(normalized) + files_per_shard - 1) // files_per_shard
    out: list[IngestShardPlan] = []
    for shard_index in range(shard_count):
        start = shard_index * files_per_shard
        batch = normalized[start : start + files_per_shard]
        idempotency_key = build_ingest_idempotency_key(
            source_path=str(batch[0]),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            explicit_key=f"list-shard-{shard_index}",
        )
        params = {
            "job_type": "rag.ingest",
            "shard_index": str(shard_index),
            "shard_count": str(shard_count),
            "idempotency_key": idempotency_key,
        }
        for idx, path in enumerate(batch):
            params[f"file_{idx}"] = str(path)
        out.append(
            IngestShardPlan(
                source_path=str(batch[0]),
                shard_index=shard_index,
                shard_count=shard_count,
                idempotency_key=idempotency_key,
                workflow_parameters=params,
            )
        )
    return out
