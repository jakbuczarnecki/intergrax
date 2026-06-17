# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.reference_workflows.rag_async_ingest import (
    iter_directory_shards,
    shard_file_paths,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_iter_directory_shards(tmp_path: Path) -> None:
    for idx in range(5):
        (tmp_path / f"doc_{idx}.txt").write_text(f"body-{idx}", encoding="utf-8")

    plans = list(iter_directory_shards(tmp_path, files_per_shard=2, tenant_id="t1"))
    assert len(plans) == 3
    assert plans[0].shard_index == 0
    assert plans[0].workflow_parameters["job_type"] == "rag.ingest"
    assert "file_0" in plans[0].workflow_parameters


def test_shard_file_paths_explicit_list(tmp_path: Path) -> None:
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for path in paths:
        path.write_text("x", encoding="utf-8")
    plans = shard_file_paths(paths, files_per_shard=1, workspace_id="ws1")
    assert len(plans) == 2
    assert plans[1].shard_index == 1
