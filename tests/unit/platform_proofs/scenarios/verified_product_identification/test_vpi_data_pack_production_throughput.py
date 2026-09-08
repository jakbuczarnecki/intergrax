"""Unit tests for VPI data pack build progress and execution profiles."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_interruption import (
    DataPackBuildInterrupted,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_progress import (
    compute_build_progress,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    ShardBuildSeams,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_execution_profiles import (
    PRODUCTION_LOCAL_GPU_PROFILE_ID,
    apply_data_pack_build_execution_profile,
    resolve_data_pack_build_execution_profile,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardBuildState,
    DataPackShardStatus,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DEFAULT_PRODUCTION_SHARD_SIZE,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    patch_canonical_model_identity,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit


def _build_config(
    tmp_path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    output_root: Path,
    shard_size: int = DEFAULT_PRODUCTION_SHARD_SIZE,
    max_records: int = 3_000,
    resume: bool = False,
    start_fresh: bool = False,
    stop_after_shard: int | None = None,
) -> DataPackBuildConfig:
    return DataPackBuildConfig(
        output_root=output_root,
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        shard_size=shard_size,
        max_records=max_records,
        resume=resume,
        start_fresh=start_fresh,
        stop_after_shard=stop_after_shard,
    )


def test_production_local_gpu_profile_applies_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPI_EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("VPI_EMBEDDING_PROVIDER_BATCH_SIZE", raising=False)
    profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
    apply_data_pack_build_execution_profile(profile)
    assert os.environ["VPI_EMBEDDING_DEVICE"] == "cuda"
    assert os.environ["VPI_EMBEDDING_PROVIDER_BATCH_SIZE"] == "16"
    assert profile.model == "BAAI/bge-m3"


def test_unknown_execution_profile_fails() -> None:
    with pytest.raises(VpiDataPackBuildError, match="unknown data pack execution profile"):
        resolve_data_pack_build_execution_profile("unknown-profile")


def test_build_progress_reports_remaining_and_estimate() -> None:
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
        DataPackBuildState,
    )

    state = DataPackBuildState(
        state_version="vpi.data_pack.build_state/1",
        build_id="build-1",
        content_identity="content",
        expected_record_count=3_000,
        shard_size=1_000,
        shard_count=3,
        catalog_id="catalog",
        started_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:01+00:00",
        completed_shards=1,
        shards=(
            DataPackShardBuildState(
                ordinal=1,
                start_row_index=0,
                end_row_index_exclusive=1_000,
                expected_record_count=1_000,
                status=DataPackShardStatus.READY,
                relational_relative_path="relational/part-000001.parquet",
                embedding_relative_path="embeddings/part-000001.parquet",
                attempt=1,
                relational_sha256="a" * 64,
                embedding_sha256="b" * 64,
                relational_source_ref_set_sha256="c" * 64,
                embedding_source_ref_set_sha256="c" * 64,
                elapsed_seconds=120.0,
                records_processed=1_000,
                embedding_count=1_000,
            ),
            DataPackShardBuildState(
                ordinal=2,
                start_row_index=1_000,
                end_row_index_exclusive=2_000,
                expected_record_count=1_000,
                status=DataPackShardStatus.PENDING,
                relational_relative_path=None,
                embedding_relative_path=None,
                attempt=0,
            ),
            DataPackShardBuildState(
                ordinal=3,
                start_row_index=2_000,
                end_row_index_exclusive=3_000,
                expected_record_count=1_000,
                status=DataPackShardStatus.PENDING,
                relational_relative_path=None,
                embedding_relative_path=None,
                attempt=0,
            ),
        ),
    )
    progress = compute_build_progress(state)
    assert progress.ready_shards == 1
    assert progress.remaining_shards == 2
    assert progress.records_completed == 1_000
    assert progress.average_shard_elapsed_seconds == 120.0
    assert progress.estimated_remaining_seconds == 240.0


def test_three_shard_interrupt_resume_skips_ready_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(
        tmp_path / "dataset",
        row_count=3_000,
    )
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)
    shard_size = DEFAULT_PRODUCTION_SHARD_SIZE

    first_embedding = FakeDataPackEmbeddingPort()
    run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=shard_size,
            max_records=3_000,
            start_fresh=True,
            stop_after_shard=1,
        ),
        embedding_port=first_embedding,
    )
    assert first_embedding.embed_calls > 0

    payload = json.loads(paths.build_state_file.read_text(encoding="utf-8"))
    payload["shards"][1]["status"] = "EMBEDDING"
    payload["shards"][1]["attempt"] = 1
    paths.build_state_file.write_text(json.dumps(payload), encoding="utf-8")

    resume_embedding = FakeDataPackEmbeddingPort()
    report = run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=shard_size,
            max_records=3_000,
            resume=True,
        ),
        embedding_port=resume_embedding,
    )
    state = read_build_state_file(paths.build_state_file)
    assert report.finalized is True
    assert report.status is DataPackStatus.READY
    assert state.completed_shards == 3
    assert all(shard.status is DataPackShardStatus.READY for shard in state.shards)
    ready_shard = state.shards[0]
    assert ready_shard.elapsed_seconds is not None
    assert ready_shard.records_processed == shard_size
    assert ready_shard.embedding_count == shard_size
    assert resume_embedding.embed_calls > 0
    assert len(resume_embedding.texts_seen) == shard_size * 2


def test_sigterm_interrupt_persists_non_ready_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=50)
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)

    def raise_sigterm() -> None:
        signal.raise_signal(signal.SIGTERM)

    with pytest.raises(DataPackBuildInterrupted):
        run_resumable_data_pack_build(
            _build_config(
                tmp_path,
                dataset_path=dataset_path,
                manifest_path=manifest_path,
                output_root=output_root,
                shard_size=25,
                max_records=50,
                start_fresh=True,
            ),
            embedding_port=FakeDataPackEmbeddingPort(),
            build_seams=ShardBuildSeams(after_both_temp_writes=raise_sigterm),
        )

    state = read_build_state_file(paths.build_state_file)
    assert state.completed_shards == 0
    assert state.shards[0].status is not DataPackShardStatus.READY
