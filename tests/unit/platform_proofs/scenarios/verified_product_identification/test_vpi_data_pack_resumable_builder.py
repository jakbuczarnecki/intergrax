"""Resume, temp-file, and integration tests for resumable VPI data pack builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_state_machine import (
    discard_shard_temp_outputs,
    recover_non_ready_shard,
    validate_ready_shard_artifacts,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    ShardBuildSeams,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardBuildState,
    DataPackShardStatus,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
    VpiDataPackBuildIdentityMismatchError,
    VpiDataPackReadyShardCorruptionError,
    VpiDataPackResumeError,
    VpiDataPackValidationError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
    resolve_data_pack_paths,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    patch_canonical_model_identity,
    write_selected_dataset_with_manifest_count,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit


def _build_config(
    tmp_path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    output_root: Path,
    shard_size: int = 25,
    max_records: int | None = 120,
    resume: bool = False,
    start_fresh: bool = False,
    max_shards: int | None = None,
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
        max_shards=max_shards,
        stop_after_shard=stop_after_shard,
    )


def test_resume_rebuilds_non_ready_shard_and_removes_temp(tmp_path: Path) -> None:
    paths = resolve_data_pack_paths(tmp_path / "pack")
    shard = DataPackShardBuildState(
        ordinal=1,
        start_row_index=0,
        end_row_index_exclusive=25,
        expected_record_count=25,
        status=DataPackShardStatus.WRITING,
        relational_relative_path=None,
        embedding_relative_path=None,
        attempt=1,
    )
    paths.relational_dir.mkdir(parents=True)
    paths.embeddings_dir.mkdir(parents=True)
    temp_rel = temp_shard_path(paths.relational_dir, 1)
    temp_rel.write_text("partial", encoding="utf-8")
    final_rel = final_shard_path(paths.relational_dir, 1)
    final_rel.write_text("orphan", encoding="utf-8")

    recovered = recover_non_ready_shard(
        shard,
        relational_dir=paths.relational_dir,
        embeddings_dir=paths.embeddings_dir,
    )
    assert recovered.status is DataPackShardStatus.PENDING
    assert recovered.attempt == 2
    assert not temp_rel.exists()
    assert not final_rel.exists()


def test_pending_shard_with_orphan_final_fails(tmp_path: Path) -> None:
    paths = resolve_data_pack_paths(tmp_path / "pack")
    shard = DataPackShardBuildState(
        ordinal=1,
        start_row_index=0,
        end_row_index_exclusive=25,
        expected_record_count=25,
        status=DataPackShardStatus.PENDING,
        relational_relative_path=None,
        embedding_relative_path=None,
        attempt=0,
    )
    paths.relational_dir.mkdir(parents=True)
    final_shard_path(paths.relational_dir, 1).write_text("orphan", encoding="utf-8")
    with pytest.raises(VpiDataPackResumeError, match="orphan final shard"):
        recover_non_ready_shard(
            shard,
            relational_dir=paths.relational_dir,
            embeddings_dir=paths.embeddings_dir,
        )


def test_ready_shard_corruption_missing_file(tmp_path: Path) -> None:
    paths = resolve_data_pack_paths(tmp_path / "pack")
    shard = DataPackShardBuildState(
        ordinal=1,
        start_row_index=0,
        end_row_index_exclusive=25,
        expected_record_count=25,
        status=DataPackShardStatus.READY,
        relational_relative_path="relational/part-000001.parquet",
        embedding_relative_path="embeddings/part-000001.parquet",
        attempt=1,
        relational_sha256="a" * 64,
        embedding_sha256="b" * 64,
        relational_source_ref_set_sha256="c" * 64,
        embedding_source_ref_set_sha256="c" * 64,
    )
    with pytest.raises(VpiDataPackReadyShardCorruptionError, match="files missing"):
        validate_ready_shard_artifacts(
            pack_root=paths.root,
            shard=shard,
            relational_schema_version=RELATIONAL_SCHEMA_VERSION,
            embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
            expected_dimension=1024,
        )


def test_failed_validation_leaves_no_ready_final_artifact(tmp_path: Path) -> None:
    paths = resolve_data_pack_paths(tmp_path / "pack")
    paths.relational_dir.mkdir(parents=True)
    temp_path = temp_shard_path(paths.relational_dir, 1)
    temp_path.write_bytes(b"not-parquet")
    final_path = final_shard_path(paths.relational_dir, 1)
    assert not final_path.exists()
    discard_shard_temp_outputs(
        relational_dir=paths.relational_dir,
        embeddings_dir=paths.embeddings_dir,
        shard_ordinal=1,
    )
    assert not temp_path.exists()
    assert not final_path.exists()


def test_small_multi_shard_build_and_finalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=120)
    output_root = tmp_path / "pack"
    fake_embedding = FakeDataPackEmbeddingPort()
    config = _build_config(
        tmp_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_root=output_root,
        shard_size=25,
        max_records=120,
        start_fresh=True,
    )
    report = run_resumable_data_pack_build(config, embedding_port=fake_embedding)
    assert report.finalized is True
    assert report.status is DataPackStatus.READY
    assert report.manifest is not None
    paths = resolve_data_pack_paths(output_root)
    assert paths.manifest_file.is_file()
    assert paths.shards_index_file.is_file()
    assert paths.checksums_file.is_file()
    state = read_build_state_file(paths.build_state_file)
    assert state.completed_shards == 5


def test_resume_skips_ready_shards_without_reembedding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=120)
    output_root = tmp_path / "pack"
    fake_embedding = FakeDataPackEmbeddingPort()
    first_config = _build_config(
        tmp_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_root=output_root,
        shard_size=25,
        max_records=120,
        start_fresh=True,
        stop_after_shard=2,
    )
    first_report = run_resumable_data_pack_build(first_config, embedding_port=fake_embedding)
    assert first_report.finalized is False
    first_calls = fake_embedding.embed_calls

    second_embedding = FakeDataPackEmbeddingPort()
    second_config = _build_config(
        tmp_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_root=output_root,
        shard_size=25,
        max_records=120,
        resume=True,
    )
    second_report = run_resumable_data_pack_build(second_config, embedding_port=second_embedding)
    assert second_report.finalized is True
    assert second_embedding.embed_calls > 0
    assert first_calls > 0


def test_content_identity_mismatch_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=25)
    output_root = tmp_path / "pack"
    config = _build_config(
        tmp_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_root=output_root,
        shard_size=25,
        max_records=25,
        start_fresh=True,
        stop_after_shard=1,
    )
    run_resumable_data_pack_build(config, embedding_port=FakeDataPackEmbeddingPort())

    payload = json.loads((output_root / "state" / "build-state.json").read_text(encoding="utf-8"))
    payload["content_identity"] = "deadbeef"
    (output_root / "state" / "build-state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VpiDataPackBuildIdentityMismatchError):
        run_resumable_data_pack_build(
            _build_config(
                tmp_path,
                dataset_path=dataset_path,
                manifest_path=manifest_path,
                output_root=output_root,
                shard_size=25,
                max_records=25,
                resume=True,
            ),
            embedding_port=FakeDataPackEmbeddingPort(),
        )


def test_interrupted_writing_shard_rebuilt_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=50)
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)
    fake_embedding = FakeDataPackEmbeddingPort()
    partial_config = _build_config(
        tmp_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_root=output_root,
        shard_size=25,
        max_records=50,
        start_fresh=True,
        stop_after_shard=1,
    )
    run_resumable_data_pack_build(partial_config, embedding_port=fake_embedding)

    payload = json.loads(paths.build_state_file.read_text(encoding="utf-8"))
    payload["shards"][1]["status"] = "WRITING"
    payload["shards"][1]["attempt"] = 1
    paths.build_state_file.write_text(json.dumps(payload), encoding="utf-8")
    paths.relational_dir.mkdir(parents=True, exist_ok=True)
    paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    temp_shard_path(paths.relational_dir, 2).write_text("partial", encoding="utf-8")

    resume_embedding = FakeDataPackEmbeddingPort()
    resume_config = _build_config(
        tmp_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_root=output_root,
        shard_size=25,
        max_records=50,
        resume=True,
    )
    report = run_resumable_data_pack_build(resume_config, embedding_port=resume_embedding)
    assert report.finalized is True
    assert not temp_shard_path(paths.relational_dir, 2).exists()


def test_temp_validation_failure_leaves_no_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=25)
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)

    def corrupt_relational_temp() -> None:
        temp_shard_path(paths.relational_dir, 1).write_bytes(b"not-parquet")

    with pytest.raises(VpiDataPackValidationError):
        run_resumable_data_pack_build(
            _build_config(
                tmp_path,
                dataset_path=dataset_path,
                manifest_path=manifest_path,
                output_root=output_root,
                shard_size=25,
                max_records=25,
                start_fresh=True,
            ),
            embedding_port=FakeDataPackEmbeddingPort(),
            build_seams=ShardBuildSeams(after_both_temp_writes=corrupt_relational_temp),
        )
    assert not final_shard_path(paths.relational_dir, 1).exists()
    assert not final_shard_path(paths.embeddings_dir, 1).exists()


def test_crash_between_renames_recovered_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=25)
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)

    def fail_embedding_commit() -> None:
        raise VpiDataPackBuildError("simulated embedding rename failure")

    with pytest.raises(VpiDataPackBuildError, match="simulated embedding rename failure"):
        run_resumable_data_pack_build(
            _build_config(
                tmp_path,
                dataset_path=dataset_path,
                manifest_path=manifest_path,
                output_root=output_root,
                shard_size=25,
                max_records=25,
                start_fresh=True,
            ),
            embedding_port=FakeDataPackEmbeddingPort(),
            build_seams=ShardBuildSeams(before_embedding_commit=fail_embedding_commit),
        )
    assert final_shard_path(paths.relational_dir, 1).exists()
    assert not final_shard_path(paths.embeddings_dir, 1).exists()

    report = run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=25,
            max_records=25,
            resume=True,
        ),
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert report.finalized is True
    assert final_shard_path(paths.relational_dir, 1).exists()
    assert final_shard_path(paths.embeddings_dir, 1).exists()


def test_validating_with_both_finals_rebuilt_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=50)
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)
    run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=25,
            max_records=50,
            start_fresh=True,
            stop_after_shard=1,
        ),
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    final_rel = final_shard_path(paths.relational_dir, 2)
    final_emb = final_shard_path(paths.embeddings_dir, 2)
    final_rel.write_bytes(b"orphan-relational")
    final_emb.write_bytes(b"orphan-embedding")
    payload = json.loads(paths.build_state_file.read_text(encoding="utf-8"))
    payload["shards"][1]["status"] = "VALIDATING"
    payload["shards"][1]["relational_relative_path"] = "relational/part-000002.parquet"
    payload["shards"][1]["embedding_relative_path"] = "embeddings/part-000002.parquet"
    paths.build_state_file.write_text(json.dumps(payload), encoding="utf-8")

    report = run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=25,
            max_records=50,
            resume=True,
        ),
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert report.finalized is True
    state = read_build_state_file(paths.build_state_file)
    assert state.shards[1].status is DataPackShardStatus.READY
    assert not temp_shard_path(paths.relational_dir, 2).exists()
    assert not temp_shard_path(paths.embeddings_dir, 2).exists()


def test_validating_with_only_relational_final_rebuilt_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    paths = resolve_data_pack_paths(tmp_path / "pack")
    shard = DataPackShardBuildState(
        ordinal=2,
        start_row_index=25,
        end_row_index_exclusive=50,
        expected_record_count=25,
        status=DataPackShardStatus.VALIDATING,
        relational_relative_path="relational/part-000002.parquet",
        embedding_relative_path="embeddings/part-000002.parquet",
        attempt=1,
    )
    paths.relational_dir.mkdir(parents=True)
    paths.embeddings_dir.mkdir(parents=True)
    final_shard_path(paths.relational_dir, 2).write_text("orphan-rel", encoding="utf-8")
    recovered = recover_non_ready_shard(
        shard,
        relational_dir=paths.relational_dir,
        embeddings_dir=paths.embeddings_dir,
    )
    assert recovered.status is DataPackShardStatus.PENDING
    assert not final_shard_path(paths.relational_dir, 2).exists()


def test_full_plan_resume_skips_ready_shard_and_builds_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_selected_dataset_with_manifest_count(
        tmp_path / "dataset",
        parquet_row_count=10_000,
        manifest_record_count=3_770_377,
    )
    output_root = tmp_path / "pack"
    paths = resolve_data_pack_paths(output_root)
    shard_size = 5_000

    first_embedding = FakeDataPackEmbeddingPort()
    run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=shard_size,
            max_records=None,
            start_fresh=True,
            stop_after_shard=1,
        ),
        embedding_port=first_embedding,
    )
    state_after_run1 = read_build_state_file(paths.build_state_file)
    assert state_after_run1.expected_record_count == 3_770_377
    assert state_after_run1.shard_count == 755
    assert state_after_run1.completed_shards == 1
    assert state_after_run1.shards[0].status is DataPackShardStatus.READY
    assert state_after_run1.shards[1].status is DataPackShardStatus.PENDING
    assert not paths.manifest_file.exists()

    resume_embedding = FakeDataPackEmbeddingPort()
    report = run_resumable_data_pack_build(
        _build_config(
            tmp_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            output_root=output_root,
            shard_size=shard_size,
            max_records=None,
            resume=True,
            stop_after_shard=2,
        ),
        embedding_port=resume_embedding,
    )
    state_after_run2 = read_build_state_file(paths.build_state_file)
    assert report.finalized is False
    assert state_after_run2.expected_record_count == 3_770_377
    assert state_after_run2.shard_count == 755
    assert state_after_run2.completed_shards == 2
    assert state_after_run2.shards[0].status is DataPackShardStatus.READY
    assert state_after_run2.shards[1].status is DataPackShardStatus.READY
    assert state_after_run2.shards[2].status is DataPackShardStatus.PENDING
    assert resume_embedding.embed_calls > 0
    assert len(resume_embedding.texts_seen) == shard_size
    assert not paths.manifest_file.exists()
