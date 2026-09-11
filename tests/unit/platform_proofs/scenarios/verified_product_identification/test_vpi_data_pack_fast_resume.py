"""Fast resume boundary and O(1) artifact validation on resume."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_state_machine import (
    recover_non_ready_shard,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resume_policy import (
    resolve_resume_boundary,
    validate_contiguous_ready_prefix,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardBuildState,
    DataPackShardStatus,
    read_build_state_file,
    write_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config import (
    build_resume_cli_argv,
    resolve_vpi_5c4f_resume_launch_plan,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_fast_resume_test_support import (
    build_state_with_ready_prefix,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    patch_canonical_model_identity,
    run_resumable_data_pack_build_with_fake_policy,
    write_selected_dataset_with_manifest_count,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit

_PRODUCTION_SHARD_COUNT = 3771
_PRODUCTION_READY = 1200
_PRODUCTION_RECORD_COUNT = 3_770_377


def _seed_build_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    row_count: int,
    shard_size: int,
) -> tuple[Path, Path, Path, str]:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_selected_dataset_with_manifest_count(
        tmp_path / "dataset",
        parquet_row_count=row_count,
        manifest_record_count=row_count,
    )
    output_root = tmp_path / "pack"
    seed_config = DataPackBuildConfig(
        output_root=output_root,
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        shard_size=shard_size,
        max_records=row_count,
        start_fresh=True,
        stop_after_shard=1,
    )
    run_resumable_data_pack_build_with_fake_policy(
        seed_config,
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    state = read_build_state_file(resolve_data_pack_paths(output_root).build_state_file)
    return dataset_path, manifest_path, output_root, state.content_identity


def test_resolve_boundary_zero_ready() -> None:
    state = build_state_with_ready_prefix(
        record_count=100,
        shard_size=25,
        ready_prefix=0,
    )
    boundary = resolve_resume_boundary(state)
    assert boundary.last_ready_ordinal is None
    assert boundary.next_shard_ordinal == 1
    assert boundary.build_complete is False


def test_resolve_boundary_one_ready() -> None:
    state = build_state_with_ready_prefix(
        record_count=100,
        shard_size=25,
        ready_prefix=1,
    )
    boundary = resolve_resume_boundary(state)
    assert boundary.last_ready_ordinal == 1
    assert boundary.next_shard_ordinal == 2


def test_resolve_boundary_1200_ready_next_1201() -> None:
    state = build_state_with_ready_prefix(
        record_count=_PRODUCTION_RECORD_COUNT,
        shard_size=1000,
        ready_prefix=_PRODUCTION_READY,
    )
    assert state.shard_count == _PRODUCTION_SHARD_COUNT
    boundary = resolve_resume_boundary(state)
    assert boundary.completed_shards == 1200
    assert boundary.last_ready_ordinal == 1200
    assert boundary.next_shard_ordinal == 1201


@pytest.mark.parametrize(
    "interrupted_status",
    [
        DataPackShardStatus.EMBEDDING,
        DataPackShardStatus.WRITING,
        DataPackShardStatus.VALIDATING,
        DataPackShardStatus.DERIVING,
    ],
)
def test_interrupted_next_shard_recovery(
    tmp_path: Path,
    interrupted_status: DataPackShardStatus,
) -> None:
    state = build_state_with_ready_prefix(
        record_count=100,
        shard_size=25,
        ready_prefix=2,
        interrupted_status=interrupted_status,
    )
    paths = resolve_data_pack_paths(tmp_path / "pack")
    paths.relational_dir.mkdir(parents=True)
    paths.embeddings_dir.mkdir(parents=True)
    next_shard = state.shards[2]
    recovered = recover_non_ready_shard(
        next_shard,
        relational_dir=paths.relational_dir,
        embeddings_dir=paths.embeddings_dir,
    )
    assert recovered.status is DataPackShardStatus.PENDING
    assert recovered.attempt == next_shard.attempt + 1


def test_ready_gap_fails_closed() -> None:
    state = build_state_with_ready_prefix(
        record_count=100,
        shard_size=25,
        ready_prefix=2,
    )
    gap_shard = replace(
        state.shards[3],
        status=DataPackShardStatus.READY,
        relational_relative_path="relational/part-000004.parquet",
        embedding_relative_path="embeddings/part-000004.parquet",
        relational_sha256="d" * 64,
        embedding_sha256="e" * 64,
        relational_source_ref_set_sha256="f" * 64,
        embedding_source_ref_set_sha256="f" * 64,
    )
    broken_shards = state.shards[:3] + (gap_shard,) + state.shards[4:]
    with pytest.raises(ValueError, match="contiguous READY prefix"):
        replace(state, shards=broken_shards, completed_shards=3)


def test_completed_shards_mismatch_with_prefix_fails() -> None:
    state = build_state_with_ready_prefix(
        record_count=100,
        shard_size=25,
        ready_prefix=2,
    )
    with pytest.raises(ValueError, match="completed_shards must equal READY shard count"):
        replace(state, completed_shards=1)


def test_all_shards_ready_build_complete() -> None:
    state = build_state_with_ready_prefix(
        record_count=100,
        shard_size=25,
        ready_prefix=4,
    )
    boundary = resolve_resume_boundary(state)
    assert boundary.build_complete is True
    assert boundary.next_shard_ordinal is None


def test_performance_contract_many_shards_one_validation_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    row_count = 1201
    shard_size = 1
    ready_prefix = 1200

    def counting_validator(**kwargs: object) -> None:
        shard = kwargs["shard"]
        assert isinstance(shard, DataPackShardBuildState)
        calls.append(shard.ordinal)

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder.validate_ready_shard_artifacts",
        counting_validator,
    )
    dataset_path, manifest_path, output_root, content_identity = _seed_build_authority(
        tmp_path,
        monkeypatch,
        row_count=row_count,
        shard_size=shard_size,
    )
    paths = resolve_data_pack_paths(output_root)
    state = build_state_with_ready_prefix(
        record_count=row_count,
        shard_size=shard_size,
        ready_prefix=ready_prefix,
        content_identity=content_identity,
    )
    write_build_state_file(paths.build_state_file, state)

    config = DataPackBuildConfig(
        output_root=output_root,
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        shard_size=shard_size,
        max_records=row_count,
        resume=True,
        stop_after_shard=ready_prefix,
    )
    report = run_resumable_data_pack_build_with_fake_policy(
        config,
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert report.finalized is False
    assert calls == [ready_prefix]


def test_historical_shard_corruption_not_scanned_on_fast_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shard 50 corruption is not detected by fast resume; 5C4G owns full discovery."""
    validated_ordinals: list[int] = []

    def track(**kwargs: object) -> None:
        shard = kwargs["shard"]
        assert isinstance(shard, DataPackShardBuildState)
        validated_ordinals.append(shard.ordinal)

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder.validate_ready_shard_artifacts",
        track,
    )
    dataset_path, manifest_path, output_root, content_identity = _seed_build_authority(
        tmp_path,
        monkeypatch,
        row_count=50_000,
        shard_size=1000,
    )
    paths = resolve_data_pack_paths(output_root)
    state = build_state_with_ready_prefix(
        record_count=50_000,
        shard_size=1000,
        ready_prefix=49,
        content_identity=content_identity,
    )
    write_build_state_file(paths.build_state_file, state)
    config = DataPackBuildConfig(
        output_root=output_root,
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        shard_size=1000,
        max_records=50_000,
        resume=True,
        stop_after_shard=49,
    )
    run_resumable_data_pack_build_with_fake_policy(
        config,
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert validated_ordinals == [49]


def test_resume_validates_only_boundary_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validated: list[int] = []

    def track_validate(**kwargs: object) -> None:
        shard = kwargs["shard"]
        assert isinstance(shard, DataPackShardBuildState)
        validated.append(shard.ordinal)

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder.validate_ready_shard_artifacts",
        track_validate,
    )
    dataset_path, manifest_path, output_root, content_identity = _seed_build_authority(
        tmp_path,
        monkeypatch,
        row_count=120,
        shard_size=25,
    )
    paths = resolve_data_pack_paths(output_root)
    state = build_state_with_ready_prefix(
        record_count=120,
        shard_size=25,
        ready_prefix=2,
        content_identity=content_identity,
    )
    write_build_state_file(paths.build_state_file, state)
    config = DataPackBuildConfig(
        output_root=output_root,
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        shard_size=25,
        max_records=120,
        resume=True,
        stop_after_shard=2,
    )
    run_resumable_data_pack_build_with_fake_policy(
        config,
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert validated == [2]


def test_resume_does_not_clear_existing_build_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path, manifest_path, output_root, content_identity = _seed_build_authority(
        tmp_path,
        monkeypatch,
        row_count=25,
        shard_size=25,
    )
    paths = resolve_data_pack_paths(output_root)
    state = build_state_with_ready_prefix(
        record_count=25,
        shard_size=25,
        ready_prefix=0,
        content_identity=content_identity,
    )
    write_build_state_file(paths.build_state_file, state)
    argv = build_resume_cli_argv(resolve_vpi_5c4f_resume_launch_plan())
    assert "--resume" in argv
    assert "--start-fresh" not in argv
    assert paths.build_state_file.is_file()
