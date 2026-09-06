"""Corruption and recovery qualification for resumable VPI data pack builder."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_state_machine import (
    recover_non_ready_shard,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    assert_data_pack_compatible,
    default_v1_expectations,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    ShardBuildSeams,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardBuildState,
    DataPackShardStatus,
    build_state_from_json_dict,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
    VpiDataPackBuildIdentityMismatchError,
    VpiDataPackBuildStateError,
    VpiDataPackReadyShardCorruptionError,
    VpiDataPackResumeError,
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
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resume_corruption_test_support import (
    PartialBuildFixture,
    ReadyShardSnapshot,
    apply_non_ready_mutation,
    assert_no_temp_files,
    assert_not_distributable,
    assert_ready_shard_immutable,
    corrupt_ready_shard_metadata_digest,
    delete_file,
    flip_file_byte,
    install_dimension_corrupt_ready_shard,
    install_duplicate_source_ref_ready_shard,
    install_pair_mismatch_ready_shard,
    install_record_count_mismatch_ready_shard,
    install_schema_corrupt_ready_shard,
    prepare_partial_build,
    read_build_state_dict,
    resume_and_finalize,
    run_build,
    setup_non_ready_state,
    snapshot_ready_shard,
    update_shard,
    write_build_state_dict,
)

pytestmark = pytest.mark.unit

TARGET_ORDINAL = 2
READY_ORDINAL = 1

NonReadySetup: TypeAlias = Callable[[PartialBuildFixture], None]


def _paths(fixture: PartialBuildFixture):
    return resolve_data_pack_paths(fixture.output_root)


def _ready_snapshot_before_resume(fixture: PartialBuildFixture) -> ReadyShardSnapshot:
    return snapshot_ready_shard(fixture, READY_ORDINAL)


def _assert_prior_ready_unchanged(
    fixture: PartialBuildFixture,
    before: ReadyShardSnapshot,
    embedding_port: FakeDataPackEmbeddingPort,
) -> None:
    after = snapshot_ready_shard(fixture, READY_ORDINAL)
    assert_ready_shard_immutable(before, after)
    assert embedding_port.embed_calls > 0


RECOVERABLE_NON_READY_CASES: list[tuple[str, DataPackShardStatus, str, int]] = [
    ("A_pending_no_files", DataPackShardStatus.PENDING, "none", 0),
    ("B_pending_orphan_tmp", DataPackShardStatus.PENDING, "orphan_tmp", 0),
    ("D_deriving", DataPackShardStatus.DERIVING, "none", 1),
    ("E_embedding", DataPackShardStatus.EMBEDDING, "none", 1),
    ("F_writing_rel_tmp", DataPackShardStatus.WRITING, "writing_rel_tmp", 1),
    ("F_writing_both_tmp", DataPackShardStatus.WRITING, "writing_both_tmp", 1),
    ("F_writing_malformed_tmp", DataPackShardStatus.WRITING, "writing_malformed_tmp", 1),
    ("G_validating_both_tmp", DataPackShardStatus.VALIDATING, "validating_both_tmp", 1),
    ("H_validating_rel_final_emb_tmp", DataPackShardStatus.VALIDATING, "validating_rel_final_emb_tmp", 1),
    ("I_validating_rel_final_only", DataPackShardStatus.VALIDATING, "validating_rel_final_only", 1),
    ("J_validating_emb_final_only", DataPackShardStatus.VALIDATING, "validating_emb_final_only", 1),
    ("J_validating_both_finals", DataPackShardStatus.VALIDATING, "validating_both_finals", 1),
]


@pytest.mark.parametrize(
    ("case_id", "status", "mutation", "attempt_increment"),
    RECOVERABLE_NON_READY_CASES,
    ids=[case[0] for case in RECOVERABLE_NON_READY_CASES],
)
def test_recoverable_non_ready_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    status: DataPackShardStatus,
    mutation: str,
    attempt_increment: int,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    before_ready = _ready_snapshot_before_resume(fixture)
    apply_non_ready_mutation(
        fixture,
        target_ordinal=TARGET_ORDINAL,
        status=status,
        mutation=mutation,
    )
    resume_embedding = FakeDataPackEmbeddingPort()
    report = resume_and_finalize(fixture, resume_embedding)
    assert report.finalized is True
    assert report.status is DataPackStatus.READY
    state = read_build_state_file(_paths(fixture).build_state_file)
    target = next(shard for shard in state.shards if shard.ordinal == TARGET_ORDINAL)
    assert target.status is DataPackShardStatus.READY
    assert target.attempt >= 1 + attempt_increment
    _assert_prior_ready_unchanged(fixture, before_ready, resume_embedding)
    assert_no_temp_files(fixture)


def test_pending_orphan_final_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    apply_non_ready_mutation(
        fixture,
        target_ordinal=TARGET_ORDINAL,
        status=DataPackShardStatus.PENDING,
        mutation="orphan_final",
    )
    with pytest.raises(VpiDataPackResumeError, match="orphan final shard"):
        resume_and_finalize(fixture, FakeDataPackEmbeddingPort())


def _delete_relational(fixture: PartialBuildFixture) -> ReadyShardSnapshot:
    before = snapshot_ready_shard(fixture, READY_ORDINAL)
    delete_file(before.relational_path)
    return before


def _delete_embedding(fixture: PartialBuildFixture) -> ReadyShardSnapshot:
    before = snapshot_ready_shard(fixture, READY_ORDINAL)
    delete_file(before.embedding_path)
    return before


def _flip_relational(fixture: PartialBuildFixture) -> ReadyShardSnapshot:
    before = snapshot_ready_shard(fixture, READY_ORDINAL)
    flip_file_byte(before.relational_path)
    return before


def _flip_embedding(fixture: PartialBuildFixture) -> ReadyShardSnapshot:
    before = snapshot_ready_shard(fixture, READY_ORDINAL)
    flip_file_byte(before.embedding_path)
    return before


FATAL_READY_CASES: list[tuple[str, Callable[[PartialBuildFixture], ReadyShardSnapshot], str]] = [
    ("L_ready_relational_missing", _delete_relational, "files missing"),
    ("M_ready_embedding_missing", _delete_embedding, "files missing"),
    ("N_ready_relational_sha", _flip_relational, "relational shard 1 sha256 mismatch"),
    ("O_ready_embedding_sha", _flip_embedding, "embedding shard 1 sha256 mismatch"),
    (
        "P_ready_relational_digest",
        lambda fixture: corrupt_ready_shard_metadata_digest(
            fixture,
            READY_ORDINAL,
            relational_digest="f" * 64,
        ),
        "validation failed",
    ),
    (
        "Q_ready_embedding_digest",
        lambda fixture: corrupt_ready_shard_metadata_digest(
            fixture,
            READY_ORDINAL,
            embedding_digest="f" * 64,
        ),
        "validation failed",
    ),
    ("R_ready_pair_mismatch", lambda fixture: install_pair_mismatch_ready_shard(fixture, READY_ORDINAL), "validation failed"),
    ("S_ready_schema_corruption", lambda fixture: install_schema_corrupt_ready_shard(fixture, READY_ORDINAL), "validation failed"),
    ("T_ready_dimension_corruption", lambda fixture: install_dimension_corrupt_ready_shard(fixture, READY_ORDINAL), "validation failed"),
    ("U_ready_duplicate_source_ref", lambda fixture: install_duplicate_source_ref_ready_shard(fixture, READY_ORDINAL), "validation failed"),
    ("V_ready_record_count_mismatch", lambda fixture: install_record_count_mismatch_ready_shard(fixture, READY_ORDINAL), "validation failed"),
]


@pytest.mark.parametrize(
    ("case_id", "setup", "message"),
    FATAL_READY_CASES,
    ids=[case[0] for case in FATAL_READY_CASES],
)
def test_fatal_ready_corruption_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    setup: Callable[[PartialBuildFixture], ReadyShardSnapshot],
    message: str,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    setup_non_ready_state(fixture, target_ordinal=TARGET_ORDINAL, status=DataPackShardStatus.PENDING)
    before = setup(fixture)
    rel_before = before.relational_path.read_bytes() if before.relational_path.is_file() else b""
    emb_before = before.embedding_path.read_bytes() if before.embedding_path.is_file() else b""
    resume_embedding = FakeDataPackEmbeddingPort()
    with pytest.raises(VpiDataPackReadyShardCorruptionError, match=message):
        resume_and_finalize(fixture, resume_embedding)
    assert resume_embedding.embed_calls == 0
    after = snapshot_ready_shard(fixture, READY_ORDINAL)
    assert after.status is DataPackShardStatus.READY
    if rel_before:
        assert after.relational_path.read_bytes() == rel_before
    if emb_before:
        assert after.embedding_path.read_bytes() == emb_before
    target = next(
        shard
        for shard in read_build_state_file(_paths(fixture).build_state_file).shards
        if shard.ordinal == TARGET_ORDINAL
    )
    assert target.status is DataPackShardStatus.PENDING


def test_valid_ready_shard_skipped_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    before = snapshot_ready_shard(fixture, READY_ORDINAL)
    rel_mtime = before.relational_path.stat().st_mtime_ns
    emb_mtime = before.embedding_path.stat().st_mtime_ns
    setup_non_ready_state(fixture, target_ordinal=TARGET_ORDINAL, status=DataPackShardStatus.PENDING)
    resume_embedding = FakeDataPackEmbeddingPort()
    resume_and_finalize(fixture, resume_embedding)
    after = snapshot_ready_shard(fixture, READY_ORDINAL)
    assert_ready_shard_immutable(before, after)
    assert after.relational_path.stat().st_mtime_ns == rel_mtime
    assert after.embedding_path.stat().st_mtime_ns == emb_mtime


def test_build_state_metadata_immutable_on_valid_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    before_state = read_build_state_file(_paths(fixture).build_state_file)
    before_shard = next(shard for shard in before_state.shards if shard.ordinal == READY_ORDINAL)
    setup_non_ready_state(fixture, target_ordinal=TARGET_ORDINAL, status=DataPackShardStatus.EMBEDDING)
    resume_and_finalize(fixture, FakeDataPackEmbeddingPort())
    after_state = read_build_state_file(_paths(fixture).build_state_file)
    after_shard = next(shard for shard in after_state.shards if shard.ordinal == READY_ORDINAL)
    assert after_shard.relational_sha256 == before_shard.relational_sha256
    assert after_shard.embedding_sha256 == before_shard.embedding_sha256
    assert after_shard.relational_source_ref_set_sha256 == before_shard.relational_source_ref_set_sha256
    assert after_shard.embedding_source_ref_set_sha256 == before_shard.embedding_source_ref_set_sha256


def test_malformed_build_state_json_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    paths = _paths(fixture)
    paths.build_state_file.write_text("{truncated", encoding="utf-8")
    with pytest.raises(VpiDataPackBuildStateError, match="failed to read build state"):
        resume_and_finalize(fixture, FakeDataPackEmbeddingPort())


def test_unsupported_build_state_version_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    payload = read_build_state_dict(fixture)
    payload["state_version"] = "vpi.data_pack.build_state/99"
    write_build_state_dict(fixture, payload)
    with pytest.raises(VpiDataPackBuildStateError, match="unsupported build state version"):
        resume_and_finalize(fixture, FakeDataPackEmbeddingPort())


def test_semantic_invalid_completed_shards_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = 2
    with pytest.raises(ValueError, match="completed_shards must equal READY shard count"):
        build_state_from_json_dict(payload)


def test_semantic_ready_without_sha_metadata_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    update_shard(fixture, READY_ORDINAL, relational_sha256=None)
    with pytest.raises(ValueError, match="relational_sha256 is required"):
        read_build_state_file(_paths(fixture).build_state_file)


def test_content_identity_mismatch_blocks_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    payload = read_build_state_dict(fixture)
    payload["content_identity"] = "deadbeef"
    write_build_state_dict(fixture, payload)
    resume_embedding = FakeDataPackEmbeddingPort()
    with pytest.raises(VpiDataPackBuildIdentityMismatchError, match="content_identity"):
        resume_and_finalize(fixture, resume_embedding)
    assert resume_embedding.embed_calls == 0


def test_shard_size_mismatch_blocks_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    resume_embedding = FakeDataPackEmbeddingPort()
    with pytest.raises(VpiDataPackResumeError, match="shard_size"):
        run_build(fixture, resume_embedding, resume=True, shard_size=10)


def test_record_count_mismatch_blocks_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    resume_embedding = FakeDataPackEmbeddingPort()
    with pytest.raises(VpiDataPackResumeError, match="expected_record_count"):
        run_build(fixture, resume_embedding, resume=True, max_records=25)


def test_crash_between_renames_recovered_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resume_corruption_test_support import (
        make_dataset,
    )

    patch_canonical = __import__(
        "tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support",
        fromlist=["patch_canonical_model_identity"],
    ).patch_canonical_model_identity
    patch_canonical(monkeypatch)
    dataset = make_dataset(tmp_path, row_count=25)
    fixture = PartialBuildFixture(
        output_root=tmp_path / "pack",
        dataset=dataset,
        shard_size=25,
        row_count=25,
    )
    paths = _paths(fixture)

    def fail_embedding_commit() -> None:
        raise VpiDataPackBuildError("simulated embedding rename failure")

    with pytest.raises(VpiDataPackBuildError, match="simulated embedding rename failure"):
        run_build(
            fixture,
            FakeDataPackEmbeddingPort(),
            start_fresh=True,
            build_seams=ShardBuildSeams(before_embedding_commit=fail_embedding_commit),
        )
    assert final_shard_path(paths.relational_dir, 1).exists()
    assert not final_shard_path(paths.embeddings_dir, 1).exists()
    report = resume_and_finalize(fixture, FakeDataPackEmbeddingPort())
    assert report.finalized is True
    assert_no_temp_files(fixture)


def test_partial_build_not_distributable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    assert_not_distributable(fixture)


def test_fresh_vs_resumed_semantic_equivalence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resume_corruption_test_support import (
        make_dataset,
    )
    from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
        patch_canonical_model_identity,
    )

    patch_canonical_model_identity(monkeypatch)
    fresh_fixture = PartialBuildFixture(
        output_root=tmp_path / "fresh" / "pack",
        dataset=make_dataset(tmp_path / "fresh"),
        shard_size=25,
        row_count=50,
    )
    run_build(fresh_fixture, FakeDataPackEmbeddingPort(), start_fresh=True)

    interrupted_fixture, _ = prepare_partial_build(tmp_path / "interrupted", monkeypatch, ready_shards=1)
    setup_non_ready_state(
        interrupted_fixture,
        target_ordinal=TARGET_ORDINAL,
        status=DataPackShardStatus.VALIDATING,
        ready_shards=1,
    )
    apply_non_ready_mutation(
        interrupted_fixture,
        target_ordinal=TARGET_ORDINAL,
        status=DataPackShardStatus.VALIDATING,
        mutation="validating_both_finals",
    )
    resume_and_finalize(interrupted_fixture, FakeDataPackEmbeddingPort())

    fresh_state = read_build_state_file(_paths(fresh_fixture).build_state_file)
    resumed_state = read_build_state_file(_paths(interrupted_fixture).build_state_file)
    assert fresh_state.expected_record_count == resumed_state.expected_record_count
    for fresh_shard, resumed_shard in zip(fresh_state.shards, resumed_state.shards, strict=True):
        assert fresh_shard.expected_record_count == resumed_shard.expected_record_count
        if fresh_shard.status is DataPackShardStatus.READY and resumed_shard.status is DataPackShardStatus.READY:
            assert (
                fresh_shard.relational_source_ref_set_sha256
                == resumed_shard.relational_source_ref_set_sha256
            )
            assert (
                fresh_shard.embedding_source_ref_set_sha256
                == resumed_shard.embedding_source_ref_set_sha256
            )


def test_finalization_after_recovery_matches_fresh_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _ = prepare_partial_build(tmp_path, monkeypatch, ready_shards=1)
    setup_non_ready_state(fixture, target_ordinal=TARGET_ORDINAL, status=DataPackShardStatus.WRITING)
    apply_non_ready_mutation(
        fixture,
        target_ordinal=TARGET_ORDINAL,
        status=DataPackShardStatus.WRITING,
        mutation="writing_both_tmp",
    )
    report = resume_and_finalize(fixture, FakeDataPackEmbeddingPort())
    assert report.finalized is True
    assert report.manifest is not None
    paths = _paths(fixture)
    assert paths.shards_index_file.is_file()
    assert paths.checksums_file.is_file()
    assert_data_pack_compatible(
        report.manifest,
        expectations=default_v1_expectations(
            derivation_version=report.manifest.derivation_version,
            semantic_text_version=report.manifest.semantic_text_version,
            embedding_provider=report.manifest.embedding_identity.provider,
            embedding_model=report.manifest.embedding_identity.model,
            embedding_model_revision=report.manifest.embedding_identity.model_revision or "",
            embedding_dimension=report.manifest.embedding_identity.dimension,
            source_dataset_sha256=report.manifest.source_dataset.dataset_sha256,
        ),
        pack_root=paths.root,
    )
    assert_no_temp_files(fixture)


def test_recover_non_ready_unit_pending_orphan_final_rejected(tmp_path: Path) -> None:
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
