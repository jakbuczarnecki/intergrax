"""Operator contract tests for VPI full Data Pack storage load."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchSize,
    BootstrapFinalStatus,
    RelationalBatch,
    RelationalTargetId,
    ResumeMode,
    StorageLoadBatchResult,
    VectorBatch,
    VectorTargetId,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.cli import (
    main as operator_cli_main,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.config import (
    OperatorRunMode,
    StorageLoadOperatorConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.exit_codes import (
    StorageLoadOperatorExitCode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.lock import (
    OperatorLock,
    build_operator_lock_metadata,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.composition import (
    StorageLoadOperatorRuntime,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.runner import (
    StorageLoadOperatorRunner,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
    write_runtime_qualification_pack,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
    FakeDataPackReader,
    FakeRelationalAdapter,
    FakeVectorAdapter,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_reader import (
    _write_ready_pack,
)

pytestmark = pytest.mark.unit

_RECORD_COUNT = 5
_BATCH_SIZE = 2
_RELATIONAL_TARGET = RelationalTargetId("vpi-products")
_VECTOR_TARGET = VectorTargetId("vpi-product-embeddings")


def _operator_config(
    tmp_path: Path,
    *,
    run_mode: OperatorRunMode,
    expected_record_count: int | None = None,
    expected_content_identity: str | None = None,
    batch_size: int = _BATCH_SIZE,
) -> StorageLoadOperatorConfig:
    return StorageLoadOperatorConfig(
        artifact_root=tmp_path / "pack",
        checkpoint_root=tmp_path / "checkpoint",
        evidence_root=tmp_path / "evidence",
        relational_target=_RELATIONAL_TARGET,
        vector_target=_VECTOR_TARGET,
        batch_size=BootstrapBatchSize(batch_size),
        run_mode=run_mode,
        verification_mode=VerificationMode.STRICT,
        expected_record_count=expected_record_count,
        expected_data_pack_content_identity=expected_content_identity,
    )


def _build_fake_runtime(
    config: StorageLoadOperatorConfig,
    *,
    pair_count: int = _RECORD_COUNT,
    status: DataPackStatus = DataPackStatus.READY,
    relational: FakeRelationalAdapter | None = None,
    vector: FakeVectorAdapter | None = None,
) -> StorageLoadOperatorRuntime:
    from dataclasses import replace

    from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.filesystem_reader import (
        FilesystemDataPackBootstrapReader,
    )
    from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.composition import (
        StorageLoadOperatorRuntime,
    )
    from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
        _build_pairs,
    )

    if config.artifact_root.is_dir():
        reader = FilesystemDataPackBootstrapReader(config.artifact_root)
    else:
        reader = FakeDataPackReader(
            manifest=replace(_ready_manifest(pair_count), status=status),
            pairs=_build_pairs(pair_count),
        )
    resolved_relational = relational or FakeRelationalAdapter()
    resolved_vector = vector or FakeVectorAdapter()
    store = FilesystemBootstrapCheckpointStore(config.checkpoint_root)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=resolved_relational,
            vector=resolved_vector,
            checkpoint_store=store,
        )
    )
    return StorageLoadOperatorRuntime(
        service=service,
        reader=reader,
        relational=resolved_relational,
        vector=resolved_vector,
    )


def _runner_with_fake_runtime(
    fake_builder: Callable[[StorageLoadOperatorConfig], StorageLoadOperatorRuntime],
) -> StorageLoadOperatorRunner:
    return StorageLoadOperatorRunner(
        runtime_builder=fake_builder,
        validate_providers=False,
    )


@pytest.fixture
def ready_pack(tmp_path: Path) -> Path:
    pack_root = tmp_path / "pack"
    return write_runtime_qualification_pack(pack_root, record_count=_RECORD_COUNT)


def _manifest_content_identity(pack_root: Path) -> str:
    from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.filesystem_reader import (
        FilesystemDataPackBootstrapReader,
    )

    reader = FilesystemDataPackBootstrapReader(pack_root)
    try:
        return reader.read_manifest().content_identity
    finally:
        reader.close()


def test_operator_config_validation() -> None:
    config = StorageLoadOperatorConfig(
        artifact_root=Path("/tmp/pack"),
        checkpoint_root=Path("/tmp/checkpoint"),
        evidence_root=Path("/tmp/evidence"),
        relational_target=_RELATIONAL_TARGET,
        vector_target=_VECTOR_TARGET,
        batch_size=BootstrapBatchSize(2),
        run_mode=OperatorRunMode.PLAN,
        verification_mode=VerificationMode.STRICT,
    )
    assert config.plan_only is True
    assert config.resume_mode is ResumeMode.FRESH


def test_cli_requires_mode(tmp_path: Path) -> None:
    code = operator_cli_main(
        [
            "--artifact-root",
            str(tmp_path / "pack"),
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--relational-target",
            "vpi-products",
            "--vector-target",
            "vpi-product-embeddings",
            "--batch-size",
            "2",
        ]
    )
    assert code == int(StorageLoadOperatorExitCode.PRECONDITION_ERROR)


def test_plan_mode_zero_provider_writes(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.PLAN)
    relational = FakeRelationalAdapter()
    vector = FakeVectorAdapter()
    checkpoint_root = config.checkpoint_root
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    runner = _runner_with_fake_runtime(
        lambda cfg: _build_fake_runtime(cfg, relational=relational, vector=vector),
    )
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.SUCCESS
    assert outcome.result is not None
    assert outcome.result.total_relational_written == 0
    assert outcome.result.total_vectors_written == 0
    assert relational.storage == {}
    assert vector.storage == {}
    assert list(checkpoint_root.glob("**/*")) == []
    assert (config.evidence_root / "run.json").is_file()
    assert (config.evidence_root / "final-report.json").is_file()


def test_fresh_load_commits_checkpoint(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    runner = _runner_with_fake_runtime(lambda cfg: _build_fake_runtime(cfg))
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.SUCCESS
    assert outcome.result is not None
    assert outcome.result.status is BootstrapFinalStatus.SUCCESS
    assert outcome.result.total_relational_written == _RECORD_COUNT
    checkpoint_files = list(config.checkpoint_root.glob("**/state.json"))
    assert checkpoint_files


def test_resume_after_complete_fresh_zero_writes(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    relational = FakeRelationalAdapter()
    vector = FakeVectorAdapter()
    runner = _runner_with_fake_runtime(
        lambda cfg: _build_fake_runtime(cfg, relational=relational, vector=vector),
    )
    fresh = runner.run(config)
    assert fresh.exit_code is StorageLoadOperatorExitCode.SUCCESS
    resume_config = _operator_config(tmp_path, run_mode=OperatorRunMode.RESUME)
    resume = runner.run(resume_config)
    assert resume.exit_code is StorageLoadOperatorExitCode.SUCCESS
    assert resume.result is not None
    assert resume.result.total_relational_written == 0
    assert resume.result.total_vectors_written == 0


class InterruptingRelationalAdapter(FakeRelationalAdapter):
    def __init__(self, interrupt_on_batch: int) -> None:
        super().__init__()
        self.interrupt_on_batch = interrupt_on_batch
        self.interrupted = False

    def write_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult:
        if batch.batch_number == self.interrupt_on_batch and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt("simulated operator interruption")
        return super().write_batch(batch)


def test_interruption_preserves_checkpoint_and_resume(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    relational = InterruptingRelationalAdapter(interrupt_on_batch=1)
    runner = _runner_with_fake_runtime(
        lambda cfg: _build_fake_runtime(cfg, relational=relational),
    )
    interrupted = runner.run(config)
    assert interrupted.exit_code is StorageLoadOperatorExitCode.INTERRUPTED
    resume_runner = _runner_with_fake_runtime(
        lambda cfg: _build_fake_runtime(cfg, relational=FakeRelationalAdapter()),
    )
    resume = resume_runner.run(_operator_config(tmp_path, run_mode=OperatorRunMode.RESUME))
    assert resume.exit_code is StorageLoadOperatorExitCode.SUCCESS
    assert resume.result is not None
    assert resume.result.total_relational_written == _RECORD_COUNT - _BATCH_SIZE


def test_reject_not_ready_artifact(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path / "pack", record_count=_RECORD_COUNT, status=DataPackStatus.BUILDING)
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.PLAN)
    runner = StorageLoadOperatorRunner(validate_providers=False)
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_wrong_expected_record_count(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(
        tmp_path,
        run_mode=OperatorRunMode.PLAN,
        expected_record_count=_RECORD_COUNT + 1,
    )
    runner = StorageLoadOperatorRunner(validate_providers=False)
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_wrong_expected_content_identity(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(
        tmp_path,
        run_mode=OperatorRunMode.PLAN,
        expected_content_identity="deadbeef",
    )
    runner = StorageLoadOperatorRunner(validate_providers=False)
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_missing_artifact(tmp_path: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.PLAN)
    runner = StorageLoadOperatorRunner(validate_providers=False)
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_fresh_with_existing_checkpoint(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    runner = _runner_with_fake_runtime(lambda cfg: _build_fake_runtime(cfg))
    assert runner.run(config).exit_code is StorageLoadOperatorExitCode.SUCCESS
    second = runner.run(config)
    assert second.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_resume_without_checkpoint(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.RESUME)
    runner = _runner_with_fake_runtime(lambda cfg: _build_fake_runtime(cfg))
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_resume_incompatible_batch_size(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    runner = _runner_with_fake_runtime(lambda cfg: _build_fake_runtime(cfg))
    assert runner.run(config).exit_code is StorageLoadOperatorExitCode.SUCCESS
    resume_config = _operator_config(tmp_path, run_mode=OperatorRunMode.RESUME, batch_size=3)
    outcome = runner.run(resume_config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_resume_different_target(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    runner = _runner_with_fake_runtime(lambda cfg: _build_fake_runtime(cfg))
    assert runner.run(config).exit_code is StorageLoadOperatorExitCode.SUCCESS
    resume_config = StorageLoadOperatorConfig(
        artifact_root=config.artifact_root,
        checkpoint_root=config.checkpoint_root,
        evidence_root=tmp_path / "evidence-resume",
        relational_target=RelationalTargetId("other-target"),
        vector_target=_VECTOR_TARGET,
        batch_size=BootstrapBatchSize(_BATCH_SIZE),
        run_mode=OperatorRunMode.RESUME,
        verification_mode=VerificationMode.STRICT,
    )
    outcome = runner.run(resume_config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_reject_missing_provider_configuration(tmp_path: Path, ready_pack: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_POSTGRESQL_HOST", raising=False)
    monkeypatch.delenv("INTERGRAX_POSTGRESQL_DSN", raising=False)
    monkeypatch.delenv("INTERGRAX_QDRANT_HOST", raising=False)
    monkeypatch.delenv("INTERGRAX_QDRANT_URL", raising=False)
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.PLAN)
    outcome = StorageLoadOperatorRunner().run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_second_writer_lock_conflict(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.FRESH)
    identity = _manifest_content_identity(config.artifact_root)
    first_lock = OperatorLock(
        checkpoint_root=config.checkpoint_root,
        metadata=build_operator_lock_metadata(artifact_content_identity=identity),
    )
    first_lock.acquire()
    runner = _runner_with_fake_runtime(lambda cfg: _build_fake_runtime(cfg))
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR
    first_lock.release()


def test_evidence_root_failure(tmp_path: Path, ready_pack: Path) -> None:
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.PLAN)
    blocker_file = tmp_path / "blocker.txt"
    blocker_file.write_text("not-a-directory", encoding="utf-8")
    config = StorageLoadOperatorConfig(
        artifact_root=config.artifact_root,
        checkpoint_root=config.checkpoint_root,
        evidence_root=blocker_file,
        relational_target=config.relational_target,
        vector_target=config.vector_target,
        batch_size=config.batch_size,
        run_mode=config.run_mode,
        verification_mode=config.verification_mode,
    )
    runner = StorageLoadOperatorRunner(validate_providers=False)
    outcome = runner.run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.PRECONDITION_ERROR


def test_secrets_not_in_evidence(tmp_path: Path, ready_pack: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_POSTGRESQL_HOST", "localhost")
    monkeypatch.setenv("INTERGRAX_POSTGRESQL_PASSWORD", "super-secret-password")
    monkeypatch.setenv("INTERGRAX_QDRANT_HOST", "localhost")
    config = _operator_config(tmp_path, run_mode=OperatorRunMode.PLAN)
    outcome = StorageLoadOperatorRunner().run(config)
    assert outcome.exit_code is StorageLoadOperatorExitCode.SUCCESS
    payload = (config.evidence_root / "run.json").read_text(encoding="utf-8")
    assert "super-secret-password" not in payload
    assert "postgresql://" not in payload.lower()


def test_verification_defaults_to_strict(tmp_path: Path) -> None:
    code = operator_cli_main(
        [
            "--artifact-root",
            str(tmp_path / "pack"),
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--relational-target",
            "vpi-products",
            "--vector-target",
            "vpi-product-embeddings",
            "--batch-size",
            "2",
            "--plan",
        ]
    )
    assert code == int(StorageLoadOperatorExitCode.PRECONDITION_ERROR)
