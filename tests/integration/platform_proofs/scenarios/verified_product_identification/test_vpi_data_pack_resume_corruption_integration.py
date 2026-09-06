"""Real CUDA qualification for VPI data pack resume/corruption semantics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    ShardBuildSeams,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardStatus,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
    VpiDataPackReadyShardCorruptionError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    patch_canonical_model_identity,
    write_tiny_selected_dataset,
)

pytestmark = [pytest.mark.integration, pytest.mark.cuda]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_QUAL_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4d4"
_CUDA_PYTHON = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4a2" / "cuda-venv" / "Scripts" / "python.exe"
_ROW_COUNT = 120
_SHARD_SIZE = 25


def _cuda_available() -> bool:
    if not _CUDA_PYTHON.is_file():
        return False
    probe = subprocess.run(
        [
            str(_CUDA_PYTHON),
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


@pytest.fixture
def cuda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _cuda_available():
        pytest.skip("CUDA qualification venv unavailable")
    monkeypatch.setenv("VPI_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("VPI_EMBEDDING_PROVIDER_BATCH_SIZE", "16")
    patch_canonical_model_identity(monkeypatch)
    ensure_embedding_provider_integrations_registered()


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    return write_tiny_selected_dataset(tmp_path / "dataset", row_count=_ROW_COUNT)


def _config(
    output_root: Path,
    dataset_path: Path,
    manifest_path: Path,
    *,
    resume: bool = False,
    start_fresh: bool = False,
    stop_after_shard: int | None = None,
) -> DataPackBuildConfig:
    return DataPackBuildConfig(
        output_root=output_root,
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        shard_size=_SHARD_SIZE,
        max_records=_ROW_COUNT,
        resume=resume,
        start_fresh=start_fresh,
        stop_after_shard=stop_after_shard,
    )


def test_cuda_preflight_reports_gpu(cuda_env: None) -> None:
    output = subprocess.check_output(
        [
            str(_CUDA_PYTHON),
            "-c",
            "import torch; print(torch.__version__); print(torch.version.cuda); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')",
        ],
        text=True,
    )
    lines = output.strip().splitlines()
    assert len(lines) == 3
    assert lines[2] != "NO_CUDA"


def test_cuda_resume_recovery_after_interrupted_validating(
    tmp_path: Path,
    cuda_env: None,
) -> None:
    output_root = _QUAL_ROOT / "resume-interrupted"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path, manifest_path = _dataset(tmp_path)
    paths = resolve_data_pack_paths(output_root)

    first = run_resumable_data_pack_build(
        _config(output_root, dataset_path, manifest_path, start_fresh=True, stop_after_shard=1),
    )
    assert first.finalized is False
    state_after_partial = read_build_state_file(paths.build_state_file)
    assert state_after_partial.completed_shards == 1

    interrupted = False

    def mark_interrupted() -> None:
        nonlocal interrupted
        interrupted = True
        raise VpiDataPackBuildError("cuda qualification interruption after temp writes")

    with pytest.raises(VpiDataPackBuildError, match="cuda qualification interruption"):
        run_resumable_data_pack_build(
            _config(output_root, dataset_path, manifest_path, resume=True),
            build_seams=ShardBuildSeams(after_both_temp_writes=mark_interrupted),
        )
    assert interrupted is True

    resumed = run_resumable_data_pack_build(
        _config(output_root, dataset_path, manifest_path, resume=True),
    )
    assert resumed.finalized is True
    assert resumed.status is DataPackStatus.READY
    assert resumed.manifest is not None
    final_state = read_build_state_file(paths.build_state_file)
    assert final_state.completed_shards == 5
    assert all(shard.status is DataPackShardStatus.READY for shard in final_state.shards)
    assert not list(paths.relational_dir.glob("*.tmp"))
    assert not list(paths.embeddings_dir.glob("*.tmp"))


def test_cuda_ready_corruption_fail_closed_before_embedding(
    tmp_path: Path,
    cuda_env: None,
) -> None:
    output_root = _QUAL_ROOT / "corrupt-ready"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path, manifest_path = _dataset(tmp_path)
    paths = resolve_data_pack_paths(output_root)
    run_resumable_data_pack_build(
        _config(output_root, dataset_path, manifest_path, start_fresh=True, stop_after_shard=1),
    )
    state = read_build_state_file(paths.build_state_file)
    ready = next(shard for shard in state.shards if shard.ordinal == 1)
    assert ready.relational_relative_path is not None
    relational_path = paths.root / ready.relational_relative_path
    data = bytearray(relational_path.read_bytes())
    data[20] ^= 0xFF
    relational_path.write_bytes(data)

    with pytest.raises(VpiDataPackReadyShardCorruptionError, match="sha256 mismatch"):
        run_resumable_data_pack_build(
            _config(output_root, dataset_path, manifest_path, resume=True),
        )
