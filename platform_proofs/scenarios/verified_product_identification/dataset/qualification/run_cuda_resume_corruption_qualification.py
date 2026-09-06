"""Real CUDA resume/corruption qualification runner (VPI-IMPLEMENTATION-5C4D4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    assert_data_pack_compatible,
    default_v1_expectations,
)
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
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    EmbeddingModelArtifactIdentity,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    write_tiny_selected_dataset,
)

_ROW_COUNT = 120
_SHARD_SIZE = 25
_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _patch_model_identity() -> None:
    ensure_embedding_provider_integrations_registered()
    import platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder as module

    module.resolve_embedding_model_identity = lambda provider, model: EmbeddingModelArtifactIdentity(
        provider=provider,
        model=model,
        revision=_MODEL_REVISION,
        artifact_fingerprint=None,
    )


def _preflight() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")


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


def _assert_finalized(output_root: Path) -> None:
    paths = resolve_data_pack_paths(output_root)
    state = read_build_state_file(paths.build_state_file)
    assert state.completed_shards == 5
    assert all(shard.status is DataPackShardStatus.READY for shard in state.shards)
    assert not list(paths.relational_dir.glob("*.tmp"))
    assert not list(paths.embeddings_dir.glob("*.tmp"))
    report_manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert report_manifest["status"] == DataPackStatus.READY.value
    assert paths.shards_index_file.is_file()
    assert paths.checksums_file.is_file()


def run_recovery_qualification(qual_root: Path, dataset_dir: Path) -> None:
    output_root = qual_root / "resume-interrupted"
    dataset_path, manifest_path = write_tiny_selected_dataset(dataset_dir, row_count=_ROW_COUNT)
    paths = resolve_data_pack_paths(output_root)

    first = run_resumable_data_pack_build(
        _config(output_root, dataset_path, manifest_path, start_fresh=True, stop_after_shard=1),
    )
    if not first.finalized:
        state = read_build_state_file(paths.build_state_file)
        assert state.completed_shards == 1

    def interrupt_after_temp_writes() -> None:
        raise VpiDataPackBuildError("cuda qualification interruption after both temp writes")

    try:
        run_resumable_data_pack_build(
            _config(output_root, dataset_path, manifest_path, resume=True),
            build_seams=ShardBuildSeams(after_both_temp_writes=interrupt_after_temp_writes),
        )
    except VpiDataPackBuildError as exc:
        if "cuda qualification interruption" not in str(exc):
            raise

    resumed = run_resumable_data_pack_build(
        _config(output_root, dataset_path, manifest_path, resume=True),
    )
    if not resumed.finalized or resumed.manifest is None:
        raise SystemExit("resume recovery did not finalize")
    assert_data_pack_compatible(
        resumed.manifest,
        expectations=default_v1_expectations(
            derivation_version=resumed.manifest.derivation_version,
            semantic_text_version=resumed.manifest.semantic_text_version,
            embedding_provider=resumed.manifest.embedding_identity.provider,
            embedding_model=resumed.manifest.embedding_identity.model,
            embedding_model_revision=resumed.manifest.embedding_identity.model_revision or "",
            embedding_dimension=resumed.manifest.embedding_identity.dimension,
            source_dataset_sha256=resumed.manifest.source_dataset.dataset_sha256,
        ),
        pack_root=paths.root,
    )
    _assert_finalized(output_root)
    print("cuda recovery qualification PASS")


def run_ready_corruption_qualification(qual_root: Path, dataset_dir: Path) -> None:
    output_root = qual_root / "corrupt-ready"
    dataset_path, manifest_path = write_tiny_selected_dataset(dataset_dir / "corrupt", row_count=_ROW_COUNT)
    paths = resolve_data_pack_paths(output_root)
    run_resumable_data_pack_build(
        _config(output_root, dataset_path, manifest_path, start_fresh=True, stop_after_shard=1),
    )
    state = read_build_state_file(paths.build_state_file)
    ready = next(shard for shard in state.shards if shard.ordinal == 1)
    if ready.relational_relative_path is None:
        raise SystemExit("READY shard missing relational path")
    relational_path = paths.root / ready.relational_relative_path
    data = bytearray(relational_path.read_bytes())
    data[20] ^= 0xFF
    relational_path.write_bytes(data)
    try:
        run_resumable_data_pack_build(
            _config(output_root, dataset_path, manifest_path, resume=True),
        )
    except VpiDataPackReadyShardCorruptionError:
        print("cuda ready corruption qualification PASS")
        return
    raise SystemExit("expected VpiDataPackReadyShardCorruptionError")


def main() -> int:
    _preflight()
    _patch_model_identity()
    qual_root = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4d4"
    dataset_dir = qual_root / "datasets"
    qual_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    run_recovery_qualification(qual_root, dataset_dir)
    run_ready_corruption_qualification(qual_root, dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
