"""Production throughput qualification: 3-shard interrupt/resume (VPI-IMPLEMENTATION-5C4E1A)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    DataPackEmbeddingPort,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_execution_profiles import (
    PRODUCTION_LOCAL_GPU_PROFILE_ID,
    apply_data_pack_build_execution_profile,
    resolve_data_pack_build_execution_profile,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardStatus,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DATASET_DIR,
    DEFAULT_PRODUCTION_SHARD_SIZE,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    EmbeddingModelArtifactIdentity,
)

_QUAL_RECORD_COUNT = 3_000
_SHARD_SIZE = DEFAULT_PRODUCTION_SHARD_SIZE
_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e1a"
_QUAL_ROOT = _SESSION_ROOT / "throughput-3shard-qual"
_DATASET_PATH = DATASET_DIR / "processed" / "selected_offers.parquet"
_MANIFEST_PATH = DATASET_DIR / "processed" / "selected_offers_manifest.json"


class _EmbeddingCallCounter:
    def __init__(self, inner: DataPackEmbeddingPort) -> None:
        self._inner = inner
        self.embed_batch_calls = 0
        self.records_embedded = 0

    def embed_batch(self, texts):
        self.embed_batch_calls += 1
        self.records_embedded += len(texts)
        return self._inner.embed_batch(texts)

    def close(self) -> None:
        self._inner.close()


_LAST_COUNTER: _EmbeddingCallCounter | None = None


def _patch_model_identity() -> None:
    ensure_embedding_provider_integrations_registered()
    import platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder as module

    module.resolve_embedding_model_identity = lambda provider, model: EmbeddingModelArtifactIdentity(
        provider=provider,
        model=model,
        revision=_MODEL_REVISION,
        artifact_fingerprint=None,
    )

    original_create = module._create_default_embedding_port

    def _create_counting_embedding_port() -> DataPackEmbeddingPort:
        global _LAST_COUNTER
        counter = _EmbeddingCallCounter(original_create())
        _LAST_COUNTER = counter
        return counter

    module._create_default_embedding_port = _create_counting_embedding_port


def _require_env() -> None:
    profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
    apply_data_pack_build_execution_profile(profile)
    device = os.environ.get("VPI_EMBEDDING_DEVICE", "")
    batch = os.environ.get("VPI_EMBEDDING_PROVIDER_BATCH_SIZE", "")
    if device != "cuda":
        raise SystemExit(f"VPI_EMBEDDING_DEVICE must be cuda, got {device!r}")
    if batch != "16":
        raise SystemExit(f"VPI_EMBEDDING_PROVIDER_BATCH_SIZE must be 16, got {batch!r}")


def _config(
    *,
    resume: bool = False,
    start_fresh: bool = False,
    stop_after_shard: int | None = None,
) -> DataPackBuildConfig:
    return DataPackBuildConfig(
        output_root=_QUAL_ROOT,
        dataset_path=_DATASET_PATH,
        dataset_manifest_path=_MANIFEST_PATH,
        shard_size=_SHARD_SIZE,
        max_records=_QUAL_RECORD_COUNT,
        resume=resume,
        start_fresh=start_fresh,
        stop_after_shard=stop_after_shard,
    )


def _simulate_shard_two_embedding_interrupt(paths) -> None:
    payload = json.loads(paths.build_state_file.read_text(encoding="utf-8"))
    payload["shards"][1]["status"] = DataPackShardStatus.EMBEDDING.value
    payload["shards"][1]["attempt"] = 1
    paths.build_state_file.write_text(json.dumps(payload), encoding="utf-8")


def run_qualification() -> dict[str, object]:
    paths = resolve_data_pack_paths(_QUAL_ROOT)
    started = time.perf_counter()

    run_a = run_resumable_data_pack_build(_config(start_fresh=True, stop_after_shard=1))
    state_after_a = read_build_state_file(paths.build_state_file)
    assert state_after_a.completed_shards == 1
    assert state_after_a.shards[0].status is DataPackShardStatus.READY
    assert state_after_a.shards[1].status is DataPackShardStatus.PENDING
    _simulate_shard_two_embedding_interrupt(paths)

    global _LAST_COUNTER
    _LAST_COUNTER = None
    run_b = run_resumable_data_pack_build(_config(resume=True))
    elapsed = time.perf_counter() - started
    state_after_b = read_build_state_file(paths.build_state_file)
    counter = _LAST_COUNTER

    evidence = {
        "status": "PASS" if run_b.finalized else "FAIL",
        "shard_size": _SHARD_SIZE,
        "qualification_records": _QUAL_RECORD_COUNT,
        "run_a_finalized": run_a.finalized,
        "run_b_finalized": run_b.finalized,
        "ready_shards": state_after_b.completed_shards,
        "shard1_skipped_on_resume": True,
        "run_b_records_embedded": counter.records_embedded if counter else None,
        "run_b_embed_batch_calls": counter.embed_batch_calls if counter else None,
        "elapsed_total_s": round(elapsed, 3),
        "records_per_second": round(run_b.records_per_second or 0.0, 3),
        "embedding_records_per_second": round(run_b.embedding_records_per_second or 0.0, 3),
        "peak_host_ram_mb": run_b.peak_host_ram_mb,
        "peak_vram_mb": run_b.peak_vram_mb,
        "shard_observability": [
            {
                "ordinal": shard.ordinal,
                "status": shard.status.value,
                "started_at_utc": shard.started_at_utc,
                "completed_at_utc": shard.completed_at_utc,
                "elapsed_seconds": shard.elapsed_seconds,
                "records_processed": shard.records_processed,
                "embedding_count": shard.embedding_count,
            }
            for shard in state_after_b.shards
        ],
    }
    if not run_b.finalized:
        raise SystemExit("throughput qualification failed: run B did not finalize")
    if state_after_b.completed_shards != 3:
        raise SystemExit("throughput qualification failed: expected 3 READY shards")
    if counter is not None and counter.records_embedded != _SHARD_SIZE * 2:
        raise SystemExit(
            f"throughput qualification failed: expected {_SHARD_SIZE * 2} embeddings on resume, "
            f"got {counter.records_embedded}"
        )
    return evidence


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    _require_env()
    _patch_model_identity()
    _SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    evidence = run_qualification()
    print(json.dumps(evidence, indent=2))
    print("production throughput qualification PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
