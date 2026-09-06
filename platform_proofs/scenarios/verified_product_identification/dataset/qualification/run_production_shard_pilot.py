"""Real CUDA production 5k shard pilot (VPI-IMPLEMENTATION-5C4E1)."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardStatus,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DATASET_DIR,
    DEFAULT_PRODUCTION_SHARD_SIZE,
    final_shard_path,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    EmbeddingModelArtifactIdentity,
)

_FULL_DATASET_RECORD_COUNT = 3_770_377
_SHARD_SIZE = DEFAULT_PRODUCTION_SHARD_SIZE
_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e1"
_PILOT_ROOT = _SESSION_ROOT / "full-plan-5k-pilot"
_DATASET_PATH = DATASET_DIR / "processed" / "selected_offers.parquet"
_MANIFEST_PATH = DATASET_DIR / "processed" / "selected_offers_manifest.json"


@dataclass(frozen=True, slots=True)
class PilotRunMetrics:
    records: int
    shard_size: int
    elapsed_total_s: float
    records_per_second: float | None
    embedding_records_per_second: float | None
    peak_vram_mb: float | None
    relational_size_bytes: int | None
    embedding_size_bytes: int | None
    combined_size_bytes: int | None
    ready_shards: int
    total_shards: int
    expected_record_count: int
    finalized: bool


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

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    print(f"gpu={torch.cuda.get_device_name(0)}")


def _require_env() -> None:
    device = os.environ.get("VPI_EMBEDDING_DEVICE", "")
    batch = os.environ.get("VPI_EMBEDDING_PROVIDER_BATCH_SIZE", "")
    if device != "cuda":
        raise SystemExit(f"VPI_EMBEDDING_DEVICE must be cuda, got {device!r}")
    if batch != "16":
        raise SystemExit(f"VPI_EMBEDDING_PROVIDER_BATCH_SIZE must be 16, got {batch!r}")


def _build_config(
    *,
    resume: bool = False,
    start_fresh: bool = False,
    stop_after_shard: int | None = None,
) -> DataPackBuildConfig:
    return DataPackBuildConfig(
        output_root=_PILOT_ROOT,
        dataset_path=_DATASET_PATH,
        dataset_manifest_path=_MANIFEST_PATH,
        shard_size=_SHARD_SIZE,
        resume=resume,
        start_fresh=start_fresh,
        stop_after_shard=stop_after_shard,
    )


def _peak_vram_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _collect_run_metrics(
    *,
    report_records: int,
    elapsed_total_s: float,
    report,
    paths,
    shard_ordinal: int,
) -> PilotRunMetrics:
    relational_path = final_shard_path(paths.relational_dir, shard_ordinal)
    embedding_path = final_shard_path(paths.embeddings_dir, shard_ordinal)
    relational_size = relational_path.stat().st_size if relational_path.is_file() else None
    embedding_size = embedding_path.stat().st_size if embedding_path.is_file() else None
    combined = None
    if relational_size is not None and embedding_size is not None:
        combined = relational_size + embedding_size
    state = read_build_state_file(paths.build_state_file)
    return PilotRunMetrics(
        records=report_records,
        shard_size=_SHARD_SIZE,
        elapsed_total_s=elapsed_total_s,
        records_per_second=report.records_per_second,
        embedding_records_per_second=report.embedding_records_per_second,
        peak_vram_mb=_peak_vram_mb(),
        relational_size_bytes=relational_size,
        embedding_size_bytes=embedding_size,
        combined_size_bytes=combined,
        ready_shards=state.completed_shards,
        total_shards=state.shard_count,
        expected_record_count=state.expected_record_count,
        finalized=report.finalized,
    )


def _print_metrics(label: str, metrics: PilotRunMetrics) -> None:
    payload = {
        "label": label,
        "records": metrics.records,
        "shard_size": metrics.shard_size,
        "elapsed_total_s": round(metrics.elapsed_total_s, 3),
        "records_per_second": round(metrics.records_per_second or 0.0, 3),
        "embedding_records_per_second": round(metrics.embedding_records_per_second or 0.0, 3),
        "peak_vram_mb": round(metrics.peak_vram_mb, 3) if metrics.peak_vram_mb is not None else None,
        "relational_size_bytes": metrics.relational_size_bytes,
        "embedding_size_bytes": metrics.embedding_size_bytes,
        "combined_size_bytes": metrics.combined_size_bytes,
        "ready_shards": metrics.ready_shards,
        "total_shards": metrics.total_shards,
        "expected_record_count": metrics.expected_record_count,
        "finalized": metrics.finalized,
    }
    print(json.dumps(payload, indent=2))


def run_pilot_shard1() -> PilotRunMetrics:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    paths = resolve_data_pack_paths(_PILOT_ROOT)
    started = time.perf_counter()
    report = run_resumable_data_pack_build(
        _build_config(start_fresh=True, stop_after_shard=1),
    )
    elapsed = time.perf_counter() - started
    state = read_build_state_file(paths.build_state_file)
    assert state.expected_record_count == _FULL_DATASET_RECORD_COUNT
    assert state.shard_count == 755
    assert state.completed_shards == 1
    assert state.shards[0].status is DataPackShardStatus.READY
    assert state.shards[1].status is DataPackShardStatus.PENDING
    assert report.finalized is False
    assert not paths.manifest_file.exists()
    metrics = _collect_run_metrics(
        report_records=_SHARD_SIZE,
        elapsed_total_s=elapsed,
        report=report,
        paths=paths,
        shard_ordinal=1,
    )
    _print_metrics("pilot_run_1", metrics)
    return metrics


def run_pilot_shard2_resume() -> PilotRunMetrics:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    paths = resolve_data_pack_paths(_PILOT_ROOT)
    state_before = read_build_state_file(paths.build_state_file)
    assert state_before.shards[0].status is DataPackShardStatus.READY
    started = time.perf_counter()
    report = run_resumable_data_pack_build(
        _build_config(resume=True, stop_after_shard=2),
    )
    elapsed = time.perf_counter() - started
    state = read_build_state_file(paths.build_state_file)
    assert state.expected_record_count == _FULL_DATASET_RECORD_COUNT
    assert state.shard_count == 755
    assert state.completed_shards == 2
    assert state.shards[0].status is DataPackShardStatus.READY
    assert state.shards[1].status is DataPackShardStatus.READY
    assert state.shards[2].status is DataPackShardStatus.PENDING
    assert report.finalized is False
    assert not paths.manifest_file.exists()
    metrics = _collect_run_metrics(
        report_records=_SHARD_SIZE,
        elapsed_total_s=elapsed,
        report=report,
        paths=paths,
        shard_ordinal=2,
    )
    _print_metrics("pilot_run_2_resume", metrics)
    return metrics


def main() -> int:
    _preflight()
    _require_env()
    _patch_model_identity()
    _SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    run1 = run_pilot_shard1()
    run2 = run_pilot_shard2_resume()
    extrapolation_records_per_second = run1.records_per_second or 0.0
    if extrapolation_records_per_second > 0:
        estimated_seconds = _FULL_DATASET_RECORD_COUNT / extrapolation_records_per_second
        estimated_hours = estimated_seconds / 3600
        estimated_days = estimated_hours / 24
        shard_duration_s = run1.elapsed_total_s
        print(
            json.dumps(
                {
                    "extrapolation": {
                        "records_per_second": round(extrapolation_records_per_second, 3),
                        "estimated_hours": round(estimated_hours, 1),
                        "estimated_days": round(estimated_days, 2),
                        "note": "extrapolation from pilot run 1 only; not a guarantee",
                    },
                    "interruption_loss": {
                        "worst_case_shard_duration_s": round(shard_duration_s, 3),
                        "average_random_interruption_loss_s": round(shard_duration_s / 2, 3),
                    },
                    "full_plan": {
                        "shard_count": 755,
                        "first_shard": "0..4999",
                        "second_shard": "5000..9999",
                        "final_shard": "3770000..3770376 (377 records)",
                    },
                },
                indent=2,
            )
        )
    if run1.ready_shards != 1 or run2.ready_shards != 2:
        raise SystemExit("pilot qualification failed")
    print("production 5k shard pilot PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
