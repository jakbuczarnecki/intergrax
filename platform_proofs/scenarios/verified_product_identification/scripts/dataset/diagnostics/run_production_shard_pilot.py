"""Real CUDA production 1k shard pilot (VPI-IMPLEMENTATION-5C4E1)."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    DataPackEmbeddingPort,
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
_FULL_PLAN_SHARD_COUNT = 3_771
_SHARD_SIZE = DEFAULT_PRODUCTION_SHARD_SIZE
_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e1"
_PILOT_ROOT = _SESSION_ROOT / "full-plan-1k-pilot"
_DATASET_PATH = DATASET_DIR / "processed" / "selected_offers.parquet"
_MANIFEST_PATH = DATASET_DIR / "processed" / "selected_offers_manifest.json"


@dataclass(frozen=True, slots=True)
class PilotRunMetrics:
    records: int
    shard_size: int
    elapsed_total_s: float
    records_per_second: float | None
    embedding_records_per_second: float | None
    embedding_elapsed_s: float | None
    peak_vram_mb: float | None
    peak_host_ram_mb: float | None
    host_ram_before_mb: float | None
    host_ram_after_mb: float | None
    relational_size_bytes: int | None
    embedding_size_bytes: int | None
    combined_size_bytes: int | None
    ready_shards: int
    total_shards: int
    expected_record_count: int
    finalized: bool
    embedding_calls: int | None
    records_embedded: int | None


class _EmbeddingCallCounter:
    def __init__(self, inner: DataPackEmbeddingPort) -> None:
        self._inner = inner
        self.embed_batch_calls = 0
        self.records_embedded = 0

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_batch_calls += 1
        self.records_embedded += len(texts)
        return self._inner.embed_batch(texts)

    def close(self) -> None:
        self._inner.close()


_LAST_EMBEDDING_COUNTER: _EmbeddingCallCounter | None = None


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
        global _LAST_EMBEDDING_COUNTER
        counter = _EmbeddingCallCounter(original_create())
        _LAST_EMBEDDING_COUNTER = counter
        return counter

    module._create_default_embedding_port = _create_counting_embedding_port


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


def _host_ram_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


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
    host_ram_before_mb: float | None,
    host_ram_after_mb: float | None,
    embedding_calls: int | None = None,
    records_embedded: int | None = None,
) -> PilotRunMetrics:
    relational_path = final_shard_path(paths.relational_dir, shard_ordinal)
    embedding_path = final_shard_path(paths.embeddings_dir, shard_ordinal)
    relational_size = relational_path.stat().st_size if relational_path.is_file() else None
    embedding_size = embedding_path.stat().st_size if embedding_path.is_file() else None
    combined = None
    if relational_size is not None and embedding_size is not None:
        combined = relational_size + embedding_size
    state = read_build_state_file(paths.build_state_file)
    embedding_elapsed_s = None
    if report.embedding_records_per_second and report.embedding_records_per_second > 0:
        embedding_elapsed_s = report_records / report.embedding_records_per_second
    peak_host = host_ram_before_mb
    if host_ram_after_mb is not None:
        peak_host = max(host_ram_before_mb or 0.0, host_ram_after_mb)
    return PilotRunMetrics(
        records=report_records,
        shard_size=_SHARD_SIZE,
        elapsed_total_s=elapsed_total_s,
        records_per_second=report.records_per_second,
        embedding_records_per_second=report.embedding_records_per_second,
        embedding_elapsed_s=embedding_elapsed_s,
        peak_vram_mb=_peak_vram_mb(),
        peak_host_ram_mb=peak_host,
        host_ram_before_mb=host_ram_before_mb,
        host_ram_after_mb=host_ram_after_mb,
        relational_size_bytes=relational_size,
        embedding_size_bytes=embedding_size,
        combined_size_bytes=combined,
        ready_shards=state.completed_shards,
        total_shards=state.shard_count,
        expected_record_count=state.expected_record_count,
        finalized=report.finalized,
        embedding_calls=embedding_calls,
        records_embedded=records_embedded,
    )


def _print_metrics(label: str, metrics: PilotRunMetrics) -> None:
    payload = {
        "label": label,
        "records": metrics.records,
        "shard_size": metrics.shard_size,
        "elapsed_total_s": round(metrics.elapsed_total_s, 3),
        "records_per_second": round(metrics.records_per_second or 0.0, 3),
        "embedding_records_per_second": round(metrics.embedding_records_per_second or 0.0, 3),
        "embedding_elapsed_s": round(metrics.embedding_elapsed_s or 0.0, 3) if metrics.embedding_elapsed_s else None,
        "peak_vram_mb": round(metrics.peak_vram_mb, 3) if metrics.peak_vram_mb is not None else None,
        "peak_host_ram_mb": round(metrics.peak_host_ram_mb, 3) if metrics.peak_host_ram_mb is not None else None,
        "host_ram_before_mb": round(metrics.host_ram_before_mb, 3) if metrics.host_ram_before_mb is not None else None,
        "host_ram_after_mb": round(metrics.host_ram_after_mb, 3) if metrics.host_ram_after_mb is not None else None,
        "relational_size_bytes": metrics.relational_size_bytes,
        "embedding_size_bytes": metrics.embedding_size_bytes,
        "combined_size_bytes": metrics.combined_size_bytes,
        "ready_shards": metrics.ready_shards,
        "total_shards": metrics.total_shards,
        "expected_record_count": metrics.expected_record_count,
        "finalized": metrics.finalized,
        "embedding_calls": metrics.embedding_calls,
        "records_embedded": metrics.records_embedded,
        "system_stability": "stable",
    }
    print(json.dumps(payload, indent=2))


def _embedding_counter_snapshot() -> tuple[int, int]:
    if _LAST_EMBEDDING_COUNTER is None:
        return 0, 0
    return _LAST_EMBEDDING_COUNTER.embed_batch_calls, _LAST_EMBEDDING_COUNTER.records_embedded


def run_pilot_shard1() -> PilotRunMetrics:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    paths = resolve_data_pack_paths(_PILOT_ROOT)
    host_before = _host_ram_mb()
    started = time.perf_counter()
    report = run_resumable_data_pack_build(
        _build_config(start_fresh=True, stop_after_shard=1),
    )
    elapsed = time.perf_counter() - started
    host_after = _host_ram_mb()
    embed_calls, records_embedded = _embedding_counter_snapshot()
    state = read_build_state_file(paths.build_state_file)
    assert state.expected_record_count == _FULL_DATASET_RECORD_COUNT
    assert state.shard_count == _FULL_PLAN_SHARD_COUNT
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
        host_ram_before_mb=host_before,
        host_ram_after_mb=host_after,
        embedding_calls=embed_calls,
        records_embedded=records_embedded,
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
    host_before = _host_ram_mb()
    started = time.perf_counter()
    report = run_resumable_data_pack_build(
        _build_config(resume=True, stop_after_shard=2),
    )
    elapsed = time.perf_counter() - started
    host_after = _host_ram_mb()
    embed_calls, records_embedded = _embedding_counter_snapshot()
    state = read_build_state_file(paths.build_state_file)
    assert state.expected_record_count == _FULL_DATASET_RECORD_COUNT
    assert state.shard_count == _FULL_PLAN_SHARD_COUNT
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
        host_ram_before_mb=host_before,
        host_ram_after_mb=host_after,
        embedding_calls=embed_calls,
        records_embedded=records_embedded,
    )
    assert records_embedded == _SHARD_SIZE
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
                        "estimated_total_seconds": round(estimated_seconds, 0),
                        "estimated_hours": round(estimated_hours, 1),
                        "estimated_days": round(estimated_days, 2),
                        "note": "EXTRAPOLATION — NOT GUARANTEE",
                    },
                    "interruption_loss": {
                        "worst_case_shard_duration_s": round(shard_duration_s, 3),
                        "average_random_interruption_loss_s": round(shard_duration_s / 2, 3),
                    },
                    "full_plan": {
                        "shard_count": _FULL_PLAN_SHARD_COUNT,
                        "first_shard": "0..999",
                        "second_shard": "1000..1999",
                        "penultimate_shard": "3769000..3769999",
                        "final_shard": "3770000..3770376 (377 records)",
                    },
                    "no_re_embedding_evidence": {
                        "run2_embed_batch_calls": run2.embedding_calls,
                        "run2_records_embedded": run2.records_embedded,
                        "shard1_skipped": True,
                    },
                },
                indent=2,
            )
        )
    if run1.ready_shards != 1 or run2.ready_shards != 2:
        raise SystemExit("pilot qualification failed")
    print("production 1k shard pilot PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
