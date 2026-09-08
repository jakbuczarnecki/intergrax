"""CUDA runtime helpers for bounded batch throughput qualification."""

from __future__ import annotations

import gc
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider

from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    BatchVariantMeasurement,
    CudaPreflightSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.metrics import (
    compute_throughput_metrics,
    compute_vram_headroom_fraction,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.selection import (
    evaluate_batch_safety,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    EmbeddingExecutionPort,
)


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA qualification preconditions are not met."""


class ResourcePreconditionError(RuntimeError):
    """Raised when GPU resources are already under substantial pressure."""


@dataclass(frozen=True, slots=True)
class GpuMemorySnapshot:
    total_bytes: int
    free_bytes: int
    used_bytes: int


def resolve_qualification_python_executable(repo_root: Path) -> Path:
    preferred = repo_root / ".venv" / "Scripts" / "python.exe"
    session_cuda = (
        repo_root
        / ".tmp"
        / "session"
        / "vpi-5c4a2"
        / "cuda-venv"
        / "Scripts"
        / "python.exe"
    )
    if preferred.is_file():
        return preferred
    if session_cuda.is_file():
        return session_cuda
    return Path(sys.executable)


def run_cuda_preflight(
    *,
    minimum_free_fraction: float = 0.20,
) -> CudaPreflightSnapshot:
    python_version = sys.version.split()[0]
    try:
        import torch
    except ImportError as exc:
        return CudaPreflightSnapshot(
            python_version=python_version,
            torch_version="unavailable",
            cuda_runtime_version=None,
            cuda_available=False,
            gpu_name=None,
            gpu_total_memory_bytes=None,
            gpu_free_memory_bytes_before_load=None,
            gpu_used_memory_bytes_before_load=None,
            resource_precondition_fail_reason=f"torch is not installed: {exc}",
        )

    torch_version = torch.__version__
    cuda_runtime_version = torch.version.cuda
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return CudaPreflightSnapshot(
            python_version=python_version,
            torch_version=torch_version,
            cuda_runtime_version=cuda_runtime_version,
            cuda_available=False,
            gpu_name=None,
            gpu_total_memory_bytes=None,
            gpu_free_memory_bytes_before_load=None,
            gpu_used_memory_bytes_before_load=None,
            resource_precondition_fail_reason="torch.cuda.is_available() returned False",
        )

    gpu_name = torch.cuda.get_device_name(0)
    memory = read_gpu_memory_snapshot()
    used_fraction = memory.used_bytes / memory.total_bytes
    free_fraction = memory.free_bytes / memory.total_bytes
    precondition_reason: str | None = None
    if free_fraction < minimum_free_fraction:
        precondition_reason = (
            "unrelated GPU applications occupy substantial VRAM before model load: "
            f"free={free_fraction:.1%}, required>={minimum_free_fraction:.0%}"
        )
    return CudaPreflightSnapshot(
        python_version=python_version,
        torch_version=torch_version,
        cuda_runtime_version=cuda_runtime_version,
        cuda_available=True,
        gpu_name=gpu_name,
        gpu_total_memory_bytes=memory.total_bytes,
        gpu_free_memory_bytes_before_load=memory.free_bytes,
        gpu_used_memory_bytes_before_load=memory.used_bytes,
        resource_precondition_fail_reason=precondition_reason,
    )


def read_gpu_memory_snapshot(device_index: int = 0) -> GpuMemorySnapshot:
    import torch

    if not torch.cuda.is_available():
        msg = "CUDA is unavailable"
        raise CudaUnavailableError(msg)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    used_bytes = total_bytes - free_bytes
    return GpuMemorySnapshot(
        total_bytes=int(total_bytes),
        free_bytes=int(free_bytes),
        used_bytes=int(used_bytes),
    )


def require_hf_embedding_provider(
    embedding_port: EmbeddingExecutionPort,
) -> HFEmbeddingProvider:
    if not isinstance(embedding_port, IntergraxEmbeddingBootstrapAdapter):
        msg = "bounded CUDA qualification requires IntergraxEmbeddingBootstrapAdapter"
        raise TypeError(msg)
    provider = embedding_port.embedding_provider()
    if not isinstance(provider, HFEmbeddingProvider):
        msg = "bounded CUDA qualification requires HFEmbeddingProvider"
        raise TypeError(msg)
    return provider


def configure_hf_provider_batch_size(
    provider: HFEmbeddingProvider,
    batch_size: int,
) -> None:
    if batch_size <= 0:
        msg = "batch_size must be > 0"
        raise ValueError(msg)
    provider._batch_size = int(batch_size)


def reset_cuda_peak_memory_stats() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release_cuda_cache() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def is_cuda_oom_error(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    message = str(exc).casefold()
    return "out of memory" in message


def run_warmup(
    embedding_port: EmbeddingExecutionPort,
    texts: Sequence[str],
    *,
    provider: HFEmbeddingProvider,
) -> None:
    if not texts:
        msg = "warmup texts must not be empty"
        raise ValueError(msg)
    warmup_texts = texts[:2]
    configure_hf_provider_batch_size(provider, 1)
    embedding_port.embed_batch(warmup_texts)
    synchronize_cuda()
    reset_cuda_peak_memory_stats()


def measure_batch_variant(
    embedding_port: EmbeddingExecutionPort,
    *,
    provider: HFEmbeddingProvider,
    document_texts: tuple[str, ...],
    token_counts: tuple[int, ...],
    batch_size: int,
    previous_smaller: BatchVariantMeasurement | None,
) -> BatchVariantMeasurement:
    record_count = len(document_texts)
    total_tokens = sum(token_counts)
    gpu_total = read_gpu_memory_snapshot().total_bytes
    free_before = read_gpu_memory_snapshot().free_bytes

    configure_hf_provider_batch_size(provider, batch_size)
    release_cuda_cache()
    reset_cuda_peak_memory_stats()

    cuda_oom = False
    wall_clock_seconds = 0.0
    try:
        synchronize_cuda()
        started = time.perf_counter()
        synchronize_cuda()
        embedding_port.embed_batch(document_texts)
        synchronize_cuda()
        wall_clock_seconds = time.perf_counter() - started
    except BaseException as exc:
        cuda_oom = is_cuda_oom_error(exc)
        if not cuda_oom:
            raise
    finally:
        synchronize_cuda()

    import torch

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    free_after = read_gpu_memory_snapshot().free_bytes
    headroom = compute_vram_headroom_fraction(
        gpu_total_memory_bytes=gpu_total,
        peak_cuda_allocated_bytes=peak_allocated,
    )

    if cuda_oom or wall_clock_seconds <= 0.0:
        records_per_second = 0.0
        tokens_per_second = 0.0
        average_milliseconds_per_record = 0.0
    else:
        records_per_second, tokens_per_second, average_milliseconds_per_record = (
            compute_throughput_metrics(
                record_count=record_count,
                total_tokens=total_tokens,
                wall_clock_seconds=wall_clock_seconds,
            )
        )

    provisional = BatchVariantMeasurement(
        batch_size=batch_size,
        record_count=record_count,
        total_bounded_input_tokens=total_tokens,
        average_tokens_per_record=total_tokens / record_count,
        p50_tokens=0.0,
        p95_tokens=0.0,
        max_tokens=max(token_counts),
        wall_clock_embedding_seconds=wall_clock_seconds,
        records_per_second=records_per_second,
        tokens_per_second=tokens_per_second,
        average_milliseconds_per_record=average_milliseconds_per_record,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        gpu_total_memory_bytes=gpu_total,
        gpu_free_memory_bytes_before=free_before,
        gpu_free_memory_bytes_after=free_after,
        cuda_oom=cuda_oom,
        vram_headroom_fraction=headroom,
        safe=False,
        safety_detail="pending",
    )
    safe, safety_detail = evaluate_batch_safety(
        provisional,
        previous_smaller=previous_smaller,
    )
    return BatchVariantMeasurement(
        batch_size=provisional.batch_size,
        record_count=provisional.record_count,
        total_bounded_input_tokens=provisional.total_bounded_input_tokens,
        average_tokens_per_record=provisional.average_tokens_per_record,
        p50_tokens=provisional.p50_tokens,
        p95_tokens=provisional.p95_tokens,
        max_tokens=provisional.max_tokens,
        wall_clock_embedding_seconds=provisional.wall_clock_embedding_seconds,
        records_per_second=provisional.records_per_second,
        tokens_per_second=provisional.tokens_per_second,
        average_milliseconds_per_record=provisional.average_milliseconds_per_record,
        peak_cuda_allocated_bytes=provisional.peak_cuda_allocated_bytes,
        peak_cuda_reserved_bytes=provisional.peak_cuda_reserved_bytes,
        gpu_total_memory_bytes=provisional.gpu_total_memory_bytes,
        gpu_free_memory_bytes_before=provisional.gpu_free_memory_bytes_before,
        gpu_free_memory_bytes_after=provisional.gpu_free_memory_bytes_after,
        cuda_oom=provisional.cuda_oom,
        vram_headroom_fraction=provisional.vram_headroom_fraction,
        safe=safe,
        safety_detail=safety_detail,
    )


def count_hf_model_loads(provider: HFEmbeddingProvider) -> int:
    return 1 if provider._model is not None else 0
