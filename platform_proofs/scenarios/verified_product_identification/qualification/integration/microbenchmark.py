"""Provider batch microbenchmark — integration-side torch usage allowed."""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    MicrobenchmarkCandidateResult,
    MicrobenchmarkCandidateStatus,
    WarmupTimingSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    EmbeddingExecutionPort,
)


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc).casefold()
    return "out of memory" in message


def _peak_vram_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except (ImportError, AttributeError):
        return None


def _reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, AttributeError):
        return


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        return


def build_embedding_execution_port(
    configuration: VpiEmbeddingConfiguration,
    *,
    provider_batch_size: int,
    device: str | None,
    max_length: int | None,
) -> IntergraxEmbeddingBootstrapAdapter:
    from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
        VpiEmbeddingProviderExecutionConfiguration,
    )

    execution_configuration = VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(
            device=device,
            batch_size=provider_batch_size,
            max_length=max_length,
        )
    )
    return IntergraxEmbeddingBootstrapAdapter(
        configuration,
        execution_configuration=execution_configuration,
    )


def measure_warmup_timing(
    embedding: EmbeddingExecutionPort,
    texts: Sequence[str],
) -> WarmupTimingSnapshot:
    if not texts:
        msg = "texts must not be empty"
        raise ValueError(msg)
    warmup_text = texts[0]
    steady_texts = texts[1:5] if len(texts) > 1 else (warmup_text,)

    init_started = time.perf_counter()
    embedding.probe()
    provider_init_seconds = time.perf_counter() - init_started

    first_started = time.perf_counter()
    embedding.embed_batch((warmup_text,))
    first_embed_seconds = time.perf_counter() - first_started

    steady_started = time.perf_counter()
    embedding.embed_batch(steady_texts)
    steady_embed_seconds = time.perf_counter() - steady_started

    return WarmupTimingSnapshot(
        provider_init_seconds=provider_init_seconds,
        first_embed_seconds=first_embed_seconds,
        steady_embed_seconds=steady_embed_seconds,
    )


def run_provider_batch_candidate(
    configuration: VpiEmbeddingConfiguration,
    texts: Sequence[str],
    *,
    provider_batch_size: int,
    device: str | None,
    max_length: int | None,
    expected_dimension: int,
) -> MicrobenchmarkCandidateResult:
    record_count = len(texts)
    embedding = build_embedding_execution_port(
        configuration,
        provider_batch_size=provider_batch_size,
        device=device,
        max_length=max_length,
    )
    _reset_peak_vram()
    try:
        embedding.probe()
        embed_started = time.perf_counter()
        vectors = embedding.embed_batch(texts)
        embed_elapsed_seconds = time.perf_counter() - embed_started
        if len(vectors) != record_count:
            return MicrobenchmarkCandidateResult(
                provider_batch_size=provider_batch_size,
                record_count=record_count,
                embed_elapsed_seconds=embed_elapsed_seconds,
                records_per_second=0.0,
                status=MicrobenchmarkCandidateStatus.FAILED,
                peak_vram_bytes=_peak_vram_bytes(),
                detail="vector count mismatch",
            )
        if vectors and len(vectors[0]) != expected_dimension:
            return MicrobenchmarkCandidateResult(
                provider_batch_size=provider_batch_size,
                record_count=record_count,
                embed_elapsed_seconds=embed_elapsed_seconds,
                records_per_second=0.0,
                status=MicrobenchmarkCandidateStatus.FAILED,
                peak_vram_bytes=_peak_vram_bytes(),
                detail="dimension mismatch",
            )
        records_per_second = (
            record_count / embed_elapsed_seconds if embed_elapsed_seconds > 0 else 0.0
        )
        return MicrobenchmarkCandidateResult(
            provider_batch_size=provider_batch_size,
            record_count=record_count,
            embed_elapsed_seconds=embed_elapsed_seconds,
            records_per_second=records_per_second,
            status=MicrobenchmarkCandidateStatus.PASS,
            peak_vram_bytes=_peak_vram_bytes(),
            detail=None,
        )
    except BaseException as exc:
        status = (
            MicrobenchmarkCandidateStatus.FAILED_OOM
            if _is_cuda_oom(exc)
            else MicrobenchmarkCandidateStatus.FAILED
        )
        return MicrobenchmarkCandidateResult(
            provider_batch_size=provider_batch_size,
            record_count=record_count,
            embed_elapsed_seconds=0.0,
            records_per_second=0.0,
            status=status,
            peak_vram_bytes=_peak_vram_bytes(),
            detail=str(exc),
        )
    finally:
        embedding.close()
        _release_cuda_cache()


def resolve_provider_device_proof(
    configuration: VpiEmbeddingConfiguration,
    *,
    provider_batch_size: int,
    device: str | None,
    max_length: int | None,
) -> tuple[str | None, str]:
    embedding = build_embedding_execution_port(
        configuration,
        provider_batch_size=provider_batch_size,
        device=device,
        max_length=max_length,
    )
    try:
        embedding.probe()
        return embedding.provider_device_snapshot()
    finally:
        embedding.close()
