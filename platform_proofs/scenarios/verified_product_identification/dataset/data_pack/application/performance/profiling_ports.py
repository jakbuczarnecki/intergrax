"""Provider-neutral profiling wrappers for Data Pack integration points."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.contracts import (
    PipelinePhase,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.metrics import (
    record_embedding_batch,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.profiler import (
    PipelineProfilerPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackEmbeddingPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.sample_selection import (
    SelectedDatasetRow,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)


def resolve_tokenize_probe(
    embedding_port: DataPackEmbeddingPort,
) -> Callable[[Sequence[str]], None] | None:
    """Best-effort tokenizer timing probe for HF-backed embedding ports."""
    if not isinstance(embedding_port, IntergraxEmbeddingBootstrapAdapter):
        return None
    provider = getattr(embedding_port, "_provider", None)
    if provider is None:
        return None

    def probe(texts: Sequence[str]) -> None:
        ensure_model = getattr(provider, "_ensure_model", None)
        if callable(ensure_model):
            ensure_model()
        model = getattr(provider, "_model", None)
        if model is None:
            return
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return
        for text in texts:
            tokenizer.encode(text, add_special_tokens=True)

    return probe


class ProfilingEmbeddingPort:
    """Wraps an embedding port with inference and optional tokenization timing."""

    def __init__(
        self,
        inner: DataPackEmbeddingPort,
        profiler: PipelineProfilerPort,
        *,
        tokenize_probe: Callable[[Sequence[str]], None] | None = None,
    ) -> None:
        self._inner = inner
        self._profiler = profiler
        self._tokenize_probe = tokenize_probe

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._tokenize_probe is not None and texts:
            with self._profiler.measure(PipelinePhase.TOKENIZE):
                self._tokenize_probe(texts)
        with self._profiler.measure(PipelinePhase.EMBEDDING_INFERENCE):
            vectors = self._inner.embed_batch(texts)
        record_embedding_batch(self._profiler, batch_size=len(texts))
        return vectors

    def close(self) -> None:
        self._inner.close()


class ProfilingDatasetReader:
    """Wraps dataset row-range reads with read-phase timing."""

    def __init__(
        self,
        inner,
        profiler: PipelineProfilerPort,
    ) -> None:
        self._inner = inner
        self._profiler = profiler

    @property
    def row_group_index(self):
        return self._inner.row_group_index

    def read_range(
        self,
        start_row_index: int,
        end_row_index_exclusive: int,
    ) -> Iterable[SelectedDatasetRow]:
        with self._profiler.measure(PipelinePhase.READ):
            return self._inner.read_range(start_row_index, end_row_index_exclusive)


def profile_write_temp_shard(
    profiler: PipelineProfilerPort,
    directory: Path,
    shard_ordinal: int,
    write_callable: Callable[[Path], None],
    *,
    write_temp_shard_fn: Callable[[Path, int, Callable[[Path], None]], Path],
) -> Path:
    with profiler.measure(PipelinePhase.PARQUET_WRITE):
        return write_temp_shard_fn(directory, shard_ordinal, write_callable)
