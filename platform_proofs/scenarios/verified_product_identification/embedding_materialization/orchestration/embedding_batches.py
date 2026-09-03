"""Embedding batch splitting for bounded provider invocation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


def iter_embedding_slices(
    texts: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[tuple[int, tuple[str, ...]]]:
    if batch_size <= 0:
        msg = "batch_size must be > 0"
        raise ValueError(msg)
    for start_index in range(0, len(texts), batch_size):
        yield start_index, tuple(texts[start_index : start_index + batch_size])
