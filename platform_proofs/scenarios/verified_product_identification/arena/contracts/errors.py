"""Typed arena evaluation errors — fail-closed semantics."""

from __future__ import annotations


class EmbeddingArenaBenchmarkGroundTruthError(ValueError):
    """Benchmark ground truth cannot be resolved within the stage corpus."""


class EmbeddingArenaTokenizerUnavailableError(RuntimeError):
    """Tokenizer required for truncation profiling is unavailable."""


class EmbeddingArenaTruncationProfileError(RuntimeError):
    """Truncation profiling failed for a candidate that requires evidence."""


class EmbeddingArenaEvaluationScopeError(ValueError):
    """Stage evaluation scope is invalid or incomplete."""


class EmbeddingArenaCandidateSessionError(RuntimeError):
    """Candidate execution session lifecycle violation."""
