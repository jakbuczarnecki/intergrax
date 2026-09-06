"""Typed arena evaluation errors — fail-closed semantics."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_environment import (
    ArenaExecutionEnvironmentSnapshot,
)


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


class ArenaExecutionEnvironmentError(RuntimeError):
    """Arena pre-flight environment validation failed before candidate execution."""

    def __init__(self, snapshot: ArenaExecutionEnvironmentSnapshot) -> None:
        self.snapshot = snapshot
        message = snapshot.detail or snapshot.status.value
        super().__init__(message)


class FinalistQualificationSelectionError(ValueError):
    """Finalist qualification selection or candidate resolution is invalid."""

