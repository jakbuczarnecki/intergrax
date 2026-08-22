# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed checkpoint revision CAS errors (PCM-CHECKPOINT-SCHEDULER-INTEGRITY · PCM-04)."""

from __future__ import annotations


class CheckpointRevisionConflictError(RuntimeError):
    """Raised when checkpoint save expected_revision does not match stored revision."""


class CheckpointStepRegressionError(RuntimeError):
    """Raised when checkpoint step_index regresses without explicit rollback semantics."""
