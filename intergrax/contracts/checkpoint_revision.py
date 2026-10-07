# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed checkpoint revision CAS errors (PCM-CHECKPOINT-SCHEDULER-INTEGRITY · PCM-04)."""

from __future__ import annotations


class CheckpointRevisionConflictError(RuntimeError):
    """Raised when checkpoint save expected_revision does not match stored revision."""


class CheckpointStepRegressionError(RuntimeError):
    """Raised when checkpoint step_index regresses without explicit rollback semantics."""


class CheckpointAgentIdentityConflictError(RuntimeError):
    """Raised when checkpoint agent_id would change within a (run_id, tenant_id) stream."""


class CheckpointStreamIdentityConflictError(RuntimeError):
    """Raised when checkpoint run_id/tenant_id disagree with the active stream or resume context."""


class CheckpointSideEffectLineageError(RuntimeError):
    """Raised when embedded side-effect records disagree with checkpoint run/step bounds."""


class CheckpointDurableCorruptionError(RuntimeError):
    """Raised when durable checkpoint bytes disagree with authoritative storage metadata."""
