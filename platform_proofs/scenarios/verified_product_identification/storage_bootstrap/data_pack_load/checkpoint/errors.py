"""Checkpoint-specific failures for storage bootstrap resume."""

from __future__ import annotations


class BootstrapCheckpointError(Exception):
    """Base error for durable bootstrap checkpoint coordination."""


class CheckpointNotFound(BootstrapCheckpointError):
    """No checkpoint exists for the requested run identity."""


class CheckpointAlreadyExists(BootstrapCheckpointError):
    """Fresh run attempted while durable checkpoint already exists."""


class CheckpointIncompatible(BootstrapCheckpointError):
    """Checkpoint does not match current bootstrap request or data pack."""


class CheckpointCorrupt(BootstrapCheckpointError):
    """Checkpoint payload is malformed, inconsistent, or fails integrity checks."""


class CheckpointConcurrentModification(BootstrapCheckpointError):
    """Optimistic concurrency guard rejected a stale checkpoint write."""


class CheckpointPersistenceError(BootstrapCheckpointError):
    """Durable checkpoint persistence failed."""
