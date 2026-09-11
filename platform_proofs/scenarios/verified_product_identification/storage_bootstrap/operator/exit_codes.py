"""Stable exit semantics for the VPI full storage load operator."""

from __future__ import annotations

from enum import IntEnum


class StorageLoadOperatorExitCode(IntEnum):
    SUCCESS = 0
    LOAD_FAILED = 1
    PRECONDITION_ERROR = 2
    INTERRUPTED = 3
