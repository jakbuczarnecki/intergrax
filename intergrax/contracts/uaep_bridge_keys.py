# © Artur Czarnecki. All rights reserved.

"""Typed metadata keys for UAEP ↔ ACP kernel bridge (ACP-STEP-3)."""

from __future__ import annotations

from enum import StrEnum


class UaepBridgeMetadataKey(StrEnum):
    """In-memory session keys on ``RuntimeExecutionContext.metadata``."""

    KERNEL_SESSION = "uaep.kernel_session.v1"


class UaepStateDeltaKey(StrEnum):
    """Keys written into ACP state via UAEP bridge merge-patch."""

    LAST_STEP_ID = "uaep.last_step_id"
    LAST_STEP_SUMMARY = "uaep.last_step_summary"
