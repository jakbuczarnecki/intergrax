# © Artur Czarnecki. All rights reserved.

"""Typed optional worker-construction fault injection for product composition qualification."""

from __future__ import annotations

from typing import Literal

WorkerConstructionFaultMode = Literal["none", "typed_bootstrap_exception"]

_QUALIFICATION_SECRET_SENTINEL = "DG001B-R5-SECRET-SENTINEL"
_TYPED_BOOTSTRAP_EXCEPTION_MESSAGE = (
    f"create_kafka_worker composition failure {_QUALIFICATION_SECRET_SENTINEL}"
)

_VALID_FAULT_MODES = frozenset({"none", "typed_bootstrap_exception"})


def parse_worker_construction_fault_mode(raw: str) -> WorkerConstructionFaultMode:
    """Parse a bounded worker-construction fault mode from operator configuration."""
    normalized = raw.strip().lower()
    if not normalized:
        return "none"
    if normalized not in _VALID_FAULT_MODES:
        raise ValueError(
            "LOCAL_WORKSPACE_WORKER_CONSTRUCTION_FAULT must be one of: "
            "none, typed_bootstrap_exception.",
        )
    if normalized == "typed_bootstrap_exception":
        return "typed_bootstrap_exception"
    return "none"


def maybe_raise_worker_construction_fault(mode: WorkerConstructionFaultMode) -> None:
    """Raise a deterministic B6 composition failure when qualification mode is enabled."""
    if mode == "typed_bootstrap_exception":
        raise TypeError(_TYPED_BOOTSTRAP_EXCEPTION_MESSAGE)


def qualification_secret_sentinel() -> str:
    """Return the qualification-only sentinel used in thrown bootstrap exceptions."""
    return _QUALIFICATION_SECRET_SENTINEL
