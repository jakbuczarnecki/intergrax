# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded task memory limits (§27 — explicit and bounded)."""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_MAX_RECORDS_PER_TASK = 256
DEFAULT_MAX_VALUE_BYTES = 65_536
DEFAULT_MAX_NAMESPACE_LENGTH = 64
DEFAULT_MAX_KEY_LENGTH = 128


@dataclass(frozen=True)
class TaskMemoryLimits:
    max_records_per_task: int = DEFAULT_MAX_RECORDS_PER_TASK
    max_value_bytes: int = DEFAULT_MAX_VALUE_BYTES
    max_namespace_length: int = DEFAULT_MAX_NAMESPACE_LENGTH
    max_key_length: int = DEFAULT_MAX_KEY_LENGTH


def estimate_value_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def validate_namespace_key(
    *,
    namespace: str,
    key: str,
    limits: TaskMemoryLimits,
) -> None:
    ns = (namespace or "").strip()
    k = (key or "").strip()
    if not ns:
        raise ValueError("task memory namespace must not be empty")
    if not k:
        raise ValueError("task memory key must not be empty")
    if len(ns) > limits.max_namespace_length:
        raise ValueError(f"task memory namespace exceeds {limits.max_namespace_length} characters")
    if len(k) > limits.max_key_length:
        raise ValueError(f"task memory key exceeds {limits.max_key_length} characters")


def validate_value_size(value: object, *, limits: TaskMemoryLimits) -> None:
    size = estimate_value_bytes(value)
    if size > limits.max_value_bytes:
        raise ValueError(
            f"task memory value exceeds {limits.max_value_bytes} bytes (got {size})"
        )
