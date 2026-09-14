# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — shared deterministic abuse-case fixtures (test-only)."""

from __future__ import annotations

from pathlib import Path

FORGED_ID_SAMPLES: tuple[str, ...] = (
    "not-a-canonical-id",
    "run_short",
    "exec_" + "g" * 31,
    "attempt_" + "X" * 32,
    "task_",
    "",
    "   ",
)

CANONICAL_FORMAT_SUFFIX: str = "a" * 32

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
