# © Artur Czarnecki. All rights reserved.

"""Canonical reproduction metadata for Scenario #1."""

from __future__ import annotations

PROOF_ID = "SCENARIO-AI-INCIDENT-INVESTIGATION-SKELETON"

CANONICAL_REPRODUCTION_PROFILE = "quick"

CANONICAL_REPRODUCTION_PREREQUISITES: tuple[str, ...] = (
    "Python 3.12",
    "uv",
    "uv sync (from repository root)",
)


def canonical_reproduction_argv() -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "python",
        "scripts/proof/run-intergrax-proof-suite.py",
        "--profile",
        CANONICAL_REPRODUCTION_PROFILE,
        "--proof-id",
        PROOF_ID,
    )


def canonical_reproduction_shell_command() -> str:
    return " ".join(canonical_reproduction_argv())


__all__ = [
    "CANONICAL_REPRODUCTION_PREREQUISITES",
    "CANONICAL_REPRODUCTION_PROFILE",
    "PROOF_ID",
    "canonical_reproduction_argv",
    "canonical_reproduction_shell_command",
]
