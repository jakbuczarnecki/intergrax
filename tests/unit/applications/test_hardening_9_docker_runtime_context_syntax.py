# © Artur Czarnecki. All rights reserved.

"""HARDENING-9 — Docker runtime-context must not block canonical application syntax gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.application_build_context import (
    validate_canonical_application_python_syntax,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]


def test_canonical_application_python_syntax_excludes_runtime_context() -> None:
    """Stale materialized ``docker/runtime-context`` trees are not canonical sources."""
    validate_canonical_application_python_syntax(REPO)
