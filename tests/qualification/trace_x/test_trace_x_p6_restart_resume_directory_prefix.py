# © Artur Czarnecki. All rights reserved.

"""T-IGX-077-C: directory prefix alone is not canonical resume."""

from __future__ import annotations

import pytest
from tests.qualification.trace_x._trace_x_p6_discovery import discover_restart_resume_path_keys
from tests.qualification.trace_x._trace_x_p6_restart_resume_classification import (
    classify_restart_resume_path,
)
from tests.qualification.trace_x._trace_x_p6_types import RestartResumePathClass

pytestmark = [pytest.mark.qualification]

_INVENTORIED = frozenset(path for path, _ in discover_restart_resume_path_keys())
_EXECUTION_INVENTORY_EXAMPLE = "intergrax/runtime/execution/boundary.py"


def test_txp6_directory_prefix_requires_inventory_for_canonical_resume() -> None:
    """Uninventoried paths under a resume directory prefix are G_UNCLEAR; inventoried paths stay A."""
    assert (
        classify_restart_resume_path(
            "intergrax/runtime/execution/__not_in_inventory__.py",
        )
        is RestartResumePathClass.G_UNCLEAR
    )
    assert _EXECUTION_INVENTORY_EXAMPLE in _INVENTORIED
    assert classify_restart_resume_path(_EXECUTION_INVENTORY_EXAMPLE) is RestartResumePathClass.A_CANONICAL_RESUME
