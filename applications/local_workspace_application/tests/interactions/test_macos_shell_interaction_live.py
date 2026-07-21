# © Artur Czarnecki. All rights reserved.

"""LKW — real macOS shell interaction adapter live proof."""

from __future__ import annotations

import platform
from collections.abc import Callable
from pathlib import Path

import pytest

from local_workspace_application.tests.interactions.os_interaction_live_helpers import (
    run_os_interaction_live_proof,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="macOS shell interaction proof requires macOS",
    ),
]


def test_macos_shell_adapter_executes_real_lkw_interactions(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    run_os_interaction_live_proof(
        os_family="macos",
        tmp_path=tmp_path,
        record_property=record_property,
    )
