# © Artur Czarnecki. All rights reserved.

"""LKW — real Linux shell interaction adapter live proof."""

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
        platform.system() != "Linux",
        reason="Linux shell interaction proof requires Linux",
    ),
]


def test_linux_shell_adapter_executes_real_lkw_interactions(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    run_os_interaction_live_proof(
        os_family="linux",
        tmp_path=tmp_path,
        record_property=record_property,
    )
