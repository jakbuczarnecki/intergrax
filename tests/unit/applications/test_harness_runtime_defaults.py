# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.runtime_defaults import harness_production_mode

pytestmark = pytest.mark.gate


def test_harness_production_mode_is_false_for_lab() -> None:
    assert harness_production_mode() is False
