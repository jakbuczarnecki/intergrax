# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring import step_loop


@pytest.mark.unit
@pytest.mark.gate
def test_advance_step_is_glue_only() -> None:
    assert step_loop.advance_step_is_glue_only() is True
