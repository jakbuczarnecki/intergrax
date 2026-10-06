# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-P0 baseline qualification entrypoint."""

from __future__ import annotations

import pytest

from tests.qualification.trace_x import test_trace_x_p5_p0_qualification_gates as _gates

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

for _name in dir(_gates):
    if _name.startswith("test_"):
        globals()[_name] = getattr(_gates, _name)
