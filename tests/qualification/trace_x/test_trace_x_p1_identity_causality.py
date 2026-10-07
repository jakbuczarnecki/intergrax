# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P1 identity & causality qualification entrypoint (TXP1-Q01..Q30)."""

from __future__ import annotations

import pytest

from tests.qualification.trace_x import _trace_x_p1_qualification_tests as _gates

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

for _name in dir(_gates):
    if _name.startswith("test_"):
        globals()[_name] = getattr(_gates, _name)
