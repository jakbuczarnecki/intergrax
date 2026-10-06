# © Artur Czarnecki. All rights reserved.

"""TRACE-X qualification package hooks."""

import pytest

from tests.qualification.trace_x._trace_x_p3_r1_pass1_session import (
    PASS1_PASSED_NODEIDS,
    pytest_runtest_logreport as _pass1_logreport,
    pytest_sessionfinish as _pass1_sessionfinish,
)
from tests.qualification.trace_x._trace_x_p3_r1_pass2_session import (
    PASS2_PASSED_NODEIDS,
    pytest_runtest_logreport as _pass2_logreport,
    pytest_sessionfinish as _pass2_sessionfinish,
)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _pass1_logreport(report)
    _pass2_logreport(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _pass1_sessionfinish(session, exitstatus)
    _pass2_sessionfinish(session, exitstatus)


__all__ = ["PASS1_PASSED_NODEIDS", "PASS2_PASSED_NODEIDS"]
