# © Artur Czarnecki. All rights reserved.

"""Shared active execution identity for external contractor adapter unit tests."""

from __future__ import annotations

import pytest

from external_contractor_adapter.tests.fakes.adapter_test_wiring import (
    bound_external_work_test_execution,
)


@pytest.fixture(autouse=True)
def _bound_external_work_test_execution_identity() -> None:
    with bound_external_work_test_execution():
        yield
