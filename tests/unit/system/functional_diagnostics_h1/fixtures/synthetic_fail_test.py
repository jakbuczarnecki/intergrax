# © Artur Czarnecki. All rights reserved.

"""Controlled synthetic failing test for H1 fail-fast classification."""

import pytest


def test_h1_synthetic_failure() -> None:
    pytest.fail("h1_synthetic_failure")
