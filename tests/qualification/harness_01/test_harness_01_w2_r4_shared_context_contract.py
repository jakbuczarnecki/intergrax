# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R4 — superseded by W2-R5 capability contract tests."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_w2_r4_shared_context_contract_migrated_to_w2_r5() -> None:
    pytest.importorskip("tests.qualification.harness_01.test_harness_01_w2_r5_shared_context_capability")
