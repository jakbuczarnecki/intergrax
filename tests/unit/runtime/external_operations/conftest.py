# © Artur Czarnecki. All rights reserved.

"""Shared isolation for external-operation qualification (global composition reset)."""

from __future__ import annotations

import pytest

from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    reset_worker_execution_ownership_revocation_trust_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_tool_life_global_composition() -> None:
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()
