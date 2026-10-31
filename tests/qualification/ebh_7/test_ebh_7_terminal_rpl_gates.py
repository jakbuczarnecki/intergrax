# © Artur Czarnecki. All rights reserved.

"""EBH-7 terminal INTERNAL / FIXED RPL qualification gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._fixed_substrate_rpl_proofs import (
    prove_fixed_integration_resolve_fail_closed,
    prove_fixed_integration_vendor_fail_closed,
    prove_memory_sqlite_conversational_fixed_negative,
    prove_memory_sqlite_conversational_fixed_positive,
)
from tests.qualification.ebh_5._internal_rpl_behavioral_proofs import (
    prove_application_composition_root_internal_negative,
    prove_application_composition_root_internal_positive,
    prove_prodq_colocated_adapter_internal_negative,
    prove_prodq_colocated_adapter_internal_positive,
)
from tests.qualification.ebh_5._terminal_rpl_normative_nodes import FIXED_RPL_CONCERNS, INTERNAL_RPL_CONCERNS
from tests.qualification.ebh_7._ebh7_terminal_operator_authorization import assert_ebh7_terminal_authorization_integrity

pytestmark = [pytest.mark.unit, pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_terminal_operator_authorization_ssot() -> None:
    assert_ebh7_terminal_authorization_integrity()


@pytest.mark.parametrize("concern_id", sorted(INTERNAL_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_internal_rpl_positive(concern_id: str) -> None:
    if concern_id == "application.composition_root":
        prove_application_composition_root_internal_positive()
        return
    if concern_id == "prodq.colocated_runtime_adapter_isinstance":
        prove_prodq_colocated_adapter_internal_positive()
        return
    raise AssertionError(concern_id)


@pytest.mark.parametrize("concern_id", sorted(INTERNAL_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_internal_rpl_negative(concern_id: str) -> None:
    if concern_id == "application.composition_root":
        prove_application_composition_root_internal_negative()
        return
    if concern_id == "prodq.colocated_runtime_adapter_isinstance":
        prove_prodq_colocated_adapter_internal_negative()
        return
    raise AssertionError(concern_id)


@pytest.mark.parametrize("concern_id", sorted(FIXED_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_fixed_substrate_rpl_positive(concern_id: str, tmp_path: Path) -> None:
    if concern_id.startswith("integration."):
        prove_fixed_integration_vendor_fail_closed(concern_id)
        return
    if concern_id == "memory.sqlite.conversational":
        prove_memory_sqlite_conversational_fixed_positive(tmp_path)
        return
    raise AssertionError(concern_id)


@pytest.mark.parametrize("concern_id", sorted(FIXED_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_fixed_substrate_rpl_negative(concern_id: str) -> None:
    if concern_id.startswith("integration."):
        prove_fixed_integration_resolve_fail_closed(concern_id)
        return
    if concern_id == "memory.sqlite.conversational":
        prove_memory_sqlite_conversational_fixed_negative()
        return
    raise AssertionError(concern_id)
