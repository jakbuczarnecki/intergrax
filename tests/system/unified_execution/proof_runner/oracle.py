# © Artur Czarnecki. All rights reserved.

"""Functional oracle for UE-11G-C1 workspace search."""

from __future__ import annotations

from tests.system.unified_execution.proof_runner.contracts import LkwRunResponse
from tests.system.unified_execution.proof_runner.expected_fact_oracle import (
    ExpectedFactOracle,
    FunctionalExpectation,
)

_EXPECTED_FACT = "2026-08-17"
_SEARCH_QUESTION = "When did Incident Orion occur?"

_C1_ORACLE = ExpectedFactOracle(
    expectation=FunctionalExpectation(expected_fact=_EXPECTED_FACT),
)


def search_request_message() -> str:
    return _SEARCH_QUESTION


def expected_fact() -> str:
    return _EXPECTED_FACT


def c1_expected_fact_oracle() -> ExpectedFactOracle:
    return _C1_ORACLE


def functional_oracle_passes(response: LkwRunResponse) -> bool:
    return _C1_ORACLE.passes(answer=response.answer)
