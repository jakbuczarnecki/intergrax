# © Artur Czarnecki. All rights reserved.

"""Reusable expected-fact oracle contract for UE certification proofs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FunctionalExpectation:
    expected_fact: str


@dataclass(frozen=True, slots=True)
class ExpectedFactOracle:
    """Deterministic oracle: final user answer must contain the expected fact."""

    expectation: FunctionalExpectation

    def passes(self, *, answer: str | None) -> bool:
        if not answer:
            return False
        return self.expectation.expected_fact in answer


__all__ = [
    "ExpectedFactOracle",
    "FunctionalExpectation",
]
