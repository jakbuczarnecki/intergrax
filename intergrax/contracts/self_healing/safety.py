# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing safety invariants — decisions never self-execute (SELF-HEALING R1)."""

from __future__ import annotations

from typing import Iterable

from intergrax.contracts.external_operations.safety import assert_no_secrets_in_audit_payload


def assert_decision_has_no_execution_surface(decision: object) -> None:
    if hasattr(decision, "execute"):
        raise TypeError("SelfHealingDecision must not expose execute()")


def assert_strategy_has_no_execution_surface(strategy: object) -> None:
    if hasattr(strategy, "execute"):
        raise TypeError("SelfHealingStrategy must not expose execute()")


def assert_no_secrets_in_self_healing_audit(parts: Iterable[str]) -> None:
    assert_no_secrets_in_audit_payload(parts)


__all__ = [
    "assert_decision_has_no_execution_surface",
    "assert_no_secrets_in_self_healing_audit",
    "assert_strategy_has_no_execution_surface",
]
