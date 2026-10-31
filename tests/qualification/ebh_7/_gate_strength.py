# © Artur Czarnecki. All rights reserved.

"""Gate strength taxonomy for EBH-7 qualification tests."""

from __future__ import annotations

from enum import StrEnum


class GateStrength(StrEnum):
    STRUCTURAL = "STRUCTURAL"
    INTEGRITY = "INTEGRITY"
    CONTRACT = "CONTRACT"
    BEHAVIORAL = "BEHAVIORAL"
    E2E = "E2E"
    ADVERSARIAL = "ADVERSARIAL"
    ACCEPTANCE = "ACCEPTANCE"


def may_substitute_for(required: GateStrength, actual: GateStrength) -> bool:
    order = (
        GateStrength.STRUCTURAL,
        GateStrength.INTEGRITY,
        GateStrength.CONTRACT,
        GateStrength.BEHAVIORAL,
        GateStrength.E2E,
        GateStrength.ADVERSARIAL,
        GateStrength.ACCEPTANCE,
    )
    return order.index(actual) >= order.index(required)
