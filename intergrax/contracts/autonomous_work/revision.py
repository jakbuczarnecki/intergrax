# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Revision and definition-version semantics for Autonomous Work (AW-1A)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.autonomous_work._validation import require_non_negative_int


def validate_revision(value: object) -> int:
    if isinstance(value, Revision):
        return value.value
    return require_non_negative_int(value, label="Revision")


def validate_definition_revision(value: object) -> int:
    if isinstance(value, DefinitionRevision):
        return value.value
    return require_non_negative_int(value, label="DefinitionRevision")


@dataclass(frozen=True, slots=True)
class Revision:
    """Optimistic-concurrency revision for durable worker entities."""

    value: int

    def __post_init__(self) -> None:
        validate_revision(self.value)

    def __lt__(self, other: object) -> bool:
        if type(other) is not Revision:
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: object) -> bool:
        if type(other) is not Revision:
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: object) -> bool:
        if type(other) is not Revision:
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: object) -> bool:
        if type(other) is not Revision:
            return NotImplemented
        return self.value >= other.value


@dataclass(frozen=True, slots=True)
class DefinitionRevision:
    """Published WorkerDefinition revision — stable identity with versioned content."""

    value: int

    def __post_init__(self) -> None:
        validate_definition_revision(self.value)

    def __lt__(self, other: object) -> bool:
        if type(other) is not DefinitionRevision:
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: object) -> bool:
        if type(other) is not DefinitionRevision:
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: object) -> bool:
        if type(other) is not DefinitionRevision:
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: object) -> bool:
        if type(other) is not DefinitionRevision:
            return NotImplemented
        return self.value >= other.value


def initial_revision() -> Revision:
    return Revision(0)


def initial_definition_revision() -> DefinitionRevision:
    return DefinitionRevision(0)
