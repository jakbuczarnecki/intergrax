"""Application-layer failure model — no ERL or reconciliation failures at this stage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ApplicationFailureCode(StrEnum):
    INVALID_SCENARIO_CONTEXT = "invalid_scenario_context"
    MISSING_BUSINESS_ENTITY = "missing_business_entity"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"


@dataclass(frozen=True, slots=True)
class ApplicationFailure:
    code: ApplicationFailureCode
    message: str


class ApplicationError(Exception):
    """Base for scenario application failures."""

    def __init__(self, failure: ApplicationFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class InvalidScenarioContextError(ApplicationError):
    """Raised when execution context fails validation."""


class MissingBusinessEntityError(ApplicationError):
    """Raised when a required order or payment entity cannot be loaded."""


class DependencyUnavailableError(ApplicationError):
    """Raised when a required port implementation is unavailable."""
