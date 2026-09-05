# © Artur Czarnecki. All rights reserved.

"""P2-003-D2-V2: event_kind validation ownership and registry import boundary."""

from __future__ import annotations

import importlib
import inspect

import pytest

from intergrax.runtime.events.event_kind import DomainSignalError, validate_event_kind
from intergrax.runtime.events.event_kind_registry import EventKindRegistryError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "kind",
    (
        "agents.legal.clause_flagged",
        "applications.foo.bar",
    ),
)
def test_validate_event_kind_accepts_valid_names(kind: str) -> None:
    validate_event_kind(kind)


@pytest.mark.parametrize(
    "kind",
    (
        "Agent.Bad",
        "foo",
        "agents.foo.Bad",
    ),
)
def test_validate_event_kind_rejects_invalid_names(kind: str) -> None:
    with pytest.raises(DomainSignalError):
        validate_event_kind(kind)


def test_signals_reexports_event_kind_contract() -> None:
    from intergrax.runtime.events import signals

    assert signals.DomainSignalError is DomainSignalError
    assert signals.validate_event_kind is validate_event_kind


def test_event_kind_registry_does_not_import_signals_module() -> None:
    registry = importlib.import_module("intergrax.runtime.events.event_kind_registry")
    source = inspect.getsource(registry)
    assert "from intergrax.runtime.events.signals import" not in source
    assert "import intergrax.runtime.events.signals" not in source


def test_event_kind_registry_error_extends_domain_signal_error() -> None:
    assert issubclass(EventKindRegistryError, DomainSignalError)
