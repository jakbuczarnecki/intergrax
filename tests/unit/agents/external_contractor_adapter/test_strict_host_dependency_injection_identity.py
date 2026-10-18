# © Artur Czarnecki. All rights reserved.

"""AC3 same-instance strict host injection (R3-F01 identity correction)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from echo.echo_agent import EchoAgent
from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from external_contractor_adapter.strict_host_dependency_injection import (
    StrictExternalWorkHostAgentInjectionError,
    materialize_with_strict_external_work_host_dependencies,
)
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _host_kwargs() -> dict:
    return {
        "external_work": MagicMock(spec=ExternalWorkIntegration),
        "authorization_boundary": None,
        "sanctioned_operation_port": MagicMock(),
        "require_sanctioned_external_work_mutations": True,
    }


def test_materialize_returns_same_instance_as_authoritative_factory() -> None:
    authoritative = ExternalContractorAdapterAgent()
    materialized = materialize_with_strict_external_work_host_dependencies(
        authoritative,
        **_host_kwargs(),
    )
    assert materialized is authoritative


def test_materialize_preserves_subclass_identity_and_factory_state() -> None:
    class _MarkedAdapter(ExternalContractorAdapterAgent):
        marker = "artifact-factory-state"

    authoritative = _MarkedAdapter()
    authoritative.marker = "set-by-artifact-factory"
    materialized = materialize_with_strict_external_work_host_dependencies(
        authoritative,
        **_host_kwargs(),
    )
    assert materialized is authoritative
    assert materialized.marker == "set-by-artifact-factory"
    assert type(materialized) is _MarkedAdapter


def test_materialize_does_not_construct_second_agent() -> None:
    construction_count = 0

    class _CountingAdapter(ExternalContractorAdapterAgent):
        def __init__(self) -> None:
            nonlocal construction_count
            construction_count += 1
            super().__init__()

    authoritative = _CountingAdapter()
    assert construction_count == 1
    materialize_with_strict_external_work_host_dependencies(
        authoritative,
        **_host_kwargs(),
    )
    assert construction_count == 1


def test_wrong_agent_type_rejected_without_mutation() -> None:
    echo = EchoAgent()
    with pytest.raises(StrictExternalWorkHostAgentInjectionError):
        materialize_with_strict_external_work_host_dependencies(echo, **_host_kwargs())


def test_require_sanctioned_mutations_false_rejected_without_mutation() -> None:
    authoritative = ExternalContractorAdapterAgent()
    kwargs = _host_kwargs()
    kwargs["require_sanctioned_external_work_mutations"] = False
    with pytest.raises(StrictExternalWorkHostAgentInjectionError):
        materialize_with_strict_external_work_host_dependencies(authoritative, **kwargs)
    assert authoritative._external_work is None
    assert authoritative._sanctioned_operation_port is None


def test_negative_identity_fails_if_reconstruction_reintroduced() -> None:
    """Fails when injection returns a newly constructed adapter instead of the input."""
    authoritative = ExternalContractorAdapterAgent()
    materialized = materialize_with_strict_external_work_host_dependencies(
        authoritative,
        **_host_kwargs(),
    )
    assert id(materialized) == id(authoritative)
