# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R2 — GCA startup boundary for execution continuation store qualification."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    default_execution_continuation_state_store,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

pytestmark = [pytest.mark.unit]


class _HarnessStartupCaptured(RuntimeError):
    """Abort factory after the sanctioned harness composition boundary."""


def _durable_continuation_store():
    return execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )


def _minimal_factory_kwargs() -> dict[str, object]:
    return {
        "registry_projection": build_governed_contractor_test_registry_projection(),
        "document_store": InMemoryDocumentStore(),
    }


def test_strict_lightweight_document_store_rejects_missing_continuation_at_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
    ) as harness_build:
        with pytest.raises(ExecutionContinuationError) as exc:
            create_governed_contractor_backend_app(**_minimal_factory_kwargs())
        assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
        harness_build.assert_not_called()


def test_strict_lightweight_document_store_rejects_non_durable_continuation_at_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    lab_store = default_execution_continuation_state_store()
    assert lab_store.is_durable is False
    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
    ) as harness_build:
        with pytest.raises(ExecutionContinuationError) as exc:
            create_governed_contractor_backend_app(
                **_minimal_factory_kwargs(),
                execution_continuation_state_store=lab_store,
            )
        assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
        harness_build.assert_not_called()


def test_strict_lightweight_document_store_accepts_durable_continuation_before_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    durable_store = _durable_continuation_store()
    captured: dict[str, object] = {}

    def _capture_harness_build(*_args, **kwargs):
        captured.update(kwargs)
        raise _HarnessStartupCaptured()

    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
        side_effect=_capture_harness_build,
    ):
        with pytest.raises(_HarnessStartupCaptured):
            create_governed_contractor_backend_app(
                **_minimal_factory_kwargs(),
                execution_continuation_state_store=durable_store,
            )
    assert captured.get("execution_continuation_state_store") is durable_store
