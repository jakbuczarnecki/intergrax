# © Artur Czarnecki. All rights reserved.

"""Architecture gate — DiagnosticReadService must not use full-tenant Problem materialization."""

from __future__ import annotations

import inspect

import pytest

from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService

pytestmark = pytest.mark.unit


def test_diagnostic_read_service_does_not_call_list_for_tenant() -> None:
    source = inspect.getsource(DiagnosticReadService)
    assert "list_for_tenant" not in source
