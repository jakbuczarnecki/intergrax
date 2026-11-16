# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for the ACP authoring unit tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from intergrax.runtime.governance import active_execution_governance_identity as _identity
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
)


@pytest.fixture
def bound_governance_identity() -> Iterator[Callable[..., None]]:
    """Bind an active execution governance identity for one test, reset afterwards.

    PRE_MODEL governance (GR-10) requires an admitted identity for every ACP run. Tests that
    drive a run call the returned function with the tenant and principal of their request.
    The fixture is deliberately not autouse: a test passes only because it asked for the
    binding, and the unbound path stays covered by
    ``tests/qualification/pdc_x/d1_repair/test_d1_repair_acp_identity_gate.py``.
    """

    def _bind(*, tenant_id: str, principal_id: str, workspace_id: str = "ws-acp-test") -> None:
        bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                principal_id=principal_id,
            )
        )

    yield _bind
    # An async test body runs in a copy of the context, so a token from this sync fixture cannot
    # be reset from here; clear the binding of the fixture's own context instead.
    _identity._active_execution_governance_identity.set(None)
