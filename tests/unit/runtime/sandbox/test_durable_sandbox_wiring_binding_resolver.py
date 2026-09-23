# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.sandbox.durable_sandbox_wiring_binding_resolver import (
    SandboxSessionManagerDurableWiringBindingResolver,
)
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.tools.durable_invocation_wiring_binding_resolver import (
    DurableToolInvocationWiringBindingResolutionError,
)
from intergrax.tools.invocation_wiring import FixedSandboxSessionWiringResolver

pytestmark = pytest.mark.unit


def test_resolve_rehydrates_session_from_disk_without_live_handle(
    tmp_path: Path,
) -> None:
    manager_a = SandboxSessionManager(root=tmp_path)
    session = manager_a.open_or_create(tenant_id="t1", task_id="task1")
    session_id = session.session_id
    del manager_a

    manager_b = SandboxSessionManager(root=tmp_path)
    resolver = SandboxSessionManagerDurableWiringBindingResolver(
        sandbox_manager=manager_b,
    )
    wiring = resolver.resolve_fixed_sandbox_session_wiring(
        sandbox_session_id=session_id,
        tenant_id="t1",
        task_id="task1",
    )
    assert isinstance(wiring, FixedSandboxSessionWiringResolver)
    assert wiring.sandbox_session.session_id == session_id


def test_resolve_fail_closed_on_identity_mismatch(tmp_path: Path) -> None:
    manager = SandboxSessionManager(root=tmp_path)
    session = manager.open_or_create(tenant_id="t1", task_id="task1")
    resolver = SandboxSessionManagerDurableWiringBindingResolver(sandbox_manager=manager)
    with pytest.raises(DurableToolInvocationWiringBindingResolutionError) as exc:
        resolver.resolve_fixed_sandbox_session_wiring(
            sandbox_session_id=session.session_id,
            tenant_id="t1",
            task_id="other-task",
        )
    assert exc.value.code == "sandbox_session_unavailable"
