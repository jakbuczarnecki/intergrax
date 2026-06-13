# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.health.category_probes import health_check_codecraft
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-1",
        task_id="task-1",
        allowed_operations=frozenset({"run_python", "write_file"}),
    )


def test_health_check_codecraft_disabled_mode() -> None:
    profile = CodeCraftProfile(mode="disabled")
    out = health_check_codecraft(ToolWiringContext(extras={"codecraft_profile": profile}), object())
    assert out.status.healthy is True
    assert out.status.detail == "mode_disabled"


def test_health_check_codecraft_requires_sandbox(sandbox_session: SandboxSession) -> None:
    profile = CodeCraftProfile(mode="autonomous")
    out = health_check_codecraft(
        ToolWiringContext(extras={"codecraft_profile": profile}),
        object(),
    )
    assert out.status.healthy is False

    ready = health_check_codecraft(
        ToolWiringContext(sandbox_session=sandbox_session, extras={"codecraft_profile": profile}),
        object(),
    )
    assert ready.status.healthy is True
