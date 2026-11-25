# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) SBX-1..SBX-5: sandbox composition fails closed in PRODUCT/STRICT postures."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.sandbox_wiring import (
    SandboxCompositionError,
    product_requires_sandbox,
    sandbox_required_for_posture,
    wire_sandbox_sessions,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    SandboxProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.sandbox.manager import ENV_SANDBOX_ROOT

_TOOLS = ["sandbox.exec", "workspace.write_file"]


def _product(*, tool_ids: list[str] | None, sandbox: SandboxProfile | None) -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.product_defaults(tool_ids=tool_ids)
    return base.model_copy(update={"sandbox": sandbox})


def _lab_strict(*, sandbox: SandboxProfile | None) -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.product_defaults(tool_ids=_TOOLS)
    meta = base.meta.model_copy(
        update={
            "application_profile": ApplicationProfile.LAB,
            "execution_mode": ExecutionMode.STRICT,
        },
    )
    return base.model_copy(update={"meta": meta, "sandbox": sandbox})


@pytest.fixture(autouse=True)
def _no_sandbox_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_SANDBOX_ROOT, raising=False)


def test_product_side_effect_tools_without_sandbox_profile_raises() -> None:
    env = _product(tool_ids=["sandbox.exec"], sandbox=None)
    with pytest.raises(SandboxCompositionError, match="no sandbox profile"):
        wire_sandbox_sessions(env)


def test_product_side_effect_tools_without_explicit_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = _product(tool_ids=_TOOLS, sandbox=SandboxProfile(root=None, enable_exec_tool=True))
    with pytest.raises(SandboxCompositionError, match="explicit root"):
        wire_sandbox_sessions(env)
    assert not (tmp_path / "build").exists()


def test_product_side_effect_tools_with_explicit_root_builds_manager(tmp_path: Path) -> None:
    root = tmp_path / "sbx"
    env = _product(tool_ids=_TOOLS, sandbox=SandboxProfile(root=root, enable_exec_tool=True))
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == root


def test_product_side_effect_tools_with_env_root_builds_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_SANDBOX_ROOT, str(tmp_path / "from-env"))
    env = _product(tool_ids=_TOOLS, sandbox=SandboxProfile(root=None, enable_exec_tool=True))
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == tmp_path / "from-env"


def test_product_without_side_effect_tools_does_not_require_sandbox() -> None:
    env = _product(tool_ids=None, sandbox=None)
    assert not product_requires_sandbox(env)
    assert not sandbox_required_for_posture(env)
    assert wire_sandbox_sessions(env) is None


def test_strict_lab_side_effect_tools_require_explicit_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = _lab_strict(sandbox=SandboxProfile(root=None, enable_exec_tool=True))
    assert env.application_profile is ApplicationProfile.LAB
    assert sandbox_required_for_posture(env)
    with pytest.raises(SandboxCompositionError):
        wire_sandbox_sessions(env)
    assert not (tmp_path / "build").exists()


def test_lab_balanced_side_effect_tools_keep_default_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = ApplicationEnvironmentProfile.lab_defaults()
    env = base.model_copy(update={"sandbox": SandboxProfile(root=None, enable_exec_tool=True)})
    assert not sandbox_required_for_posture(env)
    manager = wire_sandbox_sessions(env)
    assert manager is not None
