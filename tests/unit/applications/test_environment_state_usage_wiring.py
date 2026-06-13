# © Artur Czarnecki. All rights reserved.

"""APP-PROD-6 — environment state usage lint wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.environment_state_usage_wiring import (
    check_environment_state_usage,
    check_harness_environment_state_wiring,
    check_no_raw_app_env_state_access,
    check_on_hook_typed_state_usage,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_check_environment_state_usage_passes_on_repo() -> None:
    violations = check_environment_state_usage(REPO_ROOT)
    assert violations == []


def test_harness_must_wire_environment_state_middleware(tmp_path: Path) -> None:
    harness = tmp_path / "harness_host_runtime.py"
    harness.write_text("def build_harness_host_runtime():\n    pass\n", encoding="utf-8")
    violations = check_harness_environment_state_wiring(harness)
    assert violations
    harness.write_text(
        "def build_harness_host_runtime():\n"
        "    apply_application_environment_state_wiring(nexus)\n",
        encoding="utf-8",
    )
    assert check_harness_environment_state_wiring(harness) == []


def test_forbids_raw_runtime_state_access(tmp_path: Path) -> None:
    module = tmp_path / "bad_host.py"
    module.write_text(
        'def read(ctx):\n    return ctx.runtime_state["app_env_state.v1"]\n',
        encoding="utf-8",
    )
    violations = check_no_raw_app_env_state_access(module, repo_root=tmp_path)
    assert violations
    assert 'runtime_state["app_env_state' in violations[0]


def test_on_hook_requires_typed_helpers(tmp_path: Path) -> None:
    module = tmp_path / "bad_on_hook.py"
    module.write_text(
        "class Host:\n"
        "    def on_hook(self, point, context):\n"
        '        return context.runtime_state.get("app_env_state.v1")\n',
        encoding="utf-8",
    )
    violations = check_on_hook_typed_state_usage(module, repo_root=tmp_path)
    assert violations

    module.write_text(
        "class Host:\n"
        "    def on_hook(self, point, context):\n"
        "        state = ApplicationEnvironmentState.from_runtime_state(context.runtime_state)\n"
        "        return state\n",
        encoding="utf-8",
    )
    assert check_on_hook_typed_state_usage(module, repo_root=tmp_path) == []
