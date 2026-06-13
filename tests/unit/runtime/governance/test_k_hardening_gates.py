# © Artur Czarnecki. All rights reserved.

import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.runtime.policy.policy_engine import (
    PolicyEngine,
    coerce_replay_policy_engine,
)
from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig

pytestmark = [pytest.mark.unit, pytest.mark.no_ci]


@pytest.mark.unit
@pytest.mark.gate
def test_coerce_replay_policy_engine_wraps_execution_engine():
    config = ExecutionPolicyConfig()
    from intergrax.runtime.replay.policy import ExecutionPolicyEngine

    legacy = ExecutionPolicyEngine(config)
    facade = coerce_replay_policy_engine(legacy)
    assert isinstance(facade, PolicyEngine)
    assert facade.replay is legacy


@pytest.mark.unit
@pytest.mark.gate
def test_coerce_replay_policy_engine_requires_replay_on_facade():
    with pytest.raises(ValueError, match="replay configuration"):
        coerce_replay_policy_engine(PolicyEngine())


@pytest.mark.unit
@pytest.mark.gate
def test_production_paths_do_not_import_chat_agent() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts" / "check_production_chat_agent_imports.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.unit
@pytest.mark.gate
def test_agents_vendor_import_audit() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts" / "check_agents_vendor_imports.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
