# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R5-R1-R1 — ACP host-context checkpoint provenance closure."""

from __future__ import annotations

from pathlib import Path
from typing import Final

STATE_X_R3_R5_R1_R1_PRE_AUDIT_HEAD: Final = "941f2c3bfab0e546031b7ad033efddfc69da41f5"

STATE_X_R3_R5_R1_R1_ALLOWLIST_PATHS: Final[tuple[str, ...]] = (
    "intergrax/agents/authoring/acp_session_host.py",
    "intergrax/agents/authoring/acp_run.py",
    "intergrax/agents/persistence/session_persistence.py",
    "intergrax/agents/persistence/checkpoint_wiring.py",
    "intergrax/runtime/nexus/agents/agent_engine.py",
    "intergrax/runtime/nexus/execution/graph_executor.py",
    "intergrax/runtime/nexus/nexus_loop.py",
    "intergrax/applications/_shared/acp_checkpoint_host_wiring.py",
    "testing_support/acp_checkpoint_test_wiring.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
