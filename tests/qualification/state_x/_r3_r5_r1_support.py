# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R5-R1 shared qualification helpers (checkpoint provider provenance)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

STATE_X_R3_R5_R1_PRE_AUDIT_HEAD: Final = "70ddff7e9d087b79d148be78c22344586af0744f"

STATE_X_R3_R5_R1_ALLOWLIST_PATHS: Final[tuple[str, ...]] = (
    "intergrax/agents/authoring/acp_session_host.py",
    "intergrax/agents/authoring/acp_run.py",
    "intergrax/agents/persistence/session_persistence.py",
    "intergrax/agents/persistence/checkpoint_wiring.py",
    "intergrax/applications/_shared/acp_checkpoint_host_wiring.py",
    "intergrax/applications/_shared/acp_checkpoint_task_enricher.py",
    "intergrax/applications/_shared/acp_session_host_wiring.py",
    "intergrax/runtime/nexus/execution/graph_executor.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
