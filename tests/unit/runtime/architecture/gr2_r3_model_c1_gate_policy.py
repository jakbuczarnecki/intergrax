# © Artur Czarnecki. All rights reserved.

"""GR-2-R3 MODEL C1 architecture gate policy (closed-world allowlist)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

PRODUCTION_SCAN_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "intergrax",
    REPO_ROOT / "agents",
    REPO_ROOT / "applications",
    REPO_ROOT / "platform_proofs",
)

# Narrow allowlist: modules certified to bridge intake/runtime internally.
INTERNAL_ROOT_ENGINE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/execution/canonical_intake_adapter.py",
        "intergrax/runtime/execution/facade.py",
        "intergrax/runtime/execution/runtime.py",
        "intergrax/runtime/execution/host_task.py",
        "intergrax/runtime/execution/host_root_execution_intake.py",
        "intergrax/runtime/execution/host_root_launch_evidence.py",
        "intergrax/runtime/execution/compensation_side_effect.py",
        "intergrax/runtime/governance/default_root_execution_launcher.py",
    }
)

# Upstream evidence only — not production root-start authority minting.
AUTHORITY_RESOLUTION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/execution/host_root_launch_evidence.py",
        "intergrax/runtime/execution/orchestration.py",
    }
)

TEST_TREE_PREFIXES: tuple[str, ...] = (
    "tests/",
    "testing_support/",
)
