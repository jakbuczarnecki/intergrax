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

# Certified internal engine bridges (construction, imports, root execute surfaces).
INTERNAL_ROOT_ENGINE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/execution/__init__.py",  # package re-export surface only
        "intergrax/runtime/execution/canonical_intake_adapter.py",
        "intergrax/runtime/execution/facade.py",
        "intergrax/runtime/execution/runtime.py",
        "intergrax/runtime/execution/host_task.py",
        "intergrax/runtime/execution/host_root_execution_intake.py",
        "intergrax/runtime/execution/host_root_launch_evidence.py",
        "intergrax/runtime/execution/compensation_side_effect.py",
        "intergrax/runtime/governance/default_root_execution_launcher.py",
        "intergrax/runtime/execution/orchestration.py",  # execute_root_task: INTERNAL CERTIFIED HARNESS ENTRY
    }
)

# Upstream evidence / host composition — not production root-start authority minting.
AUTHORITY_RESOLUTION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/execution/host_root_launch_evidence.py",
        "intergrax/runtime/execution/host_task.py",
        "intergrax/runtime/execution/orchestration.py",  # certified harness resume path
    }
)

# Only these modules may import execute_root_task (harness scheduler bridge).
LEGACY_EXECUTE_ROOT_TASK_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/task/unified_task_runner.py",
    }
)

# Harness / scheduler / eval orchestration — not Tier-3 production host root entry.
LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/task/unified_task_runner.py",
        "intergrax/runtime/task/__init__.py",
        "intergrax/runtime/long_running/wiring.py",
        "intergrax/runtime/long_running/scheduler.py",
        "intergrax/applications/_shared/harness_task_routes.py",
        "intergrax/applications/_shared/task_control_wiring.py",
        "intergrax/applications/_shared/task_control.py",
        "intergrax/applications/_shared/async_task_dispatch.py",
        "intergrax/applications/_shared/async_task_index_protocol.py",
        "intergrax/applications/_shared/sqlite_async_task_index.py",
        "intergrax/eval/nexus_eval_runner.py",
        "intergrax/experiments/workflow.py",
    }
)

TEST_TREE_PREFIXES: tuple[str, ...] = (
    "tests/",
    "testing_support/",
)
