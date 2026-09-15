# © Artur Czarnecki. All rights reserved.

"""GR-3 inner enforcement architecture gate policy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

PRODUCTION_SCAN_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "intergrax",
    REPO_ROOT / "agents",
    REPO_ROOT / "applications",
    REPO_ROOT / "platform_proofs",
)

# Definition site + sole certified production caller (External Work adapter).
AUTHORIZE_AND_EXECUTE_CALL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/policy/meaningful_side_effect_authorization.py",
        "agents/external_contractor_adapter/external_work_adapter.py",
    }
)

TEST_TREE_PREFIXES: tuple[str, ...] = (
    "tests/",
    "testing_support/",
)

MEANINGFUL_SIDE_EFFECT_POLICY_BOUNDARY_REL: str = (
    "intergrax/runtime/policy/meaningful_side_effect_authorization.py"
)
