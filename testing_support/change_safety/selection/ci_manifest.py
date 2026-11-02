# © Artur Czarnecki. All rights reserved.

"""W2-09 / W3-05: EVOL VL3 execution manifest (T3) collect + integrity helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.data.loaders import load_evol_x_data_document

_REPO = Path(__file__).resolve().parents[3]
EVOL_X_TEST_ROOT = "tests/qualification/evol_x"
VL3_MANIFEST_FILE = "evol_vl3_execution_manifest.v1.json"
VL3_COLLECT_COMMAND = ("pytest", EVOL_X_TEST_ROOT, "--collect-only", "-q")
EXCLUDED_POPULATIONS = ("DIM-B-PROD-Q-LIVE",)
EXCLUSIONS = ("tests/unit/agents/test_reference_harness_strict.py::RC-7_BASELINE_DEFECT",)


def collect_evol_x_node_ids() -> tuple[str, ...]:
    """Run ``pytest --collect-only`` for the EVOL-X tree (same interpreter) and return ordered node ids."""
    proc = subprocess.run(
        [sys.executable, "-m", *VL3_COLLECT_COMMAND],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"collect-only failed (rc={proc.returncode}): {proc.stdout[-500:]} {proc.stderr[-500:]}")
    return tuple(
        line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("tests/") and "::" in line
    )


def manifest_content_digest(node_ids: tuple[str, ...] | list[str]) -> str:
    return stable_digest_hex(list(node_ids))


def build_manifest_document(node_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": "evol_vl3_execution_manifest.v1",
        "source_authority": "docs/project/maintainers/qualification/EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md#II-A-T3",
        "profile": "evol_x_p3_l3",
        "uv_extra": "dev-ci",
        "collect_command": " ".join(("uv run", *VL3_COLLECT_COMMAND)),
        "excluded_populations": list(EXCLUDED_POPULATIONS),
        "exclusions": list(EXCLUSIONS),
        "manifest_content_digest": manifest_content_digest(node_ids),
        "pytest_node_ids": list(node_ids),
    }


def load_pinned_manifest() -> dict[str, Any]:
    return load_evol_x_data_document(VL3_MANIFEST_FILE)


def verify_manifest_integrity(manifest: dict[str, Any], collected: tuple[str, ...]) -> list[str]:
    """Return problems (empty == OK): digest self-consistency, missing/extra nodes vs collect-only."""
    problems: list[str] = []
    pinned = tuple(manifest["pytest_node_ids"])
    if manifest.get("manifest_content_digest") != manifest_content_digest(pinned):
        problems.append("manifest_content_digest mismatch")
    missing = sorted(set(pinned) - set(collected))
    extra = sorted(set(collected) - set(pinned))
    if missing:
        problems.append(f"missing_from_collect={len(missing)}")
    if extra:
        problems.append(f"unpinned_collected={len(extra)}")
    if not missing and not extra and pinned != collected:
        problems.append("order mismatch")
    return problems
