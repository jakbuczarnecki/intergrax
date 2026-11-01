# © Artur Czarnecki. All rights reserved.

from pathlib import Path

import pytest
from testing_support.change_safety.data.loaders import load_evol_x_data_document

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_EVOL_X_ROOT = _REPO / "tests" / "qualification" / "evol_x"


def test_evol_x_vl3_manifest_matches_collect_only() -> None:
    manifest = load_evol_x_data_document("evol_vl3_execution_manifest.v1.json")
    pinned = tuple(manifest["pytest_node_ids"])
    assert manifest["profile"] == "evol_x_p3_l3"
    assert manifest["test_execution_baseline_sha"] == "8848857741f4b04e5b476dca3c47f7d0a60bf672"

    import subprocess

    proc = subprocess.run(
        ["uv", "run", "pytest", str(_EVOL_X_ROOT), "--collect-only", "-q"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    collected = tuple(line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("tests/"))
    assert collected == pinned
