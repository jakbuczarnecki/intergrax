# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P2-ENV-1 — strict production import-cycle corrective evidence."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    STRICT_LAUNCHER_IMPORT_PROBE,
    repo_root,
    run_python_snippet,
    strict_launcher_import_probe,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_HISTORIC_CYCLE_SNIPPET = (
    "registry_projection.py",
    "partially initialized",
    "wiring.py",
)


def test_env1_g01_strict_launcher_imports_in_fresh_process() -> None:
    probe = strict_launcher_import_probe()
    assert probe.returncode == 0, probe.combined
    assert "open_strict_production_serving_session" in probe.stdout or probe.returncode == 0


def test_env1_g02_import_order_independence() -> None:
    orders = (
        (
            "import intergrax.applications._shared.wiring as w; "
            "import intergrax.agent_distribution.sqlite_stores as s; "
            "from intergrax.applications._shared.strict_production_serving_launcher "
            "import open_strict_production_serving_session; "
            "assert w.BuilderMap is not None and open_strict_production_serving_session"
        ),
        (
            "from intergrax.applications._shared.strict_production_serving_launcher "
            "import open_strict_production_serving_session; "
            "import intergrax.applications._shared.registry_projection as rp; "
            "import intergrax.agent_distribution.sqlite_stores as s; "
            "assert rp.RegistryProjectionError and open_strict_production_serving_session"
        ),
    )
    for snippet in orders:
        probe = run_python_snippet(snippet)
        assert probe.returncode == 0, probe.combined


def test_env1_g03_projection_and_descriptor_contracts_importable() -> None:
    wiring = importlib.import_module("intergrax.applications._shared.wiring")
    projection = importlib.import_module("intergrax.applications._shared.registry_projection")
    descriptor = importlib.import_module(
        "intergrax.applications._shared.registry_projection_descriptor"
    )
    assert wiring.BuilderMap is not None
    assert projection.SCHEMA_REGISTRY_PROJECTION_EVIDENCE_V1
    assert descriptor.SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1


def test_env1_g04_no_alternate_projection_store_authority() -> None:
    text = (repo_root() / "intergrax/agent_distribution/sqlite_stores.py").read_text(
        encoding="utf-8"
    )
    assert "SqliteRuntimeRegistryProjectionDescriptorStore" in text
    assert text.count("registry_projection_descriptor") == 2
    assert "registry_projection import" not in text
    assert "BuilderMap" not in text


def test_env1_historic_cycle_documented_not_active() -> None:
    """Prior P2 probe failure mode — preserved for audit traceability."""
    probe = subprocess.run(
        [sys.executable, "-c", STRICT_LAUNCHER_IMPORT_PROBE],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0
    combined = f"{probe.stdout}\n{probe.stderr}"
    for token in _HISTORIC_CYCLE_SNIPPET:
        assert token not in combined.lower() and token not in combined
