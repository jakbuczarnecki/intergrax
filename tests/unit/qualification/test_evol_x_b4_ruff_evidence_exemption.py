# © Artur Czarnecki. All rights reserved.

"""EVOL-X B-4: Ruff gates exempt byte-preserved IIAC evidence only (T-IGX-017-A).

The P1 publication needed INTERGRAX_SKIP_PRE_COMMIT_GATE because the changed-path Ruff
gates pass explicit paths, and Ruff ignores ``exclude`` for explicit paths. These tests pin a
narrow, config-driven exemption and prove the published evidence bytes stay unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
CI_DIR = REPO / "scripts/ci"
EVIDENCE_ROOT = "docs/project/maintainers/qualification/iiac_evidence"
AUDIT_DIR = f"{EVIDENCE_ROOT}/EVOL-X/audit"
PROBES = sorted(p.relative_to(REPO).as_posix() for p in (REPO / AUDIT_DIR / "evidence/probes").glob("*.py"))
P1_AUDIT_PUBLICATION_SHA = "4dee0050d1ab2d353c982da23cc82439bd79a53a"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"b4_{name}", CI_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config() -> ModuleType:
    return _load("push_gate_config")


def test_exemption_is_exactly_the_byte_preserved_evidence_root(config: ModuleType) -> None:
    prefixes = config.load_ruff_exempt_prefixes()
    assert prefixes == (EVIDENCE_ROOT,)
    for prefix in prefixes:
        assert prefix == EVIDENCE_ROOT or prefix.startswith(EVIDENCE_ROOT + "/"), prefix


def test_probe_scripts_exist_and_violate_ruff_so_the_exemption_is_required() -> None:
    assert len(PROBES) == 7
    proc = subprocess.run(
        ["uv", "run", "ruff", "format", "--check", *PROBES], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert proc.returncode != 0


def test_split_exempts_evidence_and_keeps_everything_else(config: ModuleType) -> None:
    keep = [
        "scripts/qualification/evol_x_closure_publication_preflight.py",
        "docs/project/maintainers/qualification/iiac_evidence_other/x.py",
        "docs/project/maintainers/qualification/x.py",
        ".github/scripts/x.py",
        "platform_proofs/scenarios/verified_product_identification/scripts/build/build_wdc_dataset.py",
    ]
    lint, exempt = config.split_ruff_exempt([*PROBES, *keep])
    assert exempt == PROBES
    assert lint == keep


def test_glob_entries_are_not_honored_by_the_gate_filter(config: ModuleType, tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.ruff]\nextend-exclude = ["docs/**", "evidence/"]\n', encoding="utf-8")
    assert config.load_ruff_exempt_prefixes(pyproject) == ("evidence",)


def test_changed_paths_gate_drops_only_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("run_ruff_changed_paths")
    preflight = "scripts/qualification/evol_x_closure_publication_preflight.py"
    assert module._python_paths(frozenset({*PROBES, preflight})) == [preflight]


def test_pre_commit_gate_drops_only_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("run_pre_commit_gate")
    preflight = "scripts/qualification/evol_x_closure_publication_preflight.py"
    monkeypatch.setattr(module, "staged_paths", lambda: frozenset({*PROBES, preflight, "README.md"}))
    assert module._staged_python_files() == [preflight]


def test_published_audit_evidence_bytes_are_preserved() -> None:
    index_path = REPO / AUDIT_DIR / "EVIDENCE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for name, meta in index["files"].items():
        raw = (REPO / AUDIT_DIR / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta["sha256"], name
        assert len(raw) == meta["bytes"], name
    cert = json.loads(
        (REPO / EVIDENCE_ROOT / "EVOL-X/certification/EVOL_X_IIAC_CERTIFICATION_RECORD.json").read_text(
            encoding="utf-8"
        )
    )
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == cert["audit_evidence_index_digest"]


def test_probe_scripts_match_p1_published_blobs() -> None:
    for rel in PROBES:
        committed = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "show", f"{P1_AUDIT_PUBLICATION_SHA}:{rel}"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
        assert (REPO / rel).read_bytes() == committed, rel
