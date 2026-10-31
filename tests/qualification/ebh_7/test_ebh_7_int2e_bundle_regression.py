# © Artur Czarnecki. All rights reserved.

"""EBH-7 Wave 1 — INT2E integration bundle regression @ gap-closure production delta."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_GAP_START = "ad3e24a20d7d1b341f721ef2427b9c1f65506061"
_REPO = Path(__file__).resolve().parents[3]


def _changed_bundle_modules() -> tuple[str, ...]:
    import subprocess

    out = subprocess.check_output(
        [
            "git",
            "diff",
            f"{_GAP_START}..HEAD",
            "--name-only",
            "--",
            "intergrax/integrations/providers/",
        ],
        cwd=_REPO,
        text=True,
    )
    modules: list[str] = []
    for line in out.splitlines():
        path = line.strip().replace("/", ".")
        if path.endswith(".bundle.py") or path.endswith("/bundle.py"):
            pass
        if line.strip().endswith("bundle.py"):
            rel = line.strip().replace("\\", "/")
            mod = "intergrax." + rel.removeprefix("intergrax/").removesuffix(".py").replace("/", ".")
            modules.append(mod)
    return tuple(sorted(set(modules)))


@pytest.mark.parametrize("module_name", _changed_bundle_modules())
def test_ebh7_int2e_bundle_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_ebh7_int2e_sqlite_relational_resolve_from_profile(tmp_path: Path) -> None:
    from intergrax.integrations.contracts.base import IntegrationCategory
    from tests.qualification.ebh_5._package_c_v2_primitives import run_dual_slug_resolve_from_profile

    run_dual_slug_resolve_from_profile(
        category=IntegrationCategory.RELATIONAL_STORE,
        slug_a="sqlite",
        slug_b="duckdb",
        tmp_path=tmp_path,
    )


def test_ebh7_int2e_bundle_regression_scope_non_empty() -> None:
    assert len(_changed_bundle_modules()) >= 30
