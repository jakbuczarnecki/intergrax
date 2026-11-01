# © Artur Czarnecki. All rights reserved.

"""Shared experiment harness: fixture materialisation, oracle loading, arm results (S1..S8)."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testing_support.change_safety.contracts.artifacts import ExperimentVerdict
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory, build_closed_world_inventory

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "fixtures" / "evol_x"
_SUFFIX = ".fixture"
_ROOTS = ("intergrax/", "agents/", "applications/")


@dataclass(frozen=True)
class ArmResult:
    scenario_id: str
    arm: str  # "positive" | "negative"
    passed: bool
    detail: str

    def to_verdict(self, source_commit_sha: str) -> ExperimentVerdict:
        return ExperimentVerdict(
            scenario_id=self.scenario_id,
            arm=self.arm,  # type: ignore[arg-type]
            passed=self.passed,
            source_commit_sha=source_commit_sha,
            detail=self.detail,
        )


def load_oracle(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name / "oracle_declared.json").read_text(encoding="utf-8"))


def materialize(name: str, dest: Path) -> Path:
    """Copy ``fixtures/evol_x/<name>/**/*.py.fixture`` to ``dest`` with the suffix stripped."""
    src = FIXTURE_ROOT / name
    for file in src.rglob(f"*{_SUFFIX}"):
        target = dest / file.relative_to(src).as_posix()[: -len(_SUFFIX)]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    return dest


def fixture_sources(name: str) -> dict[str, str]:
    """Logical path (suffix stripped) -> source text, read straight from the committed fixture."""
    src = FIXTURE_ROOT / name
    return {
        file.relative_to(src).as_posix()[: -len(_SUFFIX)]: file.read_text(encoding="utf-8")
        for file in sorted(src.rglob(f"*{_SUFFIX}"))
    }


@contextmanager
def fixture_inventory(name: str) -> Iterator[ClosedWorldInventory]:
    with tempfile.TemporaryDirectory(prefix=f"evol-{name}-") as tmp:
        root = materialize(name, Path(tmp))
        yield build_closed_world_inventory(source_commit_sha="fixture", roots=_ROOTS, fixture_root=root)


def source_sha() -> str:
    from testing_support.change_safety.intake.git_change_set import rev_parse_head

    return rev_parse_head()
