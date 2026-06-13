# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gate, pytest.mark.no_ci]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_PREFIXES = (
    "intergrax/runtime/",
    "intergrax/applications/",
    "agents/",
)


def _python_files_under(*relative_roots: str) -> list[Path]:
    files: list[Path] = []
    for root in relative_roots:
        base = _REPO_ROOT / root.replace("/", "\\") if "\\" in str(_REPO_ROOT) else _REPO_ROOT / root
        if not base.exists():
            continue
        files.extend(base.rglob("*.py"))
    return files


def _imports_forbidden_module(path: Path, forbidden: str) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                hits.append(node.module)
    return hits


@pytest.mark.parametrize(
    "forbidden",
    ["intergrax.supervisor"],
)
def test_runtime_and_applications_do_not_import_experimental_packages(forbidden: str) -> None:
  roots = [p for p in _FORBIDDEN_PREFIXES if (_REPO_ROOT / p).exists()]
  violations: list[str] = []
  for path in _python_files_under(*roots):
      if "tests" in path.parts:
          continue
      for hit in _imports_forbidden_module(path, forbidden):
          rel = path.relative_to(_REPO_ROOT)
          violations.append(f"{rel}: {hit}")
  assert not violations, "Experimental imports found:\n" + "\n".join(violations)
