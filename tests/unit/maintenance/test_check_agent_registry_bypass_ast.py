# © Artur Czarnecki. All rights reserved.

"""AST policy tests for ``check_agent_registry_bypass`` (T-IGX-046-H / DP-14)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.maintenance.check_agent_registry_bypass import check_imports_in_tree

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _agents_tree(tmp_path: Path, rel: str, source: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        "import intergrax.integrations.contracts\n",
        "import intergrax.integrations.contracts.foo\n",
        "import intergrax.integrations.contracts.foo as alias\n",
        "from intergrax.integrations.contracts import Foo\n",
        "from intergrax.integrations.contracts.foo import Bar\n",
    ],
)
def test_ast_allows_contract_leaf_imports(tmp_path: Path, source: str) -> None:
    root = _agents_tree(tmp_path, "agents/sample.py", source)
    assert check_imports_in_tree(root) == []


@pytest.mark.parametrize(
    "source",
    [
        "import intergrax.integrations.providers\n",
        "import intergrax.integrations._shared\n",
        "import intergrax.integrations\n",
        "import intergrax.tools\n",
        "import intergrax.tools.providers\n",
        "from intergrax.integrations.providers import Foo\n",
        "from intergrax.integrations._shared import Foo\n",
        "from intergrax.integrations import contracts\n",
        "from intergrax import integrations\n",
        "from intergrax.tools import Foo\n",
        "from intergrax import tools\n",
    ],
)
def test_ast_denies_non_contract_integrations_and_tools(tmp_path: Path, source: str) -> None:
    root = _agents_tree(tmp_path, "agents/sample.py", source)
    findings = check_imports_in_tree(root)
    assert len(findings) == 1
    assert findings[0].startswith("agents/sample.py:")


def test_allowlist_skips_agents_init(tmp_path: Path) -> None:
    root = _agents_tree(tmp_path, "agents/__init__.py", "import intergrax.tools\n")
    assert check_imports_in_tree(root) == []


def test_skips_paths_under_tests_in_parts(tmp_path: Path) -> None:
    root = _agents_tree(tmp_path, "agents/foo/tests/bar.py", "import intergrax.tools\n")
    assert check_imports_in_tree(root) == []


def test_syntax_error_is_a_finding(tmp_path: Path) -> None:
    root = _agents_tree(tmp_path, "agents/broken.py", "def (\n")
    findings = check_imports_in_tree(root)
    assert len(findings) == 1
    assert "syntax error" in findings[0]


def test_findings_are_sorted(tmp_path: Path) -> None:
    _agents_tree(tmp_path, "agents/zed.py", "import intergrax.tools\n")
    _agents_tree(tmp_path, "agents/ace.py", "import intergrax.tools\n")
    findings = check_imports_in_tree(tmp_path)
    assert findings == sorted(findings)
    assert [f.split(":")[0] for f in findings] == ["agents/ace.py", "agents/zed.py"]
