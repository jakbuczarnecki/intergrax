# © Artur Czarnecki. All rights reserved.

"""W5-07: no duplicate authority (suite registry / contract schema) and no ``typing.Any`` in artifact contracts."""

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_PKG = _REPO / "testing_support" / "change_safety"


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_change_safety_has_no_parallel_suite_registry() -> None:
    offenders: list[str] = []
    for path in _py_files(_PKG):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("tests/qualification/qual_x/") and "::" in node.value:
                    offenders.append(f"{path.relative_to(_REPO)}: hard-coded QUAL-X node id")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("QUAL_X_"):
                        offenders.append(f"{path.relative_to(_REPO)}: redefines {target.id}")
    assert offenders == []


def test_t1_t2_are_derived_from_the_qual_x_catalog_not_copied() -> None:
    from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter
    from tests.qualification.qual_x._p5_freeze_selection_catalog import qual_x_p5_mandatory_collectable_node_ids

    assert QualXSuiteCatalogAdapter().mandatory_min_node_ids() == frozenset(qual_x_p5_mandatory_collectable_node_ids())


def test_contract_schema_authority_is_the_platform_registry_not_a_copy() -> None:
    from tests.qualification.evol_x.adapters.compat_x import CompatXRegistryAdapter

    adapter = CompatXRegistryAdapter()
    assert adapter.list_contract_names()
    data_dir = _REPO / "tests" / "qualification" / "evol_x" / "data"
    copies = [p.name for p in data_dir.glob("*.json") if "contract_schema" in p.name]
    assert copies == []  # no local copy of the contract schema registry


def test_artifact_contracts_forbid_typing_any_and_authority_field_names() -> None:
    offenders: list[str] = []
    for path in _py_files(_PKG / "contracts"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                offenders += [f"{path.name}: typing.Any" for a in node.names if a.name == "Any"]
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in {"approve", "authorize", "execute"}:
                    offenders.append(f"{path.name}: forbidden field {node.target.id}")
    # canonical_json.py is a pure serialisation helper (not a contract model) and may use Any.
    assert [o for o in offenders if not o.startswith("canonical_json.py")] == []
