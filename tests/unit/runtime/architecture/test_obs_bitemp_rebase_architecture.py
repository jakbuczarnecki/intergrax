# © Artur Czarnecki. All rights reserved.

"""OBS-BITEMP-REBASE — E/K/V/S temporal composition architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_TEMPORAL_CLUSTER = (
    _REPO_ROOT / "intergrax" / "contracts" / "bitemporal_knowledge.py",
    _REPO_ROOT / "intergrax" / "contracts" / "historical_reconstruction.py",
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "historical_reconstruction.py",
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "knowledge_reconstruction.py",
)

_FORBIDDEN_CONTROL = frozenset(
    {
        "ExecutionRuntime",
        "HistoricalRuntime",
        "ReplayRuntime",
        "TemporalRuntime",
        "TemporalProvider",
        "TimeManager",
        "HistoryClock",
    }
)

_NOW_PATTERNS = (
    re.compile(r"datetime\.now\s*\("),
    re.compile(r"datetime\.utcnow\s*\("),
    re.compile(r"time\.time\s*\("),
)

_CROSS_AXIS_PATTERNS = (
    re.compile(r"ExecutionEventPosition.*KnowledgeRevision"),
    re.compile(r"finalized_through_value.*ExecutionEventPosition"),
    re.compile(r"valid_time.*recorded_at"),
    re.compile(r"recorded_at.*valid_time"),
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_obs_bitemp_single_historical_reconstruction_service() -> None:
    path = _TEMPORAL_CLUSTER[2]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    service_classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Service")
    ]
    assert service_classes == ["HistoricalReconstructionService"]


def test_obs_bitemp_cluster_no_diagnostics_imports() -> None:
    for path in _TEMPORAL_CLUSTER:
        text = path.read_text(encoding="utf-8")
        assert "intergrax.runtime.diagnostics" not in text, path.relative_to(_REPO_ROOT)


def test_obs_bitemp_cluster_no_implicit_now() -> None:
    for path in _TEMPORAL_CLUSTER[2:]:
        text = path.read_text(encoding="utf-8")
        for pattern in _NOW_PATTERNS:
            assert pattern.search(text) is None, (
                f"implicit clock pattern {pattern.pattern!r} in {path.relative_to(_REPO_ROOT)}"
            )


def test_obs_bitemp_k_prefix_before_bitemporal_filter() -> None:
    text = (_TEMPORAL_CLUSTER[2]).read_text(encoding="utf-8")
    fn_start = text.index("def _reconstruct_bitemporal_knowledge")
    fn_body = text[fn_start:]
    k_pos = fn_body.index("reconstruct_knowledge_at_watermark")
    filter_pos = fn_body.index("revision_admissible_at_bitemporal_query")
    assert k_pos < filter_pos


def test_obs_bitemp_contract_basis_carries_full_coordinate() -> None:
    contract = (_TEMPORAL_CLUSTER[1]).read_text(encoding="utf-8")
    for field in (
        "execution_as_of: AsOfBoundary",
        "knowledge_watermark: KnowledgeRevisionWatermark",
        "bitemporal_query: BitemporalKnowledgeBasis",
    ):
        assert field in contract


def test_obs_bitemp_typed_axis_authorities_in_contracts() -> None:
    bitemp = (_TEMPORAL_CLUSTER[0]).read_text(encoding="utf-8")
    assert "class ValidTimeBasis" in bitemp
    assert "class SystemTimeBasis" in bitemp
    assert "class RevisionOrderingAuthority" in bitemp
    assert "class KnowledgeRevisionWatermark" in bitemp
    historical = (_TEMPORAL_CLUSTER[1]).read_text(encoding="utf-8")
    assert "def revision_admissible_at_bitemporal_query" in historical


def test_obs_bitemp_no_universal_temporal_provider_names() -> None:
    for path in _TEMPORAL_CLUSTER:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert names.isdisjoint(_FORBIDDEN_CONTROL), path.relative_to(_REPO_ROOT)


def test_obs_bitemp_no_provider_type_checks_in_service() -> None:
    text = (_TEMPORAL_CLUSTER[2]).read_text(encoding="utf-8")
    assert "isinstance(" not in text
    assert "getattr(" not in text
    assert "hasattr(" not in text


def test_obs_bitemp_cross_axis_misuse_patterns_absent() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in _TEMPORAL_CLUSTER[2:])
    for pattern in _CROSS_AXIS_PATTERNS:
        assert pattern.search(combined) is None, pattern.pattern


def test_obs_bitemp_revision_ordering_port_injected() -> None:
    imports = _module_imports(_TEMPORAL_CLUSTER[2])
    assert "intergrax.contracts.bitemporal_knowledge" in imports
    text = _TEMPORAL_CLUSTER[2].read_text(encoding="utf-8")
    assert "RevisionOrderingAuthority" in text
    assert "KnowledgeRevisionReader" in text


def test_obs_bitemp_lineage_r1_regression_suite_exists() -> None:
    path = (
        _REPO_ROOT
        / "tests"
        / "unit"
        / "runtime"
        / "observability"
        / "reconstruction"
        / "test_obs_asof_rebase_r1_lineage_integrity.py"
    )
    assert path.is_file()
