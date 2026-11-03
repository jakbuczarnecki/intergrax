from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.path).replace("\\", "/")
        if "/ent_audit_x/unit/" in path:
            item.add_marker(pytest.mark.ent_l1)
        elif "/ent_audit_x/integration/" in path:
            if "test_semantic_gate_execution.py" not in path:
                item.add_marker(pytest.mark.ent_l2)
            else:
                item.add_marker(pytest.mark.ent_semantic_matrix)
        elif "/ent_audit_x/l3/" in path:
            item.add_marker(pytest.mark.ent_l3_heavy)
        elif "/ent_audit_x/acceptance/" in path:
            item.add_marker(pytest.mark.ent_semantic_gate)
