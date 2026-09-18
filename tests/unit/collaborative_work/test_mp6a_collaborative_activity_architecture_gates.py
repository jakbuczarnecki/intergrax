# © Artur Czarnecki. All rights reserved.

"""MP-6A — architecture gates for Collaborative Activity contract boundary."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_activity.py"

_FORBIDDEN_IMPORT_MARKERS = (
    "sqlalchemy",
    "psycopg",
    "asyncpg",
    "kafka",
    "nats",
    "elasticsearch",
    "pymongo",
    "boto3",
    "agents.",
    "applications.",
)

_FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"payload\s*:\s*dict\[", re.I),
    re.compile(r"metadata\s*:\s*dict\[str,\s*Any\]", re.I),
    re.compile(r"ActivityDatabase", re.I),
    re.compile(r"ActivityKafka", re.I),
)


def _read_contract() -> str:
    return _CONTRACT.read_text(encoding="utf-8-sig")


def test_mp6a_contract_module_exists() -> None:
    assert _CONTRACT.is_file()


def test_mp6a_no_forbidden_imports_or_providers() -> None:
    text = _read_contract()
    lowered = text.lower()
    for marker in _FORBIDDEN_IMPORT_MARKERS:
        assert marker not in lowered, f"forbidden marker in contract: {marker}"


def test_mp6a_no_forbidden_generic_payload_patterns() -> None:
    text = _read_contract()
    for pattern in _FORBIDDEN_NAME_PATTERNS:
        assert pattern.search(text) is None, f"forbidden pattern: {pattern.pattern}"


def test_mp6a_core_models_are_frozen() -> None:
    tree = ast.parse(_read_contract())
    frozen_models = {
        "CollaborativeActivity",
        "CollaborativeActivityPublication",
        "CollaborativeActivityActorRef",
        "ActivityIdempotencyKey",
        "CollaborativeActivityScope",
        "CollaborativeActivityTypeId",
        "CollaborativeActivitySourceId",
    }
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in frozen_models:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "model_config":
                        if "frozen=True" in ast.unparse(stmt.value):
                            found.add(node.name)
    missing = frozen_models - found
    assert not missing, f"models missing frozen=True model_config: {missing}"


def test_mp6a_public_ports_declared() -> None:
    text = _read_contract()
    for name in (
        "CollaborativeActivityPublicationPort",
        "CollaborativeActivityWritePort",
        "CollaborativeActivityReadPort",
        "CollaborativeActivityAppendStore",
    ):
        assert f"class {name}" in text
