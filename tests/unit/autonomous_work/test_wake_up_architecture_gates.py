# © Artur Czarnecki. All rights reserved.

"""AW-4A corrective — architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.autonomous_work.repository import WorkerWakeUpReceiptClaimStatus
from intergrax.autonomous_work.wake_up_receipt_claim import resolve_wake_up_receipt_claim
from intergrax.contracts.autonomous_work.wake_up import WorkerWakeUpDisposition

pytestmark = pytest.mark.unit


def test_historical_migration_steps_use_fixed_target_versions() -> None:
    module = importlib.import_module("intergrax.autonomous_work.postgresql_repository")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_migrate_v1_to_v2":
            body_source = ast.get_source_segment(source, node) or ""
            assert "_SCHEMA_VERSION_V2" in body_source
            assert "_SCHEMA_VERSION," not in body_source.replace("_SCHEMA_VERSION_V", "")
        if node.name == "_migrate_v2_to_v3":
            body_source = ast.get_source_segment(source, node) or ""
            assert "_SCHEMA_VERSION_V3" in body_source
            assert "_SCHEMA_VERSION," not in body_source.replace("_SCHEMA_VERSION_V", "")
        if node.name == "_migrate_v3_to_v4":
            body_source = ast.get_source_segment(source, node) or ""
            assert "_SCHEMA_VERSION_V4" in body_source
            assert "_SCHEMA_VERSION," not in body_source.replace("_SCHEMA_VERSION_V", "")


def test_claim_status_distinguishes_duplicate_and_conflict() -> None:
    assert WorkerWakeUpReceiptClaimStatus.CLAIMED != WorkerWakeUpReceiptClaimStatus.DUPLICATE
    assert WorkerWakeUpReceiptClaimStatus.DUPLICATE != WorkerWakeUpReceiptClaimStatus.CONFLICT
    assert WorkerWakeUpReceiptClaimStatus.CLAIMED != WorkerWakeUpReceiptClaimStatus.CONFLICT


def test_service_disposition_includes_conflict() -> None:
    assert WorkerWakeUpDisposition.CONFLICT.value == "CONFLICT"


def test_wake_up_service_has_no_llm_or_execution_imports() -> None:
    module = importlib.import_module("intergrax.autonomous_work.wake_up_service")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported).lower()
    for token in (
        "openai",
        "anthropic",
        "langchain",
        "runtime.events",
        "runtime.task",
        "agents.",
        "execution_authority_admission",
    ):
        assert token not in joined


def test_wake_up_receipt_claim_module_is_port_based() -> None:
    module = importlib.import_module("intergrax.autonomous_work.wake_up_receipt_claim")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "postgresql" not in source
    assert "in_memory" not in source
    assert resolve_wake_up_receipt_claim is not None
