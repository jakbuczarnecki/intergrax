# © Artur Czarnecki. All rights reserved.

"""MP-4R1 — single canonical Decision authority (identity, lifecycle, integration SPI)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax" / "contracts"
_DECISION_IDENTITY_PATH = _CONTRACTS_ROOT / "decision_identity.py"
_LEGACY_DECISION_MODULE = _CONTRACTS_ROOT / "decision.py"
_DECISION_PACKAGE_INIT = _CONTRACTS_ROOT / "decision" / "__init__.py"
_APPROVAL_PATH = _CONTRACTS_ROOT / "approval.py"
_LIFECYCLE_ADAPTER_PATH = (
    _CONTRACTS_ROOT
    / "decision"
    / "integration"
    / "lifecycle"
    / "default_adapter.py"
)
_DECISION_ID_NEWTYPE = re.compile(
    r'DecisionId\s*=\s*NewType\s*\(\s*["\']DecisionId["\']',
)
_FORBIDDEN_BRIDGE_MARKERS = (
    "_load_mp4b_decision_module",
    "_MP4B_EXPORTS",
    "spec_from_file_location",
)


def _production_python_files() -> list[Path]:
    roots = (
        _REPO_ROOT / "intergrax",
        _REPO_ROOT / "agents",
        _REPO_ROOT / "applications",
    )
    skip_parts = (
        "docker/runtime-context",
        "__pycache__",
    )
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if any(part in rel for part in skip_parts):
                continue
            files.append(path)
    return files


def test_mp4r1_exactly_one_decision_id_newtype_in_production() -> None:
    matches: list[str] = []
    for path in _production_python_files():
        text = path.read_text(encoding="utf-8")
        if _DECISION_ID_NEWTYPE.search(text):
            matches.append(str(path.relative_to(_REPO_ROOT)))
    normalized = sorted(path.replace("\\", "/") for path in matches)
    assert normalized == ["intergrax/contracts/decision_identity.py"], "\n".join(matches)


def test_mp4r1_legacy_decision_module_removed() -> None:
    assert not _LEGACY_DECISION_MODULE.is_file()


def test_mp4r1_no_legacy_mp4b_lifecycle_enum_in_contracts() -> None:
    legacy_lifecycle = _CONTRACTS_ROOT / "decision.py"
    if legacy_lifecycle.is_file():
        source = legacy_lifecycle.read_text(encoding="utf-8")
        assert "class DecisionLifecycleState" not in source
    package_sources = list((_CONTRACTS_ROOT / "decision").rglob("*.py"))
    for path in package_sources:
        source = path.read_text(encoding="utf-8")
        assert "DRAFT = \"draft\"" not in source
        assert "class DecisionLifecycleState" not in source or "decision_lifecycle" in str(path)


def test_mp4r1_decision_package_has_no_dynamic_legacy_bridge() -> None:
    source = _DECISION_PACKAGE_INIT.read_text(encoding="utf-8")
    for marker in _FORBIDDEN_BRIDGE_MARKERS:
        assert marker not in source
    tree = ast.parse(source, filename=str(_DECISION_PACKAGE_INIT))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr":
                pytest.fail(f"forbidden getattr in decision package init: line {node.lineno}")


def test_mp4r1_decision_integration_spi_importable() -> None:
    from intergrax.contracts.decision.integration import (
        DecisionIntegrationAdapterProvider,
        DecisionSystemIntegrationEngine,
    )
    from intergrax.contracts.decision.integration.lifecycle.default_adapter import (
        DefaultDecisionLifecycleIntegrationAdapter,
    )

    assert DecisionSystemIntegrationEngine is not None
    assert DecisionIntegrationAdapterProvider is not None
    assert DefaultDecisionLifecycleIntegrationAdapter is not None


def test_mp4r1_integration_lifecycle_adapter_uses_canonical_lifecycle() -> None:
    source = _LIFECYCLE_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "from intergrax.contracts.decision_lifecycle import" in source
    assert "DecisionLifecycleStage" in source


def test_mp4r1_approval_uses_canonical_decision_id() -> None:
    tree = ast.parse(_APPROVAL_PATH.read_text(encoding="utf-8"), filename=str(_APPROVAL_PATH))
    decision_import_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "intergrax.contracts.decision":
                decision_import_modules.append(node.module)
            if node.module == "intergrax.contracts.decision_identity":
                names = [alias.name for alias in node.names]
                assert "DecisionId" in names or "validate_decision_id" in names
    assert not decision_import_modules


def test_mp4r1_no_legacy_decision_aggregate_public_export() -> None:
    from intergrax.contracts import decision as decision_namespace

    assert "Decision" not in decision_namespace.__all__
    assert "DecisionLifecycleState" not in decision_namespace.__all__
    assert "mint_decision_id" not in decision_namespace.__all__


def test_mp4r1_canonical_identity_mint_and_wire_format() -> None:
    from intergrax.contracts.decision_identity import mint_decision_id, validate_decision_id

    decision_id = mint_decision_id()
    assert decision_id.startswith("decision_")
    assert len(decision_id) == len("decision_") + 32
    assert validate_decision_id(decision_id) == decision_id


def test_mp4r1_approval_accepts_canonical_decision_id() -> None:
    from intergrax.contracts.approval import CreateApprovalRequest, mint_approval_id
    from intergrax.contracts.decision_identity import mint_decision_id

    decision_id = mint_decision_id()
    payload = {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "decision_id": str(decision_id),
        "approval_id": str(mint_approval_id()),
        "acting_principal_id": "principal-1",
    }
    request = CreateApprovalRequest.model_validate(payload)
    assert request.decision_id == str(decision_id)


def test_mp4r1_integration_lifecycle_reference_not_authoritative_state() -> None:
    from intergrax.contracts.decision.integration.references import (
        PlatformDecisionLifecycleReference,
    )
    from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage

    ref = PlatformDecisionLifecycleReference(
        reference_decision_id="decision_" + "a" * 32,
        stage=DecisionLifecycleStage.PROPOSAL,
        transition_index=0,
        mapping_version="1",
    )
    assert ref.stage is DecisionLifecycleStage.PROPOSAL
    assert type(ref).__name__ == "PlatformDecisionLifecycleReference"


def test_mp4r1_no_multiplayer_decision_repository_in_collaborative_work() -> None:
    cw_root = _REPO_ROOT / "intergrax" / "collaborative_work"
    forbidden = (
        "MultiplayerDecisionRepository",
        "CollaborativeDecisionRepository",
        "DecisionRepository",
    )
    violations: list[str] = []
    for path in cw_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if f"class {name}" in text:
                violations.append(f"{path.relative_to(_REPO_ROOT)} defines {name}")
    assert not violations, "\n".join(violations)
