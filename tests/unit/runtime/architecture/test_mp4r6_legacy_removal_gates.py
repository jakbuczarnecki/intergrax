# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — retired MP-4 authority paths removed; single canonical contract families only."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_APPROVAL_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "approval.py"
_LEGACY_APPROVAL_PACKAGE = _REPO_ROOT / "intergrax" / "approval"
_LEGACY_DECISION_MODULE = _REPO_ROOT / "intergrax" / "contracts" / "decision.py"
_LEGACY_HUMAN_COMPAT_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "human" / "legacy_human_input_compatibility.py"
)
_DECISION_INTEGRATION_PACKAGE = _REPO_ROOT / "intergrax" / "contracts" / "decision"
_INTEGRATION_NAMESPACE_PREFIXES = (
    "intergrax.contracts.decision",
    "intergrax.contracts.decision.integration",
)
_CANONICAL_DECISION_MODULE_PREFIXES = (
    "intergrax.contracts.decision_identity",
    "intergrax.contracts.decision_lifecycle",
    "intergrax.contracts.decision_human_review",
    "intergrax.contracts.decision_record",
    "intergrax.contracts.decision_finalization",
    "intergrax.contracts.decision_checkpoint",
    "intergrax.contracts.decision_revision",
    "intergrax.contracts.decision_resolution",
    "intergrax.contracts.decision_verification",
    "intergrax.contracts.decision_verification_stage",
    "intergrax.contracts.decision_authorization",
    "intergrax.contracts.decision_authoritative_exposure",
    "intergrax.contracts.decision_artifact_registry",
    "intergrax.contracts.decision_strategy",
    "intergrax.contracts.decision_exposure_selection",
    "intergrax.contracts.decision_coordination",
    "intergrax.contracts.decision_disagreement",
)
_FORBIDDEN_LEGACY_IMPORT_PREFIXES = (
    "intergrax.contracts.approval",
    "intergrax.approval",
)
_FORBIDDEN_LEGACY_HUMAN_COMPAT_IMPORT = "intergrax.runtime.human.legacy_human_input_compatibility"
_FORBIDDEN_BRIDGE_MARKERS = (
    "_load_mp4b_decision_module",
    "_MP4B_EXPORTS",
    "spec_from_file_location",
)
_FORBIDDEN_CONTINUATION_AUTHORITY_PREFIXES = (
    "MultiplayerContinuation",
    "CollaborativeContinuation",
    "ApprovalContinuation",
    "DecisionContinuation",
)
_FORBIDDEN_EVIDENCE_STORE_NAMES = frozenset(
    {
        "MultiplayerEvidenceRepository",
        "CollaborativeEvidenceStore",
        "DecisionBindingEvidenceRepository",
        "MultiplayerEvidence",
        "CollaborativeEvidenceRecord",
        "DecisionBindingEvidenceRecord",
    },
)
_PRODUCTION_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
)
_COLLABORATIVE_ROOT = _REPO_ROOT / "intergrax" / "collaborative_work"
_DECISION_ID_NEWTYPE = re.compile(
    r'DecisionId\s*=\s*NewType\s*\(\s*["\']DecisionId["\']',
)


def _production_python_files() -> list[Path]:
    skip_parts = ("docker/runtime-context", "__pycache__", "tests")
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
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


def _collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _is_legacy_mp4b_decision_import(module: str) -> bool:
    if any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _INTEGRATION_NAMESPACE_PREFIXES
    ):
        return False
    if not module.startswith("intergrax.contracts.decision."):
        return module == "intergrax.contracts.decision"
    return not any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _CANONICAL_DECISION_MODULE_PREFIXES
    )


def test_mp4r6_retired_legacy_modules_absent_on_disk() -> None:
    assert not _LEGACY_APPROVAL_CONTRACT.is_file()
    assert not _LEGACY_APPROVAL_PACKAGE.exists()
    assert not _LEGACY_DECISION_MODULE.is_file()
    assert not _LEGACY_HUMAN_COMPAT_MODULE.is_file()
    assert _DECISION_INTEGRATION_PACKAGE.is_dir()


def test_mp4r6_no_production_imports_of_legacy_approval_authority() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_import_modules(path):
            for prefix in _FORBIDDEN_LEGACY_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{rel} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp4r6_no_production_imports_of_retired_mp4b_decision_spi() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_import_modules(path):
            if _is_legacy_mp4b_decision_import(module):
                violations.append(f"{rel} imports retired decision SPI {module}")
    assert not violations, "\n".join(violations)


def test_mp4r6_no_legacy_human_input_compatibility_shim_module() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_import_modules(path):
            if module == _FORBIDDEN_LEGACY_HUMAN_COMPAT_IMPORT or module.startswith(
                f"{_FORBIDDEN_LEGACY_HUMAN_COMPAT_IMPORT}."
            ):
                violations.append(f"{rel} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp4r6_decision_package_has_no_dynamic_legacy_bridge() -> None:
    init_path = _DECISION_INTEGRATION_PACKAGE / "__init__.py"
    source = init_path.read_text(encoding="utf-8-sig")
    for marker in _FORBIDDEN_BRIDGE_MARKERS:
        assert marker not in source, f"forbidden bridge marker {marker!r} in decision __init__"


def test_mp4r6_exactly_one_decision_id_newtype_in_production() -> None:
    matches: list[str] = []
    for path in _production_python_files():
        text = path.read_text(encoding="utf-8-sig")
        if _DECISION_ID_NEWTYPE.search(text):
            matches.append(path.relative_to(_REPO_ROOT).as_posix())
    assert matches == ["intergrax/contracts/decision_identity.py"], "\n".join(matches)


def test_mp4r6_collaborative_work_has_no_duplicate_continuation_authority_types() -> None:
    if not _COLLABORATIVE_ROOT.is_dir():
        pytest.skip("collaborative_work root missing")
    violations: list[str] = []
    for path in _COLLABORATIVE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for prefix in _FORBIDDEN_CONTINUATION_AUTHORITY_PREFIXES:
            if re.search(rf"\bclass\s+{prefix}\w*", source):
                violations.append(f"{rel} defines continuation authority {prefix}")
    assert not violations, "\n".join(violations)


def test_mp4r6_collaborative_work_has_no_multiplayer_evidence_store_types() -> None:
    if not _COLLABORATIVE_ROOT.is_dir():
        pytest.skip("collaborative_work root missing")
    violations: list[str] = []
    for path in _COLLABORATIVE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for name in _FORBIDDEN_EVIDENCE_STORE_NAMES:
            if name in source:
                violations.append(f"{rel}: {name}")
    assert not violations, "\n".join(violations)


def test_mp4r6_canonical_authority_contracts_importable() -> None:
    from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
    from intergrax.contracts.decision_human_review import DecisionHumanReviewPort
    from intergrax.contracts.decision_record import DecisionProposalRef
    from intergrax.contracts.execution_continuation import ExecutionContinuationPort
    from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence

    assert CollaborativeDecisionBinding is not None
    assert DecisionHumanReviewPort is not None
    assert DecisionProposalRef is not None
    assert ExecutionContinuationPort is not None
    assert FunctionalEvidencePersistence is not None


def test_mp4r6_human_review_uses_metadata_bridge_not_compat_module() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "human_response.py"
    source = path.read_text(encoding="utf-8-sig")
    assert "legacy_human_input_compatibility" not in source
    assert "promote_legacy_human_verdict_from_metadata" in source


_HUMAN_RESPONSE_RESTORE = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "human_response.py"
)
_GENERIC_RESTORE_ALLOWLIST = frozenset(
    {
        "intergrax/debug/hitl_service.py",
        "intergrax/tools/providers/hitl/service.py",
        "intergrax/applications/_shared/task_control.py",
    }
)
_HUMAN_DECISION_STORE = _REPO_ROOT / "intergrax" / "runtime" / "human" / "store.py"


def test_mp4r6_generic_checkpoint_restore_does_not_import_local_dev_approver() -> None:
    source = _HUMAN_RESPONSE_RESTORE.read_text(encoding="utf-8-sig")
    assert "local_development_approver_evidence" not in source


def test_mp4r6_generic_restore_paths_do_not_synthesize_approver_from_task_user() -> None:
    violations: list[str] = []
    needle_call = "local_development_approver_evidence("
    user_fallback = "actor_id=task.user_id"
    for path in _production_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _GENERIC_RESTORE_ALLOWLIST:
            continue
        if "prepare_hitl_resume_after_checkpoint_restore" in path.read_text(encoding="utf-8-sig"):
            source = path.read_text(encoding="utf-8-sig")
            if needle_call in source or user_fallback in source:
                violations.append(rel)
    assert not violations, "\n".join(violations)


def test_mp4r6_prepare_hitl_restore_fail_closed_marker_present() -> None:
    source = _HUMAN_RESPONSE_RESTORE.read_text(encoding="utf-8-sig")
    assert "HitlCheckpointRestoreError" in source
    assert "approver evidence missing during HITL checkpoint restore" in source
    assert "local_development_approver_evidence" not in source


def test_mp4r6_sqlite_human_decision_store_read_path_does_not_synthesize_approver() -> None:
    source = _HUMAN_DECISION_STORE.read_text(encoding="utf-8-sig")
    assert "local_development_approver_evidence" not in source
    assert "legacy_unknown_approver" not in source
    assert "deserialize_persisted_human_approver_evidence" in source


def test_mp4r6_persistence_deserialization_does_not_map_user_id_to_approver() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "human" / "persistence_errors.py"
    source = path.read_text(encoding="utf-8-sig")
    assert 'row["user_id"]' not in source
    assert "user_id" not in source or "tenant_id" in source
    assert "local_development_approver_evidence" not in source
    assert "legacy_unknown_approver" not in source
