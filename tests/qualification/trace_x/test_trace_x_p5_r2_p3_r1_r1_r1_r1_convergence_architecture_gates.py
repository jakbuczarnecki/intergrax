# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1-R1-R1-R1 convergence / reuse architecture gates (docs-only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "2cfcac2907428ac6d634671913cb6958dbc82929"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK.md"
)
_PRIOR_LOCK = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_CONFIGURED_EXECUTION_SUBJECT_ARCHITECTURE_LOCK.md"
)
_ROADMAP = _REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_IDENTITY_KEY = _REPO_ROOT / "intergrax/contracts/capability_catalog/identity_key.py"
_EXEC_BOUND_DISPATCH = (
    _REPO_ROOT
    / "intergrax/contracts/execution/execution_bound_capability_execution_dispatch.py"
)
_EXEC_BOUND_SERVICE = (
    _REPO_ROOT
    / "intergrax/runtime/execution/execution_bound_capability_execution_dispatch_service.py"
)
_EXEC_BOUND_DELEGATE = (
    _REPO_ROOT
    / "intergrax/runtime/execution/execution_bound_capability_execution_runtime_delegate.py"
)
_CANDIDATE_CONTRACT = (
    _REPO_ROOT / "intergrax/contracts/autonomous_work/capability_acquisition.py"
)
_RUNTIME_ROOT = _REPO_ROOT / "intergrax/runtime/execution"


def test_txp5r2p3r1r1r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1r1r1r1_q02_convergence_lock_ready_for_audit() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in text
    assert "R2-P3-EXECUTION-INGRESS-DUPLICATION-14" in text
    assert "R2-P3-BUSINESS-TARGET-IDENTITY-REUSE-15" in text
    assert "FRZ-TRC-11" in text and "OPEN" in text


def test_txp5r2p3r1r1r1r1_q03_prior_lock_superseded_duplicate_prone() -> None:
    prior = _PRIOR_LOCK.read_text(encoding="utf-8")
    assert "SUPERSEDED" in prior
    assert "TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK" in prior


def test_txp5r2p3r1r1r1r1_q04_execution_bound_family_exists() -> None:
    assert _EXEC_BOUND_DISPATCH.is_file()
    assert _EXEC_BOUND_SERVICE.is_file()
    assert _EXEC_BOUND_DELEGATE.is_file()
    service = _EXEC_BOUND_SERVICE.read_text(encoding="utf-8")
    assert "ExecutionBoundCapabilityExecutionDispatchService" in service
    assert "RootExecutionLaunchPort" in service
    delegate = _EXEC_BOUND_DELEGATE.read_text(encoding="utf-8")
    assert "QualifiedCapabilityExecutionBindingHandlerRegistry" in delegate


def test_txp5r2p3r1r1r1r1_q05_prior_lock_proposed_parallel_configured_ingress() -> None:
    prior = _PRIOR_LOCK.read_text(encoding="utf-8")
    assert "ConfiguredCapabilityExecutionDispatchService" in prior
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "DUPLICATE / BLOCKER" in lock
    assert "FACTOR SHARED CORE" in lock


def test_txp5r2p3r1r1r1r1_q06_capability_identity_key_exists() -> None:
    source = _IDENTITY_KEY.read_text(encoding="utf-8")
    assert "class CapabilityIdentityKey" in source


def test_txp5r2p3r1r1r1r1_q07_candidate_lacks_capability_identity_key() -> None:
    source = _CANDIDATE_CONTRACT.read_text(encoding="utf-8")
    assert "class WorkerCapabilityCandidate" in source
    assert "CapabilityIdentityKey" not in source


def test_txp5r2p3r1r1r1r1_q08_no_production_configured_dispatch_service() -> None:
    configured_dispatch = list(_RUNTIME_ROOT.glob("configured_capability_execution*.py"))
    assert configured_dispatch == []


def test_txp5r2p3r1r1r1r1_q09_no_second_intent_repository_module() -> None:
    forbidden = _REPO_ROOT / "intergrax/tools/configured_marketplace_tool_execution_intent.py"
    assert not forbidden.is_file()
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "second intent repository" in lock.lower() or "Second marketplace intent" in lock


def test_txp5r2p3r1r1r1r1_q10_single_handler_registry_pattern() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "QualifiedCapabilityExecutionBindingHandlerRegistry" in lock
    assert "second handler registry" in lock.lower() or "Second handler registry" in lock
    forbidden_registry = (
        _REPO_ROOT
        / "intergrax/runtime/execution/configured_capability_execution_handler_registry.py"
    )
    assert not forbidden_registry.is_file()


def test_txp5r2p3r1r1r1r1_q11_rejects_handoff_identity_for_variant_b() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "MarketplaceQualifiedToolBusinessTarget" in lock
    assert "handoff_id" in lock
    assert "ELIMINATED" in lock or "REJECT" in lock


def test_txp5r2p3r1r1r1r1_q12_dup_x_roadmap_singleton_and_replay_dependency() -> None:
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert roadmap.count("### 3.0.3 `DUP-X`") == 1
    assert "**Depends on:** `DUP-X`" in roadmap or "Depends on:** `DUP-X`" in roadmap
    assert "`ROADMAP-REPLAY-X` must not start until `DUP-X = CLOSED`" in roadmap
    assert "**`DUP-X` = CLOSED**" in roadmap
