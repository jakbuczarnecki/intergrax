# © Artur Czarnecki. All rights reserved.

"""AW-7A — capability acquisition architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_AW7A_MODULE_PATHS = (
    "capability_acquisition_service.py",
    "capability_acquisition_ports.py",
    "capability_discovery_adapters.py",
)

_FORBIDDEN_IMPORTS = (
    "intergrax.runtime.execution",
    "ExecutionRuntime",
    "ParentExecutionAuthority",
    "AuthorityGrant",
    "codecraft",
    "CodeCraft",
    "sqlalchemy",
    "psycopg",
    "openai",
    "anthropic",
    "sandbox",
)

_FORBIDDEN_REGISTRY_TOKENS = (
    "WorkerToolRegistry",
    "WorkerSkillRegistry",
    "WorkerIntegrationRegistry",
    "WorkerCapabilityRegistry",
    "CapabilityManager",
    "CapabilityEngine",
    "AdaptiveCapabilityRuntime",
    "CapabilityOrchestratorFactory",
)


def _aw7a_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [base / name for name in _AW7A_MODULE_PATHS]


def test_aw7a_modules_forbidden_imports() -> None:
    for path in _aw7a_paths():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        joined = "\n".join(imported)
        for token in _FORBIDDEN_IMPORTS:
            assert token.lower() not in joined.lower(), f"{path.name} imports {token}"


def test_aw7a_modules_forbidden_registry_tokens() -> None:
    for path in _aw7a_paths():
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_REGISTRY_TOKENS:
            assert token not in source, f"{path.name} defines forbidden token {token}"


def test_aw7a_contracts_no_codecraft_execution_import() -> None:
    module = importlib.import_module(
        "intergrax.contracts.autonomous_work.capability_acquisition",
    )
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported).lower()
    assert "codecraft" not in joined
    assert "ExecutionRuntime" not in joined


def test_aw7a_public_import_smoke() -> None:
    from intergrax.autonomous_work.capability_acquisition_service import (
        WorkerCapabilityAcquisitionDecisionService,
    )
    from intergrax.contracts.autonomous_work.capability_acquisition import (
        WorkerAutonomyLevel,
        a4_never_self_authorized,
    )

    assert WorkerCapabilityAcquisitionDecisionService is not None
    assert a4_never_self_authorized(WorkerAutonomyLevel.A4_AUTHORITY_CHANGE) is True


def test_a4_invariant_gate_rejects_use_existing_with_a4() -> None:
    from intergrax.contracts.autonomous_work.capability_acquisition import (
        CapabilityAcquisitionDisposition,
        CapabilityAcquisitionReasonCode,
        WorkerAutonomyLevel,
        WorkerCapabilityAcquisitionDecision,
        WorkerCapabilityCandidate,
        WorkerCapabilityCandidateKind,
    )
    from intergrax.contracts.autonomous_work.profile_reference import (
        CapabilityProfileRef,
        initial_profile_version,
    )
    from intergrax.contracts.autonomous_work.references import ProblemReference
    from datetime import UTC, datetime

    candidate = WorkerCapabilityCandidate(
        candidate_id="tool:sample",
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:sample",
        source_domain="tools",
        operations=("document.parse_csv",),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(ProblemReference("problem/evidence/1"),),
        discovered_at=datetime.now(tz=UTC),
    )
    with pytest.raises(ValueError, match="USE_EXISTING requires selected_candidate and autonomy A0"):
        WorkerCapabilityAcquisitionDecision(
            decision_id="decision-a4",
            worker_instance_id="winst_00000000000000000000000000000001",
            obstacle_id="winst_00000000000000000000000000000001:execution_failure:x:y",
            recovery_decision_id="recovery-a4",
            need_id="need-a4",
            disposition=CapabilityAcquisitionDisposition.USE_EXISTING,
            selected_candidate=candidate,
            autonomy_level=WorkerAutonomyLevel.A4_AUTHORITY_CHANGE,
            capability_profile_ref=CapabilityProfileRef(
                profile_id="cap/default",
                version=initial_profile_version(),
            ),
            reason_code=CapabilityAcquisitionReasonCode.EXISTING_TOOL_SELECTED,
            evidence_refs=(ProblemReference("problem/evidence/1"),),
            decided_at=datetime.now(tz=UTC),
        )
