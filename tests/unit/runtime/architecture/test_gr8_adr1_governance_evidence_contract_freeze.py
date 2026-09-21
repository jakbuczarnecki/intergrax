# © Artur Czarnecki. All rights reserved.

"""GR-8-ADR1 — frozen Governance Evidence public contract regression gates."""

from __future__ import annotations

import ast
import inspect
from dataclasses import MISSING, fields
from pathlib import Path

import pytest

from intergrax.contracts.governed_execution_governance_evidence import (
    SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1,
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
    GovernedExecutionEvaluationPoint,
    build_governance_fact_from_policy_decision,
)
from intergrax.contracts.runtime_execution_admission import RootExecutionAuthorityAdmissionRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

REPO_ROOT = Path(__file__).resolve().parents[4]
ADR_PATH = (
    REPO_ROOT
    / "docs/project/technical/adr/entries/2026-09-17/ADR-GR-8-001.md"
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_FULL_COVERAGE = (
    "all Governance evaluation points covered",
    "full Governance evidence coverage",
    "all Governance decisions are now persisted",
)

_GOVERNANCE_CORE_PATHS = (
    "intergrax/runtime/governance/root_execution_authority_admission.py",
    "intergrax/runtime/governance/governance_evidence_recorder.py",
    "intergrax/runtime/policy/meaningful_side_effect_authorization.py",
)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_gr8_adr1_contract_symbols_exist() -> None:
    assert GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION.value == "root_execution_admission"
    assert GovernanceEvidencePersistencePort.persist is not None
    assert SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1 == (
        "governed_execution_governance_decision_fact.v1"
    )


def test_gr8_adr1_fact_model_is_frozen() -> None:
    assert GovernanceDecisionEvidenceFact.model_config.get("frozen") is True
    assert GovernanceDecisionEvidenceFact.model_config.get("extra") == "forbid"


def test_gr8_adr1_persistence_outcome_not_policy_decision() -> None:
    hints = inspect.signature(GovernanceEvidencePersistencePort.persist).return_annotation
    assert hints is not inspect.Signature.empty
    assert "PolicyDecision" not in str(hints)
    source = inspect.getsource(GovernanceEvidencePersistencePort.persist)
    assert "PolicyDecision" not in source


def test_gr8_adr1_build_helper_projects_policy_decision_only() -> None:
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="ok",
        policy_bundle_id="b",
        policy_bundle_version="1",
        policy_bundle_digest="sha256:" + "a" * 64,
        policy_rule_id="r1",
    )
    fact = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION,
        tenant_id="t",
        workspace_id="w",
        principal_id="p",
        decision=decision,
        request_digest="sha256:" + "b" * 64,
        idempotency_key="idem-1",
        action="root_admit",
    )
    assert fact.decision is PolicyAction.ALLOW
    assert fact.schema_version == SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1


def test_gr8_adr1_governance_core_provider_neutral() -> None:
    forbidden = ("InMemoryRuntimeEventStore", "sqlite3", "kafka", "Elasticsearch")
    for rel in _GOVERNANCE_CORE_PATHS:
        text = _read(rel).lower()
        for token in forbidden:
            assert token.lower() not in text, f"{rel} must not reference provider {token}"


def test_gr8_adr1_no_global_evidence_sink_in_runtime() -> None:
    governance_dir = REPO_ROOT / "intergrax/runtime/governance"
    for path in governance_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "get_global" not in text
        assert "GLOBAL_" not in text


def test_gr8_adr1_schema_version_frozen_in_contract() -> None:
    text = _read("intergrax/contracts/governed_execution_governance_evidence.py")
    assert SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1 in text


def test_gr8_adr1_adr_scope_honesty_and_no_overclaim() -> None:
    assert ADR_PATH.is_file(), "ADR-GR-8-001 must exist for contract freeze"
    adr = ADR_PATH.read_text(encoding="utf-8")
    norm = adr.replace("**", "")
    assert "GR-8 = Governance Evidence SPINE" in norm
    assert "GR-10" in adr and "GR-13" in adr and "GR-12" in adr
    assert "runtime adoption remains partial" in adr or "partial" in adr.lower()
    for phrase in _FORBIDDEN_FULL_COVERAGE:
        if phrase not in adr:
            continue
        idx = adr.index(phrase)
        prefix = adr[max(0, idx - 40) : idx].lower()
        assert "do not" in prefix or "not claim" in prefix, f"overclaim phrase {phrase!r}"


def test_gr8_adr1_adr_documents_non_authority_and_failure_semantics() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")
    assert "MUST NOT return `PolicyDecision`" in adr or "MUST NOT** return `PolicyDecision`" in adr
    assert "Persistence failure" in adr
    assert (
        "ALLOW/DENY/REQUIRE_HUMAN unchanged" in adr
        or "ALLOW/DENY/REQUIRE_HUMAN/ESCALATE unchanged" in adr
        or "ALLOW remains ALLOW" in adr
    )
    assert "ESCALATE" in adr
    assert "additive-compatible" in adr.lower() or "additive compatible" in adr.lower()


def test_gr8_adr1_build_helper_accepts_escalate_matches_adr() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")
    assert "governed_execution_governance_decision_fact.v1" in adr
    fact = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION,
        tenant_id="t",
        workspace_id="w",
        principal_id="p",
        decision=PolicyDecision(action=PolicyAction.ESCALATE, reason="escalate"),
        request_digest="sha256:" + "cc" * 32,
        idempotency_key="adr-escalate",
        action="root",
    )
    assert fact.schema_version == SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1
    assert fact.decision is PolicyAction.ESCALATE


def test_gr8_adr1_root_admission_request_backward_compatible_optional_ids() -> None:
    optional_names = {
        f.name for f in fields(RootExecutionAuthorityAdmissionRequest) if f.default is not MISSING
    }
    for name in ("task_id", "run_id", "attempt_id", "execution_id"):
        assert name in optional_names, f"{name} must remain optional for backward compatibility"


def test_gr8_adr1_no_duplicate_governed_execution_evidence_enum() -> None:
    contracts = REPO_ROOT / "intergrax/contracts"
    enum_names: list[str] = []
    for path in contracts.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "StrEnum" for b in node.bases
            ):
                if node.name == "GovernedExecutionEvaluationPoint" and path.name != "governed_execution_governance_evidence.py":
                    pytest.fail(f"duplicate GovernedExecutionEvaluationPoint in {path}")
                if "EvaluationPoint" in node.name and node.name != "GovernedExecutionEvaluationPoint":
                    if "GovernedExecution" in node.name or node.name == "GovernanceEvaluationPoint":
                        enum_names.append(f"{path.name}:{node.name}")
    assert enum_names.count("governed_execution_governance_evidence.py:GovernedExecutionEvaluationPoint") <= 1


def test_gr8_adr1_persist_method_ast_return_not_policy_decision() -> None:
    path = REPO_ROOT / "intergrax/contracts/governed_execution_governance_evidence.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GovernanceEvidencePersistencePort":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "persist":
                    if item.returns is not None:
                        ret = ast.unparse(item.returns)
                        assert "PolicyDecision" not in ret
