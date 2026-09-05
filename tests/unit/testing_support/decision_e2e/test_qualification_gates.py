# © Artur Czarnecki. All rights reserved.

"""Unit tests for fail-closed DS-E2E qualification constructors."""

from __future__ import annotations

import pytest

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.bindings import ProviderBindingEvidence
from testing_support.decision_e2e.evidence import scenario_execution_evidence_ref
from testing_support.decision_e2e.qualification_evidence import (
    DockerCrashEvidence,
    ScenarioExecutionEvidence,
)
from testing_support.decision_e2e.reporting import (
    QualificationReportCollector,
    validate_qualification_result,
)
from testing_support.decision_e2e.requirements import (
    qualify_docker_crash_resume,
    qualify_independent_verifier,
    qualify_live_scenario,
    qualify_real_multi_model,
)


def _binding(profile_id: str, provider: str, model: str) -> ProviderBindingEvidence:
    return ProviderBindingEvidence(profile_id=profile_id, provider=provider, model=model)


def test_qualify_real_multi_model_blocks_profile_only() -> None:
    result = qualify_real_multi_model(
        council_bindings=(
            _binding("profile-a", "ollama", "llama3.1:8b"),
            _binding("profile-b", "ollama", "llama3.1:8b"),
        ),
        evidence=(),
    )
    assert result.disposition is QualificationDisposition.BLOCKED


def test_qualify_independent_verifier_blocks_same_model() -> None:
    result = qualify_independent_verifier(
        producer=_binding("profile-producer", "ollama", "llama3.1:8b"),
        verifier=_binding("profile-verifier", "ollama", "llama3.1:8b"),
        evidence=(),
    )
    assert result.disposition is QualificationDisposition.BLOCKED


def test_subprocess_evidence_cannot_pass_docker_qualification() -> None:
    result = qualify_docker_crash_resume(
        crash_evidence=DockerCrashEvidence(
            kill_method="subprocess_exit",
            killed_container_id="",
            killed_exit_code=0,
            resume_container_id="",
            durable_store_path="/tmp",
            window="subprocess",
            final_disposition="finalization",
        ),
    )
    assert result.disposition is QualificationDisposition.BLOCKED


def test_assets_only_pass_rejected_by_collector_gate() -> None:
    blocked = validate_qualification_result(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_12,
            disposition=QualificationDisposition.PASSED,
            evidence=(),
            reason="assets only",
        ),
    )
    assert blocked.disposition is QualificationDisposition.BLOCKED


def test_live_scenario_requires_scenario_execution_evidence() -> None:
    passed = qualify_live_scenario(
        proof_id=DecisionE2EProofId.DS_E2E_12,
        scenario_evidence=ScenarioExecutionEvidence(
            scenario_id="ai_incident_investigation",
            invocation="uv run python ...",
            provider="ollama",
            model="llama3.1:8b",
            executed=True,
            decision_path_exercised=True,
            used_mock_provider=False,
            outcome="RESOLVED",
        ),
    )
    assert passed.disposition is QualificationDisposition.PASSED
    assert any(item.kind == "scenario_execution" for item in passed.evidence)


def test_false_passed_construction_still_blocked_for_ds_e2e_02() -> None:
    blocked = validate_qualification_result(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_02,
            disposition=QualificationDisposition.PASSED,
            evidence=(
                scenario_execution_evidence_ref(
                    ScenarioExecutionEvidence(
                        scenario_id="ignored",
                        invocation="x",
                        provider="ollama",
                        model="llama3.1:8b",
                        executed=True,
                        decision_path_exercised=True,
                        used_mock_provider=False,
                    ),
                ),
            ),
        ),
    )
    assert blocked.disposition is QualificationDisposition.BLOCKED


def test_qualification_report_rejects_duplicate_proof_ids() -> None:
    collector = QualificationReportCollector()
    collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_06,
            disposition=QualificationDisposition.BLOCKED,
            evidence=(),
            reason="negative contract gate",
        ),
    )
    collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_06,
            disposition=QualificationDisposition.PASSED,
            evidence=(),
            reason="authoritative proof",
        ),
    )
    with pytest.raises(ValueError, match="duplicate proof_id entries"):
        collector.build_report(environment_profile="test")
