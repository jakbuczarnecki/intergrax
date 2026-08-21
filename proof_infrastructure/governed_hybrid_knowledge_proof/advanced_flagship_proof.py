# © Artur Czarnecki. All rights reserved.

"""Advanced flagship proof runner for COMM-5 F3-F."""

from __future__ import annotations

import asyncio
import sys

from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_docker_environment import (
    AdvancedFlagshipDockerEnvironmentV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_docker_scenario import (
    FlagshipDockerScenarioV1,
    _FRESH_SECURITY_UPDATED_AT,
    _STALE_SECURITY_UPDATED_AT,
    build_flagship_docker_scenario,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_formatter import (
    build_history_comparison,
    build_scenario_proof,
    distinct_live_identities,
    format_scenario_section,
    live_call_failure_for_suffix,
    summary_label_for_admissibility,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_models import (
    AdvancedFlagshipProofResultV1,
    FlagshipScenarioIdV1,
    FlagshipScenarioProofV1,
    FlagshipSummaryRowV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_policy import (
    FLAGSHIP_POLICY_REV_17,
    FLAGSHIP_POLICY_REV_18,
    FLAGSHIP_QUESTION,
)

_BANNER = "=" * 72


async def _run_scenarios() -> AdvancedFlagshipProofResultV1:
    environment = AdvancedFlagshipDockerEnvironmentV1.from_defaults()
    environment.ensure_ready()
    scenario = await build_flagship_docker_scenario(environment)
    proofs: list[FlagshipScenarioProofV1] = []

    scenario.seed_valid_baseline(security_updated_at=_STALE_SECURITY_UPDATED_AT)
    scenario.set_policy_revision(policy_revision=FLAGSHIP_POLICY_REV_17)
    scenario.use_inner_orchestrator()
    scenario.restore_active_bindings()
    rev17_run = await scenario.ask(
        run_id="flagship-rev17-valid",
        request_id="request-flagship-rev17-valid",
    )
    rev17_passed = (
        FlagshipDockerScenarioV1.is_satisfied(rev17_run)
        and scenario.llm.calls == 1
        and len(rev17_run.live_call_failures) == 0
    )
    providers, connections, capabilities, call_ids = distinct_live_identities(rev17_run)
    rev17_proof = build_scenario_proof(
        scenario_id=FlagshipScenarioIdV1.REV17_ALL_SATISFIED,
        run=rev17_run,
        llm_calls=scenario.llm.calls,
        passed=rev17_passed and len(providers) == 4,
        detail=(
            f"providers={len(providers)}, connections={len(connections)}, "
            f"capabilities={len(capabilities)}, call_ids={len(call_ids)}"
        ),
    )
    proofs.append(rev17_proof)

    reloaded_success = scenario.reload_run(run_id="flagship-rev17-valid")
    reload_success_ok = (
        reloaded_success.evidence_admissibility is not None
        and reloaded_success.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )
    proofs[-1] = rev17_proof.model_copy(
        update={
            "passed": rev17_proof.passed and reload_success_ok,
            "detail": rev17_proof.detail + f"; reload_success={reload_success_ok}",
        }
    )

    scenario.set_policy_revision(policy_revision=FLAGSHIP_POLICY_REV_18)
    scenario.reset_vendor_read_counts()
    rev18_stale_run = await scenario.ask(
        run_id="flagship-rev18-stale",
        request_id="request-flagship-rev18-stale",
    )
    stale_security = scenario.evaluation_for_suffix(rev18_stale_run, "security")
    rev18_stale_passed = (
        FlagshipDockerScenarioV1.is_unsatisfied(rev18_stale_run)
        and scenario.llm.calls == 0
        and stale_security.reason_code
        is RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID
        and rev17_run.policy_basis is not None
        and rev18_stale_run.policy_basis is not None
        and rev17_run.policy_basis.derivation_snapshot_id
        != rev18_stale_run.policy_basis.derivation_snapshot_id
    )
    rev18_stale_proof = build_scenario_proof(
        scenario_id=FlagshipScenarioIdV1.REV18_STALE_SECURITY,
        run=rev18_stale_run,
        llm_calls=scenario.llm.calls,
        passed=rev18_stale_passed,
        detail=f"security_reason={stale_security.reason_code}",
    )
    proofs.append(rev18_stale_proof)

    reloaded_failure = scenario.reload_run(run_id="flagship-rev18-stale")
    reload_failure_ok = (
        scenario.evaluation_for_suffix(reloaded_failure, "security").reason_code
        is RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID
    )
    proofs[-1] = rev18_stale_proof.model_copy(
        update={
            "passed": rev18_stale_proof.passed and reload_failure_ok,
            "detail": rev18_stale_proof.detail + f"; reload_failure={reload_failure_ok}",
        }
    )

    scenario.refresh_security_evidence(updated_at=_FRESH_SECURITY_UPDATED_AT)
    scenario.reset_vendor_read_counts()
    rev18_fresh_run = await scenario.ask(
        run_id="flagship-rev18-fresh",
        request_id="request-flagship-rev18-fresh",
    )
    rev18_fresh_passed = (
        FlagshipDockerScenarioV1.is_satisfied(rev18_fresh_run)
        and scenario.llm.calls == 1
    )
    proofs.append(
        build_scenario_proof(
            scenario_id=FlagshipScenarioIdV1.REV18_FRESH_SECURITY,
            run=rev18_fresh_run,
            llm_calls=scenario.llm.calls,
            passed=rev18_fresh_passed,
            detail="fresh security evidence under rev18",
        )
    )

    scenario.set_policy_revision(policy_revision=FLAGSHIP_POLICY_REV_17)
    scenario.use_revoking_orchestrator()
    scenario.restore_active_bindings()
    scenario.reset_vendor_read_counts()
    revoked_run = await scenario.ask(
        run_id="flagship-authority-revoked",
        request_id="request-flagship-authority-revoked",
    )
    architecture_eval = scenario.evaluation_for_suffix(revoked_run, "architecture")
    counts = scenario.vendor_read_counts()
    revoked_passed = (
        FlagshipDockerScenarioV1.is_unsatisfied(revoked_run)
        and scenario.llm.calls == 0
        and counts["governance"] == 0
        and architecture_eval.reason_code
        is RequirementAdmissibilityReasonCodeV1.AUTHORITY_UNAVAILABLE
    )
    proofs.append(
        build_scenario_proof(
            scenario_id=FlagshipScenarioIdV1.AUTHORITY_REVOKED,
            run=revoked_run,
            llm_calls=scenario.llm.calls,
            passed=revoked_passed,
            detail=f"governance_http={counts['governance']}; reason={architecture_eval.reason_code}",
        )
    )

    scenario.use_inner_orchestrator()
    scenario.restore_active_bindings()
    scenario.recover_security_provider()
    scenario.fail_security_provider()
    scenario.reset_vendor_read_counts()
    failed_run = await scenario.ask(
        run_id="flagship-provider-503",
        request_id="request-flagship-provider-503",
    )
    security_eval = scenario.evaluation_for_suffix(failed_run, "security")
    security_failure = live_call_failure_for_suffix(failed_run, "security-read")
    counts = scenario.vendor_read_counts()
    provider_503_passed = (
        FlagshipDockerScenarioV1.is_unsatisfied(failed_run)
        and scenario.llm.calls == 0
        and counts["security"] == 1
        and security_eval.reason_code is RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED
        and security_failure is not None
        and security_failure.call_id.endswith(":security-read")
    )
    proofs.append(
        build_scenario_proof(
            scenario_id=FlagshipScenarioIdV1.PROVIDER_503,
            run=failed_run,
            llm_calls=scenario.llm.calls,
            passed=provider_503_passed,
            detail=f"security_http={counts['security']}; call_id={security_failure.call_id if security_failure else 'none'}",
        )
    )

    scenario.recover_security_provider()
    scenario.malformed_security_provider()
    scenario.reset_vendor_read_counts()
    malformed_run = await scenario.ask(
        run_id="flagship-malformed",
        request_id="request-flagship-malformed",
    )
    malformed_eval = scenario.evaluation_for_suffix(malformed_run, "security")
    malformed_passed = (
        FlagshipDockerScenarioV1.is_unsatisfied(malformed_run)
        and scenario.llm.calls == 0
        and malformed_eval.reason_code
        is RequirementAdmissibilityReasonCodeV1.PROVIDER_RESPONSE_INVALID
    )
    proofs.append(
        build_scenario_proof(
            scenario_id=FlagshipScenarioIdV1.MALFORMED_RESPONSE,
            run=malformed_run,
            llm_calls=scenario.llm.calls,
            passed=malformed_passed,
            detail=f"reason={malformed_eval.reason_code}",
        )
    )

    scenario.recover_security_provider()
    scenario.set_policy_revision(policy_revision=FLAGSHIP_POLICY_REV_17)
    scenario.reset_vendor_read_counts()
    environment.restart_project_vendor()
    restart_run = await scenario.ask(
        run_id="flagship-vendor-restart",
        request_id="request-flagship-vendor-restart",
    )
    restart_passed = (
        FlagshipDockerScenarioV1.is_satisfied(restart_run)
        and scenario.llm.calls == 1
        and scenario.evaluation_for_suffix(restart_run, "readiness").status
        is RequirementEvaluationStatusV1.SATISFIED
    )
    proofs.append(
        build_scenario_proof(
            scenario_id=FlagshipScenarioIdV1.VENDOR_RESTART,
            run=restart_run,
            llm_calls=scenario.llm.calls,
            passed=restart_passed,
            detail="reseed=NO; same external record through integration=YES",
        )
    )

    history = build_history_comparison(rev17=rev17_proof, rev18=rev18_stale_proof)
    proofs.append(
        FlagshipScenarioProofV1(
            scenario_id=FlagshipScenarioIdV1.STRUCTURAL_HISTORY,
            passed=True,
            detail="history comparison generated",
            llm_calls=0,
            run_id="flagship-structural-history",
        )
    )

    summary_rows = (
        FlagshipSummaryRowV1(
            scenario_label="rev17 valid",
            evidence="OK",
            authority="OK",
            temporal="OK",
            result=summary_label_for_admissibility(rev17_proof.overall_admissibility),
            llm=rev17_proof.llm_calls,
        ),
        FlagshipSummaryRowV1(
            scenario_label="rev18 stale",
            evidence="OK",
            authority="OK",
            temporal="STALE",
            result=summary_label_for_admissibility(rev18_stale_proof.overall_admissibility),
            llm=rev18_stale_proof.llm_calls,
        ),
        FlagshipSummaryRowV1(
            scenario_label="rev18 fresh",
            evidence="OK",
            authority="OK",
            temporal="OK",
            result=summary_label_for_admissibility(
                proofs[2].overall_admissibility,
            ),
            llm=proofs[2].llm_calls,
        ),
        FlagshipSummaryRowV1(
            scenario_label="authority revoked",
            evidence="partial",
            authority="DENIED",
            temporal="—",
            result=summary_label_for_admissibility(proofs[3].overall_admissibility),
            llm=proofs[3].llm_calls,
        ),
        FlagshipSummaryRowV1(
            scenario_label="provider 503",
            evidence="partial",
            authority="OK",
            temporal="—",
            result=summary_label_for_admissibility(proofs[4].overall_admissibility),
            llm=proofs[4].llm_calls,
        ),
    )

    all_passed = all(proof.passed for proof in proofs)
    return AdvancedFlagshipProofResultV1(
        flagship_question=FLAGSHIP_QUESTION,
        scenarios=tuple(proofs),
        summary_rows=summary_rows,
        distinct_providers=len(providers),
        distinct_connections=len(connections),
        distinct_capabilities=len(capabilities),
        distinct_call_ids=len(call_ids),
        all_passed=all_passed,
        history_comparison=history,
    )


def _print_result(result: AdvancedFlagshipProofResultV1) -> None:
    print(_BANNER)
    print("ADVANCED FLAGSHIP PROOF — COMM-5F3-F")
    print(f"FLAGSHIP QUESTION: {result.flagship_question}")
    print(_BANNER)
    print("SUMMARY")
    print("| Scenario | Evidence | Authority | Temporal | Result | LLM |")
    print("|----------|----------|-----------|----------|--------|-----|")
    for row in result.summary_rows:
        print(
            f"| {row.scenario_label} | {row.evidence} | {row.authority} | "
            f"{row.temporal} | {row.result} | {row.llm} |"
        )
    print(_BANNER)
    for proof in result.scenarios:
        if proof.scenario_id is FlagshipScenarioIdV1.STRUCTURAL_HISTORY:
            print(result.history_comparison)
            continue
        print(format_scenario_section(proof))
        print(_BANNER)
    print(
        f"DISTINCT PROVIDERS: {result.distinct_providers} "
        f"({'PASS' if result.distinct_providers == 4 else 'FAIL'})"
    )
    print(
        f"DISTINCT CONNECTIONS: {result.distinct_connections} "
        f"({'PASS' if result.distinct_connections == 4 else 'FAIL'})"
    )
    print(
        f"DISTINCT CAPABILITIES: {result.distinct_capabilities} "
        f"({'PASS' if result.distinct_capabilities == 4 else 'FAIL'})"
    )
    print(
        f"DISTINCT CALL IDS: {result.distinct_call_ids} "
        f"({'PASS' if result.distinct_call_ids == 4 else 'FAIL'})"
    )
    print(
        "ADVANCED FLAGSHIP: "
        f"{'ALL SCENARIOS PASS' if result.all_passed else 'FAILURES DETECTED'}"
    )


def main() -> int:
    try:
        result = asyncio.run(_run_scenarios())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL advanced_flagship_proof: {exc}")
        environment = AdvancedFlagshipDockerEnvironmentV1.from_defaults()
        print(f"Required compose: {environment.compose_up_hint()}")
        return 1
    _print_result(result)
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
