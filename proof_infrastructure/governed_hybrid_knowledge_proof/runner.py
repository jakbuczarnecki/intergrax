# © Artur Czarnecki. All rights reserved.

"""Four-scenario flagship runner for governed hybrid knowledge proof."""

from __future__ import annotations

import asyncio
import httpx

from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    EvidenceTypeV1,
    PersistedIndexedEvidenceV2,
    PersistedLiveEvidenceProvenanceV2,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
)
from proof_infrastructure.controlled_project_status_service.lifecycle import (
    ControlledProjectStatusServer,
)
from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
    seed_orion_fixture,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.fixtures import (
    DEPLOYMENT_POLICY_CONTENT,
    ORION_DEPLOYMENT_QUESTION,
    PROOF_BINDING_ID,
    PROOF_CONNECTION_REF,
    PROOF_TENANT_ID,
    PROOF_WORKSPACE_ID,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.harness import (
    GovernedHybridKnowledgeHarness,
    build_harness,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.models import (
    FlagshipProofResultV1,
    FlagshipProofScenarioResultV1,
    FlagshipScenarioChecksV1,
    FlagshipScenarioMetricsV1,
    SemanticDecisionV1,
)

_BANNER = "=" * 60


def normalize_decision(*, answer: str | None, status: AskRunStatus) -> SemanticDecisionV1:
    if status is AskRunStatus.INSUFFICIENT_EVIDENCE or answer is None:
        return SemanticDecisionV1.CANNOT_DETERMINE
    normalized = answer.strip().upper()
    if normalized.startswith("YES"):
        return SemanticDecisionV1.YES
    if normalized.startswith("NO"):
        return SemanticDecisionV1.NO
    return SemanticDecisionV1.CANNOT_DETERMINE


def _assert_ephemeral_live_body_not_persisted(
    persisted_evidence: tuple[PersistedIndexedEvidenceV2 | PersistedLiveEvidenceProvenanceV2, ...],
) -> bool:
    for item in persisted_evidence:
        if item.evidence_type is EvidenceTypeV1.LIVE:
            if not isinstance(item, PersistedLiveEvidenceProvenanceV2):
                return False
    return True


def _scenario_checks(
    *,
    name: str,
    expected: SemanticDecisionV1,
    observed: SemanticDecisionV1,
    http_reads: int | None,
    expected_http_reads: int | None,
    llm_calls: int | None,
    expected_llm_calls: int | None,
    admissibility: str | None,
    expected_admissibility: str | None,
    ask_status: str | None,
    expected_ask_status: str | None,
    extra_assertions: FlagshipScenarioChecksV1,
    run_id: str | None = None,
    key_metrics: FlagshipScenarioMetricsV1 | None = None,
) -> FlagshipProofScenarioResultV1:
    checks = FlagshipScenarioChecksV1(
        decision=observed == expected,
        http_reads=(
            http_reads == expected_http_reads
            if expected_http_reads is not None
            else True
        ),
        llm_calls=(
            llm_calls == expected_llm_calls if expected_llm_calls is not None else True
        ),
        admissibility=(
            admissibility == expected_admissibility
            if expected_admissibility is not None
            else True
        ),
        ask_status=(
            ask_status == expected_ask_status if expected_ask_status is not None else True
        ),
        indexed_evidence_present=extra_assertions.indexed_evidence_present,
        live_evidence_present=extra_assertions.live_evidence_present,
        ephemeral_body_not_durable=extra_assertions.ephemeral_body_not_durable,
        indexed_policy_present=extra_assertions.indexed_policy_present,
        same_configuration_revision=extra_assertions.same_configuration_revision,
        same_plan_policy_contract=extra_assertions.same_plan_policy_contract,
        binding_disabled_before_live=extra_assertions.binding_disabled_before_live,
        indexed_only_insufficient=extra_assertions.indexed_only_insufficient,
    )
    passed = all(
        [
            checks.decision,
            checks.http_reads,
            checks.llm_calls,
            checks.admissibility,
            checks.ask_status,
            checks.indexed_evidence_present if checks.indexed_evidence_present is not None else True,
            checks.live_evidence_present if checks.live_evidence_present is not None else True,
            checks.ephemeral_body_not_durable if checks.ephemeral_body_not_durable is not None else True,
            checks.indexed_policy_present if checks.indexed_policy_present is not None else True,
            checks.same_configuration_revision if checks.same_configuration_revision is not None else True,
            checks.same_plan_policy_contract if checks.same_plan_policy_contract is not None else True,
            checks.binding_disabled_before_live if checks.binding_disabled_before_live is not None else True,
            checks.indexed_only_insufficient if checks.indexed_only_insufficient is not None else True,
        ]
    )
    metrics = key_metrics or FlagshipScenarioMetricsV1()
    metrics = metrics.model_copy(update={"checks": checks})
    return FlagshipProofScenarioResultV1(
        name=name,
        passed=passed,
        expected=expected,
        observed=observed,
        http_read_count=http_reads,
        llm_call_count=llm_calls,
        admissibility=admissibility,
        ask_status=ask_status,
        run_id=run_id,
        key_metrics=metrics,
    )


async def _run_ask(harness: GovernedHybridKnowledgeHarness, *, run_id: str) -> tuple[
    object,
    int,
    int,
]:
    harness.reset_http_counter()
    llm_before = harness.llm.calls
    command = harness.build_command(run_id=run_id)
    run = await harness.service.ask(command)
    return run, harness.http_read_count(), harness.llm.calls - llm_before


def _close_blocker(server: ControlledProjectStatusServer) -> None:
    response = httpx.put(
        f"{server.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
        timeout=2.0,
    )
    response.raise_for_status()


async def _run_all_scenarios(
    server: ControlledProjectStatusServer,
) -> FlagshipProofResultV1:
    seed_orion_fixture(server.store)
    harness = await build_harness(server=server)

    scenario_1_run_id = harness.next_run_id()
    run_1, http_1, llm_1 = await _run_ask(harness, run_id=scenario_1_run_id)
    decision_1 = normalize_decision(answer=run_1.answer, status=run_1.status)
    indexed_present = any(
        item.evidence_type is EvidenceTypeV1.INDEXED for item in run_1.persisted_evidence
    )
    live_present = any(
        isinstance(item, PersistedLiveEvidenceProvenanceV2)
        for item in run_1.persisted_evidence
    )
    indexed_policy_present = any(
        item.document_id == harness.indexed_stack.indexed_document_id
        and item.evidence_type is EvidenceTypeV1.INDEXED
        for item in run_1.persisted_evidence
        if isinstance(item, PersistedIndexedEvidenceV2)
    )
    scenario_1 = _scenario_checks(
        name="01 REALITY — current blocker changes the decision",
        expected=SemanticDecisionV1.NO,
        observed=decision_1,
        http_reads=http_1,
        expected_http_reads=1,
        llm_calls=llm_1,
        expected_llm_calls=1,
        admissibility=(
            run_1.evidence_admissibility.overall_status.value
            if run_1.evidence_admissibility is not None
            else None
        ),
        expected_admissibility=EvidenceAdmissibilityStatusV1.SATISFIED.value,
        ask_status=run_1.status.value,
        expected_ask_status=AskRunStatus.COMPLETED.value,
        extra_assertions=FlagshipScenarioChecksV1(
            indexed_evidence_present=indexed_present,
            live_evidence_present=live_present,
            ephemeral_body_not_durable=_assert_ephemeral_live_body_not_persisted(
                run_1.persisted_evidence
            ),
            indexed_policy_present=indexed_policy_present,
        ),
        run_id=run_1.run_id,
        key_metrics=FlagshipScenarioMetricsV1(
            policy_revision=run_1.configuration_revision,
            plan_id=run_1.plan_id,
        ),
    )

    _close_blocker(server)
    scenario_2_run_id = harness.next_run_id()
    run_2, http_2, llm_2 = await _run_ask(harness, run_id=scenario_2_run_id)
    decision_2 = normalize_decision(answer=run_2.answer, status=run_2.status)
    scenario_2 = _scenario_checks(
        name="02 FRESHNESS — reality changes, policy does not",
        expected=SemanticDecisionV1.YES,
        observed=decision_2,
        http_reads=http_2,
        expected_http_reads=1,
        llm_calls=llm_2,
        expected_llm_calls=1,
        admissibility=(
            run_2.evidence_admissibility.overall_status.value
            if run_2.evidence_admissibility is not None
            else None
        ),
        expected_admissibility=EvidenceAdmissibilityStatusV1.SATISFIED.value,
        ask_status=run_2.status.value,
        expected_ask_status=AskRunStatus.COMPLETED.value,
        extra_assertions=FlagshipScenarioChecksV1(
            same_configuration_revision=(
                run_2.configuration_revision == run_1.configuration_revision
            ),
            same_plan_policy_contract=run_2.question == ORION_DEPLOYMENT_QUESTION,
        ),
        run_id=run_2.run_id,
    )

    revoke_harness = await build_harness(server=server, revoke_after_indexed=True)
    scenario_3_run_id = revoke_harness.next_run_id()
    run_3, http_3, llm_3 = await _run_ask(revoke_harness, run_id=scenario_3_run_id)
    decision_3 = normalize_decision(answer=run_3.answer, status=run_3.status)
    disabled_configuration = revoke_harness.configuration_service.get_configuration(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
    )
    disabled_binding = next(
        (
            binding
            for binding in disabled_configuration.live_access_bindings
            if binding.live_access_binding_id == PROOF_BINDING_ID
        ),
        None,
    )
    scenario_3 = _scenario_checks(
        name="03 AUTHORITY — revoked means physically not called",
        expected=SemanticDecisionV1.CANNOT_DETERMINE,
        observed=decision_3,
        http_reads=http_3,
        expected_http_reads=0,
        llm_calls=llm_3,
        expected_llm_calls=0,
        admissibility=(
            run_3.evidence_admissibility.overall_status.value
            if run_3.evidence_admissibility is not None
            else None
        ),
        expected_admissibility=EvidenceAdmissibilityStatusV1.UNSATISFIED.value,
        ask_status=run_3.status.value,
        expected_ask_status=AskRunStatus.INSUFFICIENT_EVIDENCE.value,
        extra_assertions=FlagshipScenarioChecksV1(
            binding_disabled_before_live=(
                disabled_binding is not None
                and disabled_binding.status is LiveAccessBindingStatusV1.DISABLED
            ),
            indexed_only_insufficient=any(
                item.evidence_type is EvidenceTypeV1.INDEXED
                for item in run_3.persisted_evidence
            ),
        ),
        run_id=run_3.run_id,
        key_metrics=FlagshipScenarioMetricsV1(
            revoke_boundary="canonical_disable_after_indexed_retrieval",
            configuration_revision_before_disable=(
                revoke_harness.indexed_retriever.configuration_revision_before_disable
            ),
            configuration_revision_after_disable=(
                revoke_harness.indexed_retriever.configuration_revision_after_disable
            ),
        ),
    )

    historical = harness.service.get_run(
        tenant_id=run_1.tenant_id,
        run_id=scenario_1_run_id,
    )
    live_provenance = next(
        (
            item
            for item in historical.persisted_evidence
            if isinstance(item, PersistedLiveEvidenceProvenanceV2)
        ),
        None,
    )
    indexed_provenance = next(
        (
            item
            for item in historical.persisted_evidence
            if item.evidence_type is EvidenceTypeV1.INDEXED
        ),
        None,
    )
    history_ok = all(
        [
            historical.run_id == scenario_1_run_id,
            historical.status is AskRunStatus.COMPLETED,
            historical.question == ORION_DEPLOYMENT_QUESTION,
            historical.configuration_revision == run_1.configuration_revision,
            bool(historical.required_evidence_obligations),
            indexed_provenance is not None,
            live_provenance is not None,
            historical.evidence_admissibility is not None
            and historical.evidence_admissibility.overall_status
            is EvidenceAdmissibilityStatusV1.SATISFIED,
            normalize_decision(
                answer=historical.answer,
                status=historical.status,
            )
            is SemanticDecisionV1.NO,
            live_provenance.content_hash,
            live_provenance.call_id,
            live_provenance.live_access_binding_id,
        ]
    )
    scenario_4 = FlagshipProofScenarioResultV1(
        name="04 HISTORY — why was NO valid then?",
        passed=history_ok,
        expected=SemanticDecisionV1.NO,
        observed=normalize_decision(
            answer=historical.answer,
            status=historical.status,
        ),
        admissibility=(
            historical.evidence_admissibility.overall_status.value
            if historical.evidence_admissibility is not None
            else None
        ),
        ask_status=historical.status.value,
        run_id=historical.run_id,
        key_metrics=FlagshipScenarioMetricsV1(
            configuration_revision=historical.configuration_revision,
            plan_id=historical.plan_id,
            indexed_evidence_id=(
                indexed_provenance.evidence_id if indexed_provenance else None
            ),
            live_content_hash=(
                live_provenance.content_hash if live_provenance else None
            ),
            live_binding_id=(
                live_provenance.live_access_binding_id if live_provenance else None
            ),
            historical_live_body_retained=False,
            structural_provenance_only=True,
        ),
    )

    return FlagshipProofResultV1(
        scenario_1=scenario_1,
        scenario_2=scenario_2,
        scenario_3=scenario_3,
        scenario_4=scenario_4,
        overall_status="PASS"
        if all(
            scenario.passed
            for scenario in (scenario_1, scenario_2, scenario_3, scenario_4)
        )
        else "FAIL",
        scenario_1_run_id=scenario_1_run_id,
    )


def run_flagship_proof(
    *,
    emit_terminal: bool = True,
) -> FlagshipProofResultV1:
    server = ControlledProjectStatusServer.start()
    try:
        result = asyncio.run(_run_all_scenarios(server))
        if emit_terminal:
            _print_terminal(result)
        return result
    finally:
        server.stop()


def _print_terminal(result: FlagshipProofResultV1) -> None:
    print(_BANNER)
    print("INTERGRAX — GOVERNED HYBRID KNOWLEDGE PROOF")
    print(_BANNER)
    print()
    for scenario in (
        result.scenario_1,
        result.scenario_2,
        result.scenario_3,
        result.scenario_4,
    ):
        print(scenario.name)
        if scenario.http_read_count is not None:
            print(f"HTTP reads: {scenario.http_read_count}")
        if scenario.admissibility is not None:
            print(f"Admissibility: {scenario.admissibility.upper()}")
        if scenario.llm_call_count is not None and scenario.name.startswith("03"):
            print(f"LLM calls: {scenario.llm_call_count}")
        print(f"Decision: {scenario.observed}")
        print(f"RESULT: {'PASS' if scenario.passed else 'FAIL'}")
        if not scenario.passed and scenario.key_metrics.checks is not None:
            print(f"Failed checks: {scenario.key_metrics.checks.model_dump()}")
        print()
    print(_BANNER)
    print(f"{result.passed_count} / 4 PROOFS PASSED")
    print(_BANNER)
