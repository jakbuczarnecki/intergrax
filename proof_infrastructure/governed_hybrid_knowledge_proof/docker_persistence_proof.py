# © Artur Czarnecki. All rights reserved.

"""Docker-backed vendor persistence proof for COMM-5 F3-E-R1."""

from __future__ import annotations

import asyncio
import sys

from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.docker_environment import (
    GovernedHybridDockerEnvironmentV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.security_docker_scenario import (
    build_governed_security_docker_scenario,
)


async def _run_scenarios() -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    def record(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status} {name}: {detail}")

    environment = GovernedHybridDockerEnvironmentV1.from_defaults()
    environment.ensure_ready()
    scenario = await build_governed_security_docker_scenario(environment)

    seeded = scenario.seed_baseline()
    baseline = await scenario.ask(
        run_id="docker-r1-baseline",
        request_id="request-docker-r1-baseline",
    )
    baseline_security = scenario.security_evaluation(baseline)
    record(
        "scenario_1_persisted_baseline",
        baseline.status is AskRunStatus.COMPLETED
        and baseline.evidence_admissibility is not None
        and baseline.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
        and baseline_security.status is RequirementEvaluationStatusV1.SATISFIED
        and scenario.vendor_read_count() == 1
        and scenario.llm.calls == 1,
        (
            f"status={baseline.status.value}, "
            f"requirement={baseline_security.status.value}, "
            f"llm={scenario.llm.calls}, "
            f"vendor_reads={scenario.vendor_read_count()}, "
            f"seeded_status={seeded.status}"
        ),
    )

    scenario.fail_provider()
    failed = await scenario.ask(
        run_id="docker-r1-provider-failure",
        request_id="request-docker-r1-provider-failure",
    )
    failed_security = scenario.security_evaluation(failed)
    security_call_id = (
        failed.live_call_failures[0].call_id if failed.live_call_failures else ""
    )
    record(
        "scenario_2_real_provider_failure",
        failed.status is AskRunStatus.INSUFFICIENT_EVIDENCE
        and scenario.llm.calls == 0
        and scenario.vendor_read_count() == 1
        and failed_security.reason_code is RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED
        and len(failed.live_call_failures) == 1
        and security_call_id.endswith(":security-read"),
        (
            f"status={failed.status.value}, "
            f"reason={failed_security.reason_code.value if failed_security.reason_code else 'none'}, "
            f"call_id={security_call_id}, "
            f"llm={scenario.llm.calls}, "
            f"vendor_reads={scenario.vendor_read_count()}"
        ),
    )

    reloaded = scenario.reload_run(run_id="docker-r1-provider-failure")
    reloaded_security = scenario.security_evaluation(reloaded)
    reloaded_call_id = (
        reloaded.live_call_failures[0].call_id if reloaded.live_call_failures else ""
    )
    record(
        "scenario_2_run_reload",
        reloaded_security.reason_code is RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED
        and len(reloaded.live_call_failures) == 1
        and reloaded_call_id == security_call_id,
        (
            f"reason={reloaded_security.reason_code.value if reloaded_security.reason_code else 'none'}, "
            f"call_id={reloaded_call_id}"
        ),
    )

    scenario.recover_provider()
    recovered = await scenario.ask(
        run_id="docker-r1-recovery",
        request_id="request-docker-r1-recovery",
    )
    recovered_security = scenario.security_evaluation(recovered)
    record(
        "scenario_3_service_recovery",
        recovered.status is AskRunStatus.COMPLETED
        and scenario.llm.calls == 1
        and recovered_security.status is RequirementEvaluationStatusV1.SATISFIED
        and scenario.assert_same_fixture(recovered),
        (
            f"status={recovered.status.value}, "
            f"requirement={recovered_security.status.value}, "
            f"llm={scenario.llm.calls}, "
            f"same_record={scenario.assert_same_fixture(recovered)}"
        ),
    )

    environment.restart_security_vendor()
    persisted = await scenario.ask(
        run_id="docker-r1-vendor-restart",
        request_id="request-docker-r1-vendor-restart",
    )
    persisted_security = scenario.security_evaluation(persisted)
    record(
        "scenario_4_vendor_process_restart",
        persisted.status is AskRunStatus.COMPLETED
        and scenario.llm.calls == 1
        and persisted_security.status is RequirementEvaluationStatusV1.SATISFIED
        and scenario.assert_same_fixture(persisted),
        (
            "VENDOR PROCESS RESTARTED: YES; "
            "MONGODB CONTAINER/VOLUME PRESERVED: YES; "
            "RESEED AFTER RESTART: NO; "
            f"RECORD STILL AVAILABLE THROUGH INTERGRAX: "
            f"{'YES' if scenario.assert_same_fixture(persisted) else 'NO'}"
        ),
    )

    return ok, lines


def main() -> int:
    try:
        ok, lines = asyncio.run(_run_scenarios())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL docker_persistence_proof: {exc}")
        return 1
    for line in lines:
        print(line)
    print("PASS docker_persistence_proof" if ok else "FAIL docker_persistence_proof")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
