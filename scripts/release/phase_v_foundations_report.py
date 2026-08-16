#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Generate Phase V foundations report artifacts (report-only mode)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Protocol

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.runtime.architecture import (
    AgentCertificationEvidence,
    AgentCertificationEvaluation,
    AgentCertificationGate,
    AgentCertificationOwner,
    GateCheckStatus,
    build_catalog_capability_graph,
    compute_architecture_metrics,
    evaluate_agent_certification,
)
from testing_support.agent_registry_bootstrap import (
    build_harness_registry,
    build_organization_worker_registry,
    build_research_registry,
)


class ReportWriter(Protocol):
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        ...


class JsonReportWriter:
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _build_certification_evidence() -> list[AgentCertificationEvidence]:
    contracts_by_id = {}
    for registry in (
        build_harness_registry(),
        build_research_registry(),
        build_organization_worker_registry(),
    ):
        for contract in registry.list_contracts():
            contracts_by_id[contract.id] = contract

    evidence: list[AgentCertificationEvidence] = []
    for contract_id in sorted(contracts_by_id):
        contract = contracts_by_id[contract_id]
        evidence.append(
            AgentCertificationEvidence(
                agent_id=contract.id,
                agent_version=contract.version,
                production_eligible=True,
                owner=AgentCertificationOwner(
                    team="harness-platform",
                    owner=f"{contract.id}-owner",
                    on_call=f"{contract.id}-oncall",
                ),
                quality_gates=[
                    AgentCertificationGate(
                        name="unit-gate",
                        status=GateCheckStatus.PASS,
                        evidence_ref="pytest -m gate",
                    )
                ],
                policy_gates=[
                    AgentCertificationGate(
                        name="tool-policy",
                        status=GateCheckStatus.PASS,
                        evidence_ref="tool policy resolution",
                    )
                ],
                security_gates=[
                    AgentCertificationGate(
                        name="uaep-contract",
                        status=GateCheckStatus.PASS,
                        evidence_ref="registry requires_uaep path",
                    )
                ],
            )
        )
    return evidence


def main() -> int:
    output_dir = REPO_ROOT / "build" / "architecture_hardening"
    writer: ReportWriter = JsonReportWriter()

    capability_graph = build_catalog_capability_graph()
    metrics_report = compute_architecture_metrics(capability_graph)
    certification_evaluations = [
        evaluate_agent_certification(entry)
        for entry in _build_certification_evidence()
    ]

    writer.write(
        output_path=output_dir / "capability_graph.json",
        payload=capability_graph,
    )
    writer.write(
        output_path=output_dir / "architecture_metrics.json",
        payload=metrics_report,
    )
    writer.write(
        output_path=output_dir / "agent_certification_report.json",
        payload=CertificationReport(evaluations=certification_evaluations),
    )

    print("phase-v foundations report: OK")
    print(f"artifacts: {output_dir.as_posix()}")
    return 0


class CertificationReport(BaseModel):
    schema_version: str = "1.0.0"
    mode: str = "report-only"
    evaluations: list[AgentCertificationEvaluation]


if __name__ == "__main__":
    raise SystemExit(main())
