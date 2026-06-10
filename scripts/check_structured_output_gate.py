#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-6.1 — structured output validation on reference + certified agents."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "agents", ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from echo.echo_agent import EchoAgent
from intergrax.contracts.agent_contract_meta import AgentContract
from legal.legal_agent import LegalAgent
from organization_worker.organization_worker_agent import OrganizationWorkerAgent
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from signoff_probe.signoff_probe_agent import SignoffProbeAgent

_REFERENCE_AGENTS: tuple[Callable[[], object], ...] = (
    EchoAgent,
    LegalAgent,
    ResearchAgent,
    SummaryAgent,
    SignoffProbeAgent,
    OrganizationWorkerAgent,
)


def _has_structured_output(contract: AgentContract) -> bool:
    if "structured_output" in contract.validation_rules:
        return True
    return contract.output_schema is not None and bool(contract.output_schema)


def main() -> int:
    errors: list[str] = []
    for factory in _REFERENCE_AGENTS:
        contract: AgentContract = factory().get_contract()
        if contract.production_eligible and not _has_structured_output(contract):
            errors.append(f"{contract.id}: production_eligible agent missing structured output contract")
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print(f"OK: structured output gate ({len(_REFERENCE_AGENTS)} reference agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
