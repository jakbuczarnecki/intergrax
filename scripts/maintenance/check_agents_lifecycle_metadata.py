#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ensure harness reference agents declare lifecycle metadata (FAUDIT-ALG.1 adoption)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "agents", ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from echo.echo_agent import EchoAgent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_assembly_resolver import validate_agent_assembly
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


def main() -> int:
    errors: list[str] = []
    for factory in _REFERENCE_AGENTS:
        contract: AgentContract = factory().get_contract()
        if not (contract.owner_team or "").strip():
            errors.append(f"{contract.id}: missing owner_team")
        if contract.production_eligible and not (contract.on_call_contact or contract.owner_contact or "").strip():
            errors.append(f"{contract.id}: missing on_call_contact for production_eligible agent")
        if contract.production_eligible and not (contract.modality_profile_id or "").strip():
            errors.append(f"{contract.id}: missing modality_profile_id for production_eligible agent")
        result = validate_agent_assembly(contract)
        if not result.valid:
            errors.extend(f"{contract.id}: {err}" for err in result.errors)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print(f"agents lifecycle metadata: ok ({len(_REFERENCE_AGENTS)} agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
