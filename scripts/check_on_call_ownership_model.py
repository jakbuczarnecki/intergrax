#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-30.3 — on-call ownership model for production components."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "agents"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.on_call_ownership_wiring import resolve_on_call_ownership_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from legal.legal_agent import LegalAgent
from research.research_agent import ResearchAgent


_REFERENCE_AGENTS: tuple[Callable[[], object], ...] = (
    EchoAgent,
    LegalAgent,
    ResearchAgent,
)


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    contracts = tuple(factory().get_contract() for factory in _REFERENCE_AGENTS)
    registry = resolve_on_call_ownership_registry(env, contracts=contracts)
    if not registry.enabled:
        print("product host must enable on-call ownership registry", file=sys.stderr)
        return 1

    errors: list[str] = []
    for record in registry.records:
        contract = next(item for item in contracts if item.id == record.agent_id)
        if contract.production_eligible and not record.on_call.strip():
            errors.append(f"{record.agent_id}: missing on_call contact")
        if contract.production_eligible and not record.approved:
            errors.append(f"{record.agent_id}: ownership evaluation not approved")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"OK: on-call ownership model ({len(registry.records)} agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
