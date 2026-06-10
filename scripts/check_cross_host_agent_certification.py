#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-18.2 — cross-host agent reuse certification suite."""

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
from intergrax.applications._shared.cross_host_agent_certification import certify_agent_across_hosts
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from legal.legal_agent import LegalAgent
from research.research_agent import ResearchAgent

_REFERENCE_AGENTS: tuple[Callable[[], object], ...] = (
    EchoAgent,
    LegalAgent,
    ResearchAgent,
)


def main() -> int:
    environments = (
        ApplicationEnvironmentProfile.lab_defaults(),
        ApplicationEnvironmentProfile.product_defaults(),
    )
    errors: list[str] = []
    for factory in _REFERENCE_AGENTS:
        contract = factory().get_contract()
        report = certify_agent_across_hosts(contract, environments=environments)
        if not report.passed:
            for item in report.results:
                if not item.passed:
                    errors.extend(f"{item.agent_id}@{item.host_profile_id}: {err}" for err in item.errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"OK: cross-host agent certification ({len(_REFERENCE_AGENTS)} agents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
