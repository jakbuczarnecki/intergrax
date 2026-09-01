"""Thin scenario proof runner — configure, invoke application, evaluate, write artifacts."""

from __future__ import annotations

import argparse
import asyncio
import sys

from platform_proofs.scenarios.verified_product_identification.application.runtime_composition import SYNTHETIC_SCENARIO_TENANT_ID
from platform_proofs.scenarios.verified_product_identification.application.scenario import execute_scenario
from platform_proofs.scenarios.verified_product_identification.proof.evaluator import evaluate_scenario_run


async def _run() -> int:
    # TODO: wire configuration, invoke application, collect evidence, evaluate.
    # Critic/HITL/RAG/web/memory/hosting are opt-in via ApplicationEnvironmentProfile.
    raise NotImplementedError("Implement proof runner workflow.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scenario proof runner.")
    parser.add_argument("--validate-only", action="store_true")
    _ = parser.parse_args(argv)
    try:
        return asyncio.run(_run())
    except NotImplementedError:
        print("Proof runner not yet implemented.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
