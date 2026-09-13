# © Artur Czarnecki. All rights reserved.

"""Container worker for DS-E2E-15J Docker system qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testing_support.decision_e2e.docker_system_scenarios import (
    run_docker_system_scenario,
    scenario_result_to_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DS-E2E-15J Docker system qualification worker",
    )
    parser.add_argument(
        "scenario",
        choices=(
            "startup-health",
            "flow-success",
            "governance-deny",
            "governance-approval",
            "evidence-chain",
            "execution-failure",
            "missing-governance",
            "invalid-config-startup",
            "plugin-compatibility",
        ),
    )
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    outcome = run_docker_system_scenario(args.scenario)
    payload = scenario_result_to_dict(outcome)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
