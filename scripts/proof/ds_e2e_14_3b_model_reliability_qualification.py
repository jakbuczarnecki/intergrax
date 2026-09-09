# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DS-E2E-14.3b live model reliability qualification entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from testing_support.decision_e2e.ai_incident_qualification_run import (
    CANONICAL_SCENARIO_INPUT_IDENTITY,
    execute_ai_incident_qualification_run,
)
from testing_support.decision_e2e.env_bootstrap import bootstrap_qualification_environment
from testing_support.decision_e2e.reliability_qualification import (
    CallableDecisionQualificationRunExecutor,
    DecisionReliabilityQualificationPlan,
    execute_reliability_qualification,
)
from testing_support.decision_e2e.reliability_reporting import write_qualification_artifacts
from testing_support.decision_e2e.scenario_qualification import AI_INCIDENT_SCENARIO_ID


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_RUN_COUNT = 20
DEFAULT_OUTPUT = _REPO_ROOT / ".tmp" / "session" / "DS-E2E-14.3b"


def _resolve_git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


async def _run_qualification(
    *,
    run_count: int,
    provider: str,
    model: str,
    output_dir: Path,
) -> int:
    os.environ["INTERGRAX_LLM_PROVIDER"] = provider
    os.environ["INTERGRAX_LLM_MODEL"] = model
    os.environ["INTERGRAX_DECISION_E2E_QUALIFICATION"] = "1"

    bootstrap = bootstrap_qualification_environment(start_path=_REPO_ROOT)
    plan = DecisionReliabilityQualificationPlan(
        run_count=run_count,
        provider_id=provider,
        model_id=model,
        scenario_id=AI_INCIDENT_SCENARIO_ID,
        scenario_input_identity=CANONICAL_SCENARIO_INPUT_IDENTITY,
    )
    executor = CallableDecisionQualificationRunExecutor(
        _callable=lambda run_index: execute_ai_incident_qualification_run(run_index=run_index),
    )
    result = await execute_reliability_qualification(
        plan,
        executor,
        git_sha=_resolve_git_sha(),
        env_bootstrap=bootstrap,
    )
    write_qualification_artifacts(result, output_dir)

    if not result.session_complete:
        print("QUALIFICATION INCOMPLETE")
        print(f"valid_model_trials={result.valid_model_trial_count}/{plan.run_count}")
        return 2
    print("QUALIFICATION COMPLETE")
    print(
        f"platform={result.summary.platform_pass_count}/{result.summary.total_runs} "
        f"model={result.summary.model_pass_count}/{result.summary.total_runs}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DS-E2E-14.3b model reliability qualification")
    parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return asyncio.run(
        _run_qualification(
            run_count=args.run_count,
            provider=args.provider,
            model=args.model,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
