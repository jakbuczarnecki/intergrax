#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Controlled LKW Sentry observability proof helper (LKW-OBS-SENTRY-0)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

_FORBIDDEN_EXPORT_SAMPLES = (
    "secret prompt",
    "raw body",
    "secret-api-key",
    "tool_arguments",
    "raw_chunks",
    "/home/user/",
    "c:\\users\\",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a controlled LKW problem signal through the platform Sentry integration.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id for the proof event (generated when omitted).",
    )
    parser.add_argument(
        "--correlation-id",
        default="",
        help="Optional correlation id for the proof event (generated when omitted).",
    )
    return parser.parse_args()


async def _run_proof(*, run_id: str, correlation_id: str) -> int:
    from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
    from intergrax.runtime.observability.problem_reporter import ProblemReportContext, report_problem
    from intergrax.runtime.observability.sentry_export_wiring import (
        build_sentry_observability_integration,
    )
    from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()
    if config is None or not config.enabled:
        print("proof_result=FAIL")
        print("reason=observability export is disabled")
        return 1
    if config.backend_id != "sentry":
        print("proof_result=FAIL")
        print(f"reason=expected backend sentry, got {config.backend_id}")
        return 1

    integration = build_sentry_observability_integration(config)
    context = ProblemReportContext(
        run_id=run_id,
        agent_id=settings.default_agent_id,
        capability="local.workspace.proof",
        correlation_id=correlation_id,
    )
    result = await report_problem(
        context=context,
        problem_kind="lkw.proof_controlled_failure",
        error_code="LKW_PROOF_CONTROLLED_FAILURE",
        source_layer="lkw",
        source_component="sentry_proof",
        exporter=integration,
        policy=ObservabilityExportPolicy(enabled=True, export_content=False),
    )

    if not result.exported or result.envelope is None:
        print("proof_result=FAIL")
        print("reason=problem export did not complete")
        return 1

    serialized = result.envelope.model_dump_json()
    safety_passed = all(sample not in serialized.lower() for sample in _FORBIDDEN_EXPORT_SAMPLES)
    if not safety_passed:
        print("proof_result=FAIL")
        print("safety_check=failed")
        return 1

    print("proof_result=PASS")
    print("backend=sentry")
    print("problem_kind=lkw.proof_controlled_failure")
    print("problem_error_code=LKW_PROOF_CONTROLLED_FAILURE")
    print(f"run_id={run_id}")
    print(f"correlation_id={correlation_id}")
    print("sentry_event_sent=true")
    print("safety_check=passed")
    return 0


def main() -> int:
    args = _parse_args()
    run_id = args.run_id.strip() or f"lkw-sentry-proof-{uuid.uuid4().hex[:12]}"
    correlation_id = args.correlation_id.strip() or f"corr-{uuid.uuid4().hex[:12]}"
    try:
        return asyncio.run(_run_proof(run_id=run_id, correlation_id=correlation_id))
    except ValueError as exc:
        print("proof_result=FAIL")
        print(f"reason={exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - proof helper surfaces operator-safe failure
        print("proof_result=FAIL")
        print(f"reason={type(exc).__name__}")
        _ = json.dumps({"error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    sys.exit(main())
