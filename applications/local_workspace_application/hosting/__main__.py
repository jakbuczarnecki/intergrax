# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint: python -m local_workspace_application.hosting (APP-HOST-8C)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.hosting import (
    HostedApplicationExitKind,
    HostedApplicationSupervisorResult,
)
from local_workspace_application.hosting.foreground import (
    run_local_workspace_hosted_application,
)


def _safe_result_payload(result: HostedApplicationSupervisorResult) -> dict[str, Any]:
    return {
        "schema_version": "local_workspace.hosted_process_result.v1",
        "application_id": result.application_id,
        "profile_digest": result.profile_digest,
        "definition_digest": result.definition_digest,
        "final_exit": result.final_exit.model_dump(mode="json"),
        "attempts": [
            {
                "attempt_number": attempt.attempt_number,
                "instance_id": attempt.instance_id,
                "exit_kind": (
                    attempt.exit_record.exit_kind.value
                    if attempt.exit_record is not None
                    else ""
                ),
                "reason_code": (
                    attempt.exit_record.reason_code
                    if attempt.exit_record is not None
                    else ""
                ),
                "cleanup_verified": attempt.cleanup_verified,
            }
            for attempt in result.attempts
        ],
        "restart_exhausted": result.restart_exhausted,
    }


def _exit_code(result: HostedApplicationSupervisorResult) -> int:
    kind = result.final_exit.exit_kind
    if kind is HostedApplicationExitKind.CLEAN_STOP:
        return 0
    if kind is HostedApplicationExitKind.INSTANCE_CONFLICT:
        return 2
    return 1


def main() -> int:
    result = run_local_workspace_hosted_application()
    print(
        json.dumps(
            _safe_result_payload(result),
            sort_keys=True,
        ),
        flush=True,
    )
    return _exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
