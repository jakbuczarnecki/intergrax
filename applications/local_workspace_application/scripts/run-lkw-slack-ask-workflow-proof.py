# © Artur Czarnecki. All rights reserved.

"""Proof harness checklist for LKW-SLACK-WORKFLOW-1A (not LIVE_VERIFIED by default).

Usage (after configuring real Slack + LKW Ask):

  uv run python applications/local_workspace_application/scripts/run-lkw-slack-ask-workflow-proof.py

This script never prints tokens, API keys, excerpts, or local absolute paths.
It only prints a structured checklist for a later live PASS recording.
"""

from __future__ import annotations

import os
import sys


def _present(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def main() -> int:
    required = [
        "LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED",
        "LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID",
        "LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID",
        "LOCAL_WORKSPACE_SLACK_TENANT_ID",
        "LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID",
        "LOCAL_WORKSPACE_SLACK_ASK_BASE_URL",
        "INTERGRAX_SLACK_APP_TOKEN",
        "INTERGRAX_SLACK_BOT_TOKEN",
        "INTERGRAX_SLACK_CONVERSATION_ENABLED",
    ]
    missing = [name for name in required if not _present(name)]
    print("slice=LKW-SLACK-WORKFLOW-1A")
    print("live_verified=no")
    print("summary=NOT_RUN")
    print("checklist:")
    for item in (
        "real approved Slack DM",
        "resolved tenant_id",
        "configured workspace_id",
        "Ask run_id",
        "typed Ask status",
        "threaded final response",
        "safe source labels",
        "duplicate event suppressed",
        "clean shutdown",
    ):
        print(f"  - {item}: pending")
    if missing:
        print("config_ready=false")
        print(f"missing_env_count={len(missing)}")
        # Names only — never values.
        print("missing_env=" + ",".join(missing))
        return 2
    print("config_ready=true")
    print(
        "note=Start the LKW host with Slack companion enabled, send one approved DM, "
        "confirm threaded ack+answer, redeliver/duplicate once, then shut down cleanly. "
        "Record evidence under applications/local_workspace_application/docs/proof/ "
        "only after a real Slack + Ask execution."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
