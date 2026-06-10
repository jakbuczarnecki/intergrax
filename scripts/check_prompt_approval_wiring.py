#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-17.1 — prompt approval workflow on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.prompt_approval_wiring import resolve_prompt_approval_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_prompt_approval_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled or wiring.queue is None:
        print("product host must enable prompt approval workflow", file=sys.stderr)
        return 1

    record = wiring.queue.approve(
        prompt_id="nexus_task_planner",
        version=1,
        approver_id="platform@intergrax",
        change_ticket_ref="AUDIT-IDEAL-17.1",
    )
    if not wiring.queue.is_approved(record.prompt_id, record.version):
        print("approved prompt must be retrievable from queue", file=sys.stderr)
        return 1

    print("OK: prompt approval workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
