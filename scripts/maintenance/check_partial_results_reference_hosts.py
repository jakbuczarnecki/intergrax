#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-22.2 — partial results contract on reference hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.partial_results_wiring import apply_partial_results_task_defaults
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.task.task import Task


def main() -> int:
    for factory in (
        ApplicationEnvironmentProfile.lab_defaults,
        ApplicationEnvironmentProfile.product_defaults,
    ):
        env = factory()
        if not env.reliability_profile.partial_results_enabled:
            print(f"{env.profile_id}: partial_results_enabled must be True", file=sys.stderr)
            return 1
        task = Task(tenant_id="t1", user_id="u1", message="m")
        task = apply_reliability_task_defaults(task, env)
        if "partial_result_contract.v1" not in task.metadata:
            print(f"{env.profile_id}: missing partial_result_contract metadata", file=sys.stderr)
            return 1

    print("OK: partial results reference hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
