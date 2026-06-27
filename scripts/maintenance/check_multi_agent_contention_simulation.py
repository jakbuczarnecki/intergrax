#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-26.2 — simulation tests for multi-agent contention."""

from __future__ import annotations

import sys

from intergrax.runtime.architecture.multi_agent_contention_simulation import (
    ContentionAgentRequest,
    simulate_multi_agent_contention,
)


def main() -> int:
    report = simulate_multi_agent_contention(
        pool_size=4,
        requests=[
            ContentionAgentRequest(agent_id="a1", requested_slots=2),
            ContentionAgentRequest(agent_id="a2", requested_slots=2),
            ContentionAgentRequest(agent_id="a3", requested_slots=2),
        ],
    )
    if not report.deadlock_free:
        print("contention simulation must remain deadlock-free", file=sys.stderr)
        return 1
    if not report.acceptance_passed:
        print("multi-agent acceptance must pass for contention scenario", file=sys.stderr)
        return 1
    print(f"OK: multi-agent contention simulation ({len(report.allocations)} allocations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
