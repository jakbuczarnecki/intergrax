#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CW-08/CW-10 full-tree evidence: two independent cold builds + one content-addressed incremental build."""

from __future__ import annotations

import json
import sys

from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.discovery.closed_world_gates import run_cw08, run_cw10
from testing_support.change_safety.discovery.extractors import reset_fact_cache
from testing_support.change_safety.intake.git_change_set import rev_parse_head


def main() -> int:
    head = rev_parse_head()
    cold_a = build_closed_world_inventory(source_commit_sha=head, use_cache=False)
    cold_b = build_closed_world_inventory(source_commit_sha=head, use_cache=False)
    reset_fact_cache()
    incremental = build_closed_world_inventory(source_commit_sha=head, use_cache=True)
    cw08 = run_cw08(cold_a, cold_b)
    cw10 = run_cw10(cold_a, incremental)
    sys.stdout.write(
        json.dumps(
            {
                "source_commit_sha": head,
                "CW-08": {"passed": cw08.passed, **cw08.metrics},
                "CW-10": {"passed": cw10.passed, **cw10.metrics},
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if cw08.passed and cw10.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
