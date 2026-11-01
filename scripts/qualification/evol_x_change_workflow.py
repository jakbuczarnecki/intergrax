# © Artur Czarnecki. All rights reserved.

"""EVOL-X change workflow CLI (qualification tooling)."""

from __future__ import annotations

import argparse
import json
import sys

from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import GitChangeSetPort, rev_parse_head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EVOL-X change workflow (inventory + impact)")
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", default=None)
    args = parser.parse_args(argv)
    head = args.head_sha or rev_parse_head()
    change_set = GitChangeSetPort().load(base_sha=args.base_sha, head_sha=head)
    inventory = build_closed_world_inventory(source_commit_sha=head)
    impact = assess_path_change_impact(inventory, changed_paths=change_set.paths)
    sys.stdout.write(json.dumps(impact.model_dump(), indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
