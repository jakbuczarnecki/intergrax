#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-9.2 — swarm + peer-to-peer coordination graph templates."""

from __future__ import annotations

import sys

from intergrax.applications._shared.swarm_graph_templates import (
    peer_to_peer_graph_template,
    swarm_exploration_graph_template,
)
from intergrax.runtime.architecture.multi_agent_coordination import (
    CoordinationPattern,
    build_default_coordination_catalog,
)


def main() -> int:
    catalog = build_default_coordination_catalog()
    patterns = {item.pattern for item in catalog.patterns}
    if CoordinationPattern.SWARM not in patterns or CoordinationPattern.PEER_TO_PEER not in patterns:
        print("coordination catalog missing swarm or peer_to_peer", file=sys.stderr)
        return 1

    swarm = swarm_exploration_graph_template(
        worker_agent_ids=("worker_a", "worker_b", "worker_c"),
        aggregator_agent_id="aggregator",
    )
    if len(swarm.nodes) != 4 or len(swarm.edges) != 3:
        print("invalid swarm graph template", file=sys.stderr)
        return 1

    p2p = peer_to_peer_graph_template(agent_ids=("agent_a", "agent_b", "agent_c"))
    if len(p2p.edges) != 3:
        print("invalid peer_to_peer graph template", file=sys.stderr)
        return 1

    print("OK: swarm coordination templates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
