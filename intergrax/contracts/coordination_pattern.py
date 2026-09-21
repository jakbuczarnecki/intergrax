# © Artur Czarnecki. All rights reserved.

"""Multi-agent coordination pattern enum — public declarative contract."""

from __future__ import annotations

from enum import Enum


class CoordinationPattern(str, Enum):
    HIERARCHICAL = "hierarchical"
    ORCHESTRATOR_WORKER = "orchestrator_worker"
    SUPERVISOR_WORKER = "supervisor_worker"
    PEER_TO_PEER = "peer_to_peer"
    SWARM = "swarm"
    EVALUATOR_LOOP = "evaluator_loop"
