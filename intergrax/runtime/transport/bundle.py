# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.queueing.contracts.task_queue import TaskQueue
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase


@dataclass(frozen=True)
class TransportBundle:
    """
    Immutable container aggregating runtime transport components.

    This bundle represents a fully wired transport layer
    for a selected backend (kafka or rabbitmq).
    """

    task_queue: TaskQueue
    worker: BrokerWorkerBase