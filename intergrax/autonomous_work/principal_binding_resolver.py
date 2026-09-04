# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker→Principal identity resolution boundary (AW-3A).

Resolves ``worker_instance_id`` to scoped Collaborative Principal identity only.
Effective authority remains ``CollaborativeWorkAuthorityResolver``.
"""

from __future__ import annotations

from intergrax.autonomous_work.repository import WorkerPrincipalBindingRepository
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.principal_binding import ResolvedWorkerPrincipal


class WorkerPrincipalBindingRequired(Exception):
    """Worker has no durable acting Principal binding — fail closed."""


class WorkerPrincipalBindingResolver:
    """Resolve Worker identity binding to scoped Collaborative Principal identity."""

    def __init__(self, repository: WorkerPrincipalBindingRepository) -> None:
        self._repository = repository

    def resolve(self, *, worker_instance_id: WorkerInstanceId) -> ResolvedWorkerPrincipal:
        binding = self._repository.get(worker_instance_id=worker_instance_id)
        if binding is None:
            raise WorkerPrincipalBindingRequired(
                f"no principal binding for worker {worker_instance_id}"
            )
        return binding.to_resolved_principal()
