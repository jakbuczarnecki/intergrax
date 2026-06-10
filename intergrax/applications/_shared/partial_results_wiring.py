# © Artur Czarnecki. All rights reserved.

"""Partial results contract wiring for reference/product hosts (AUDIT-IDEAL-22.2)."""

from __future__ import annotations

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.partial_result_contract import PartialResultContract
from intergrax.runtime.task.task import Task


def apply_partial_results_task_defaults(task: Task, env: ApplicationEnvironmentProfile) -> Task:
    """Attach partial-result contract metadata when host profile enables it."""
    if env.application_profile not in (ApplicationProfile.PRODUCT, ApplicationProfile.LAB):
        return task
    if not env.reliability_profile.partial_results_enabled:
        return task
    contract = PartialResultContract(
        completed_steps=(),
        recoverable=True,
        metadata={"host_profile": env.profile_id},
    )
    task.metadata["partial_result_contract.v1"] = contract.model_dump()
    task.sync_metadata()
    return task
