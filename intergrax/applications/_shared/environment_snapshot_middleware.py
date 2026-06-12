# © Artur Czarnecki. All rights reserved.

"""Capture :class:`EnvironmentSnapshot` on task intake (APP-EVOL-1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.applications._shared.environment_snapshot_wiring import capture_environment_snapshot
from intergrax.applications.contracts.environment_snapshot import (
    ENV_SNAPSHOT_RUNTIME_KEY,
    SnapshotCaptureSource,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware

if TYPE_CHECKING:
    from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.manifest import ApplicationManifest


class EnvironmentSnapshotMiddleware(RuntimeMiddleware):
    """Materializes environment snapshot on ``BEFORE_TASK_INTAKE`` (priority 35)."""

    priority = 35
    name = "environment_snapshot"

    def __init__(
        self,
        *,
        manifest: ApplicationManifest,
        environment: ApplicationEnvironmentProfile,
        registry_snapshot: HarnessRegistrySnapshot | None = None,
    ) -> None:
        self._manifest = manifest
        self._environment = environment
        self._registry_snapshot = registry_snapshot

    async def before(self, point: HookPoint, context: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_TASK_INTAKE:
            return HookResult()
        if context.runtime_state.get(ENV_SNAPSHOT_RUNTIME_KEY) is not None:
            return HookResult()

        snapshot = capture_environment_snapshot(
            self._manifest,
            self._environment,
            registry_snapshot=self._registry_snapshot,
            captured_by=SnapshotCaptureSource.INTAKE,
        )
        payload = snapshot.model_dump(mode="json")
        context.runtime_state[ENV_SNAPSHOT_RUNTIME_KEY] = payload
        context.runtime_state["profile_snapshot_id"] = snapshot.profile_snapshot_id

        if (
            self._environment.execution_mode == ExecutionMode.STRICT
            and not snapshot.profile_snapshot_id
        ):
            return HookResult(
                action=HookAction.BLOCK,
                reason="STRICT intake requires profile_snapshot_id",
            )
        return HookResult()

    async def after(self, point: HookPoint, context: HookContext) -> HookResult:
        return HookResult()
