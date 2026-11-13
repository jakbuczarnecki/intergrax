# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Spill of large tool output to an artifact reference (PDC-D8-R012).

Owned by the tool runtime class ``TOOL_OUTPUT``: output above the threshold is written to the
execution artifact store with ephemeral retention and replaced in the model-visible result by a
reference and a short preview. The round trip goes back through the governed access port.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactLineage,
    ArtifactProvenance,
    ArtifactRetentionClass,
)
from intergrax.runtime.nexus.artifacts.context_reference import ArtifactReferenceToken
from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSchedule, resolve_expiry
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore
from intergrax.utils.time_provider import SystemTimeProvider

DEFAULT_SPILL_THRESHOLD_BYTES = 16 * 1024
DEFAULT_PREVIEW_CHARS = 512


@dataclass(frozen=True)
class ToolOutputSpill:
    """Either the inline text (small output) or a reference plus a preview (spilled output)."""

    inline: str | None
    reference: ArtifactReferenceToken | None
    preview: str = ""


class ToolOutputSpiller:
    def __init__(
        self,
        store: ArtifactStore,
        *,
        threshold_bytes: int = DEFAULT_SPILL_THRESHOLD_BYTES,
        schedule: ArtifactRetentionSchedule | None = None,
        clock: Callable[[], datetime] | None = None,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
    ) -> None:
        if threshold_bytes < 1:
            raise ValueError("threshold_bytes must be positive")
        self._store = store
        self._threshold = threshold_bytes
        self._schedule = schedule or ArtifactRetentionSchedule.default()
        self._clock = clock or SystemTimeProvider.utc_now
        self._preview_chars = preview_chars

    def spill_if_large(
        self,
        *,
        tenant_id: str,
        run_id: str,
        tool_id: str,
        agent_id: str,
        content: str | bytes,
        mime_type: str = "text/plain",
        step_id: str | None = None,
        task_id: str | None = None,
        workspace_id: str | None = None,
        trace_id: str = "",
    ) -> ToolOutputSpill:
        body = content.encode("utf-8") if isinstance(content, str) else content
        if len(body) <= self._threshold:
            text = content if isinstance(content, str) else content.decode("utf-8", errors="replace")
            return ToolOutputSpill(inline=text, reference=None)

        created_at = self._clock()
        stored = self._store.put(
            Artifact(
                tenant_id=tenant_id,
                artifact_id=f"spill_{uuid4().hex}",
                run_id=run_id,
                step_id=step_id,
                kind="tool_output",
                mime_type=mime_type,
                created_at_utc=created_at,
                data=body,
                size_bytes=len(body),
                artifact_class=ArtifactClass.TOOL_OUTPUT,
                retention_class=ArtifactRetentionClass.EPHEMERAL,
                expires_at_utc=resolve_expiry(
                    created_at,
                    retention_class=ArtifactRetentionClass.EPHEMERAL,
                    schedule=self._schedule,
                ),
                workspace_id=workspace_id,
                producer=ArtifactProvenance(created_by_agent_id=agent_id, created_by_tool_id=tool_id),
                lineage=ArtifactLineage(run_id=run_id, task_id=task_id, trace_id=trace_id),
            )
        )
        token = ArtifactReferenceToken(
            tenant_id=tenant_id,
            artifact_class=ArtifactClass.TOOL_OUTPUT,
            artifact_id=stored.artifact_id,
            version=stored.version,
            content_digest=stored.content_digest,
        )
        preview = body[: self._preview_chars * 4].decode("utf-8", errors="ignore")[: self._preview_chars]
        return ToolOutputSpill(inline=None, reference=token, preview=preview)
