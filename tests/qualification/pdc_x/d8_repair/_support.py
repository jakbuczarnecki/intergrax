# © Artur Czarnecki. All rights reserved.

"""Helpers of the T-IGX-041-H repair tests (real governance pipeline, real stores)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactReadRequest,
    GovernedArtifactReadPort,
)
from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource
from tests.qualification.pdc_x.d8._d8_support import (
    TENANT_A,
    GovernanceHarness,
    MutableClock,
    make_governance,
    reader_grants,
)

WORKSPACE = "ws-1"  # the workspace ``make_artifact`` stores artifacts in


def make_clocked_port(
    store: Any,
    *,
    harness: GovernanceHarness | None = None,
    clock: Callable[[], datetime] | None = None,
    object_storage: Any = None,
    max_download_ttl_seconds: int = 300,
    governance: Any = None,
) -> tuple[GovernedArtifactReadPort, GovernanceHarness]:
    """Like ``_d8_support.make_port``, but the sources share the port's clock (expiry at read)."""
    harness = harness or make_governance(granted=reader_grants())
    clock = clock or MutableClock()
    port = GovernedArtifactReadPort(
        governance=governance or harness.boundary,
        audit=harness.recorder,
        sources={
            ArtifactClass.EXECUTION_EVIDENCE: ExecutionArtifactSource(store, clock=clock),
            ArtifactClass.TOOL_OUTPUT: ExecutionArtifactSource(store, clock=clock),
        },
        download_issuer=object_storage,
        max_download_ttl_seconds=max_download_ttl_seconds,
        clock=clock,
    )
    return port, harness


def request(
    artifact_id: str = "art-1",
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str | None = WORKSPACE,
    version: int | None = None,
    artifact_class: ArtifactClass = ArtifactClass.EXECUTION_EVIDENCE,
) -> ArtifactReadRequest:
    return ArtifactReadRequest(
        artifact_class=artifact_class,
        tenant_id=tenant_id,
        artifact_id=artifact_id,
        version=version,
        workspace_id=workspace_id,
    )
