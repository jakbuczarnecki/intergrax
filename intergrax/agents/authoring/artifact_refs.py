# © Artur Czarnecki. All rights reserved.

"""Collect typed artifact refs from step outcomes (ACP-PROD-6)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.artifact_ref import ArtifactRef, artifact_ref_from_payload


def artifact_refs_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    run_id: str,
    trace_id: str,
    agent_id: str,
    step_index: int | None = None,
) -> list[ArtifactRef]:
    return [
        artifact_ref_from_payload(
            item,
            run_id=run_id,
            trace_id=trace_id,
            agent_id=agent_id,
            step_index=step_index,
        )
        for item in payloads
        if isinstance(item, dict)
    ]
