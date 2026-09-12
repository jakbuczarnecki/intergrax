# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Immutable execution correlation context (SELF-HEALING R3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SelfHealingExecutionContext:
    """
    Auditable correlation bundle — never mints substitute execution identifiers.

    ``execution_ids`` and ``operation_attempt_ids`` must originate from
    ExecutionRuntime / ExternalOperation spine records only.
    """

    workflow_id: str
    plan_id: str
    strategy_id: str
    tenant_id: str
    execution_ids: tuple[str, ...]
    operation_attempt_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    created_at: datetime
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if not self.plan_id.startswith("sh_plan_"):
            raise ValueError("plan_id must be sh_plan_*")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs required")

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "plan_id": self.plan_id,
            "strategy_id": self.strategy_id,
            "tenant_id": self.tenant_id,
            "execution_ids": list(self.execution_ids),
            "operation_attempt_ids": list(self.operation_attempt_ids),
            "evidence_refs": list(self.evidence_refs),
            "created_at": self.created_at.isoformat(),
            "metadata": {k: v for k, v in self.metadata},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_serializable_dict(), sort_keys=True)


__all__ = ["SelfHealingExecutionContext"]
