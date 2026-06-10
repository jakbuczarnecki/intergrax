# © Artur Czarnecki. All rights reserved.

"""Human review sample queue for evaluation workflows (AUDIT-IDEAL-25.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class HumanReviewSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    run_id: str
    agent_id: str
    scenario_id: str
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed: bool = False
    reviewer_id: str | None = None


class HumanReviewSampleQueue:
    """In-process FIFO queue for shadow/critic human review sampling."""

    def __init__(self) -> None:
        self._samples: list[HumanReviewSample] = []

    def enqueue(
        self,
        *,
        run_id: str,
        agent_id: str,
        scenario_id: str,
        reason: str,
    ) -> HumanReviewSample:
        sample = HumanReviewSample(
            sample_id=f"hrs_{uuid4().hex[:12]}",
            run_id=run_id,
            agent_id=agent_id,
            scenario_id=scenario_id,
            reason=reason,
        )
        self._samples.append(sample)
        return sample

    def list_pending(self, *, limit: int = 50) -> list[HumanReviewSample]:
        pending = [item for item in self._samples if not item.reviewed]
        return pending[:limit]

    def mark_reviewed(self, sample_id: str, *, reviewer_id: str) -> HumanReviewSample | None:
        for index, sample in enumerate(self._samples):
            if sample.sample_id != sample_id:
                continue
            updated = sample.model_copy(update={"reviewed": True, "reviewer_id": reviewer_id})
            self._samples[index] = updated
            return updated
        return None
