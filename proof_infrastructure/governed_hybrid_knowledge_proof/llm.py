# © Artur Czarnecki. All rights reserved.

"""Deterministic deployment-readiness synthesizer for the flagship proof."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from proof_infrastructure.governed_hybrid_knowledge_proof.fixtures import (
    DEPLOYMENT_POLICY_CONTENT,
)


class DeploymentReadinessDeterministicLLM(LLMAdapter):
    """Apply indexed deployment policy to live project status without network LLM calls."""

    provider = "proof"
    model = "deployment-readiness-deterministic"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        del temperature, max_tokens, run_id
        self.calls += 1
        payload = self._decide(messages)
        return build_adapter_response(content=json.dumps(payload, ensure_ascii=False))

    def _decide(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        evidence = self._extract_evidence(messages)
        if not evidence:
            return {
                "status": "insufficient_evidence",
                "answer": None,
                "used_evidence_ids": [],
            }

        policy_present = any(
            item.get("evidence_type") == "indexed"
            and DEPLOYMENT_POLICY_CONTENT.splitlines()[0]
            in str(item.get("content", ""))
            for item in evidence
        )
        live_payload = next(
            (
                json.loads(str(item.get("content", "")))
                for item in evidence
                if item.get("evidence_type") == "live"
            ),
            None,
        )
        if not policy_present or live_payload is None:
            return {
                "status": "insufficient_evidence",
                "answer": None,
                "used_evidence_ids": [],
            }

        readiness = int(live_payload["readiness_score"])
        open_blockers = [
            blocker
            for blocker in live_payload.get("blockers", [])
            if str(blocker.get("status", "")).upper() == "OPEN"
        ]
        ready = readiness >= 90 and not open_blockers
        used_ids = [
            str(item["evidence_id"])
            for item in evidence
            if isinstance(item.get("evidence_id"), str)
        ]
        return {
            "status": "completed",
            "answer": "YES" if ready else "NO",
            "used_evidence_ids": used_ids,
        }

    @staticmethod
    def _extract_evidence(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        for message in reversed(messages):
            if message.role != "user":
                continue
            try:
                payload = json.loads(str(message.content))
            except json.JSONDecodeError:
                continue
            evidence = payload.get("evidence")
            if isinstance(evidence, list):
                return [item for item in evidence if isinstance(item, dict)]
        return []
