# © Artur Czarnecki. All rights reserved.

"""Deterministic deployment-readiness synthesizer for the flagship proof."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypedDict

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from proof_infrastructure.governed_hybrid_knowledge_proof.fixtures import (
    DEPLOYMENT_POLICY_CONTENT,
)


class _AssemblerEvidenceItem(TypedDict, total=False):
    evidence_id: str
    evidence_type: str
    content: str


class _AssemblerDecision(TypedDict, total=False):
    status: str
    answer: str | None
    used_evidence_ids: list[str]


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

    def _decide(self, messages: Sequence[ChatMessage]) -> _AssemblerDecision:
        evidence = self._extract_evidence(messages)
        if not evidence:
            return {
                "status": "insufficient_evidence",
                "answer": None,
                "used_evidence_ids": [],
            }

        policy_id = next(
            (
                str(item.get("evidence_id", ""))
                for item in evidence
                if item.get("evidence_type") == "indexed"
                and DEPLOYMENT_POLICY_CONTENT.splitlines()[0]
                in str(item.get("content", ""))
            ),
            None,
        )
        live_id = next(
            (
                str(item.get("evidence_id", ""))
                for item in evidence
                if item.get("evidence_type") == "live"
            ),
            None,
        )
        live_payload = next(
            (
                json.loads(str(item.get("content", "")))
                for item in evidence
                if item.get("evidence_type") == "live"
            ),
            None,
        )
        if not policy_id or live_payload is None or not live_id:
            return {
                "status": "insufficient_evidence",
                "answer": None,
                "used_evidence_ids": [],
            }

        readiness = int(live_payload["readiness_score"])
        blockers = live_payload.get("blockers", [])
        open_blocker = any(
            isinstance(blocker, dict) and blocker.get("status") == "OPEN"
            for blocker in blockers
        )
        used_ids = [policy_id, live_id]
        if readiness >= 90 and not open_blocker:
            return {
                "status": "completed",
                "answer": "YES — ORION satisfies the approved deployment policy.",
                "used_evidence_ids": used_ids,
            }
        return {
            "status": "completed",
            "answer": "NO — ORION does not satisfy the approved deployment policy.",
            "used_evidence_ids": used_ids,
        }

    def _extract_evidence(
        self,
        messages: Sequence[ChatMessage],
    ) -> list[_AssemblerEvidenceItem]:
        for message in reversed(messages):
            if message.role != "user":
                continue
            try:
                parsed = json.loads(message.content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("evidence"), list):
                evidence: list[_AssemblerEvidenceItem] = []
                for item in parsed["evidence"]:
                    if isinstance(item, dict):
                        evidence.append(
                            {
                                "evidence_id": str(item.get("evidence_id", "")),
                                "evidence_type": str(item.get("evidence_type", "")),
                                "content": str(item.get("content", "")),
                            }
                        )
                return evidence

            marker = "Evidence payload:"
            if marker not in message.content:
                continue
            payload_text = message.content.split(marker, 1)[1].strip()
            legacy_parsed = json.loads(payload_text)
            if not isinstance(legacy_parsed, list):
                return []
            legacy_evidence: list[_AssemblerEvidenceItem] = []
            for item in legacy_parsed:
                if isinstance(item, dict):
                    legacy_evidence.append(
                        {
                            "evidence_id": str(item.get("evidence_id", "")),
                            "evidence_type": str(item.get("evidence_type", "")),
                            "content": str(item.get("content", "")),
                        }
                    )
            return legacy_evidence
        return []
