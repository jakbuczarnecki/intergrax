# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailBackendOptions,
    GuardrailContext,
    GuardrailRiskLevel,
    GuardrailScanResult,
    LlmGuardrailBackend,
)
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter


def _bedrock_policy_id(options: GuardrailBackendOptions) -> str | None:
    if options.bedrock_guardrail_policy_id:
        return options.bedrock_guardrail_policy_id
    return os.environ.get("INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID", "").strip() or None


def _bedrock_scan(text: str, *, mode: str, policy_id: str | None) -> GuardrailScanResult | None:
    if not policy_id:
        return None
    from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens import (
        bedrock_apply_guardrail,
    )

    try:
        response = bedrock_apply_guardrail(text, policy_id=policy_id, mode=mode)
    except Exception as exc:  # noqa: BLE001 — vendor boundary
        return GuardrailScanResult(
            allowed=True,
            detail=f"bedrock_guardrails skipped: {exc}",
            audit_payload={"engine": "bedrock_guardrails", "policy_id": policy_id},
        )
    action = str(response.get("action", "NONE"))
    blocked = action.upper() in {"GUARDRAIL_INTERVENED", "BLOCKED"}
    if blocked:
        return GuardrailScanResult(
            allowed=False,
            risk_level=GuardrailRiskLevel.HIGH,
            categories=("bedrock_guardrails",),
            matched_rules=(f"bedrock:{policy_id}",),
            detail="AWS Bedrock Guardrails intervened",
            audit_payload={"engine": "bedrock_guardrails", "policy_id": policy_id, "action": action},
        )
    return GuardrailScanResult(
        allowed=True,
        detail="bedrock_guardrails pass",
        audit_payload={"engine": "bedrock_guardrails", "policy_id": policy_id},
    )


class BedrockGuardrailsAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = _bedrock_scan(text, mode="input", policy_id=_bedrock_policy_id(self._options))
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="input", slug=self._slug)

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        vendor = _bedrock_scan(text, mode="output", policy_id=_bedrock_policy_id(self._options))
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)


def create_bedrock_guardrails_backend(*, options: GuardrailBackendOptions | None = None) -> LlmGuardrailBackend:
    return BedrockGuardrailsAdapter(slug="bedrock_guardrails", options=options)
