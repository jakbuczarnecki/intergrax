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


def _nemo_colang_path(options: GuardrailBackendOptions) -> str | None:
    if options.colang_config_path:
        return options.colang_config_path
    env_path = os.environ.get("INTERGRAX_NEMO_COLANG_CONFIG_PATH", "").strip()
    return env_path or None


def _nemo_scan(text: str, *, mode: str, colang_path: str | None) -> GuardrailScanResult | None:
    if not colang_path:
        return None
    try:
        from nemoguardrails import RailsConfig
        from nemoguardrails.rails.llm.llmrails import LLMRails
    except ImportError:
        return None
    try:
        config = RailsConfig.from_path(colang_path)
        rails = LLMRails(config)
        messages = [{"role": "user", "content": text}]
        blocked = not bool(rails.generate(messages=messages))
    except Exception as exc:  # noqa: BLE001 — vendor boundary
        return GuardrailScanResult(
            allowed=True,
            detail=f"nemo_guardrails skipped: {exc}",
            audit_payload={"engine": "nemo_guardrails", "colang_path": colang_path, "mode": mode},
        )
    if blocked:
        return GuardrailScanResult(
            allowed=False,
            risk_level=GuardrailRiskLevel.HIGH,
            categories=("nemo_guardrails",),
            matched_rules=("nemo_guardrails:colang",),
            detail="nemo_guardrails Colang policy blocked",
            audit_payload={"engine": "nemo_guardrails", "colang_path": colang_path},
        )
    return GuardrailScanResult(
        allowed=True,
        detail="nemo_guardrails pass",
        audit_payload={"engine": "nemo_guardrails", "colang_path": colang_path},
    )


class NemoGuardrailsAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = _nemo_scan(text, mode="input", colang_path=_nemo_colang_path(self._options))
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
        vendor = _nemo_scan(text, mode="output", colang_path=_nemo_colang_path(self._options))
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)


def create_nemo_guardrails_backend(*, options: GuardrailBackendOptions | None = None) -> LlmGuardrailBackend:
    return NemoGuardrailsAdapter(slug="nemo_guardrails", options=options)
