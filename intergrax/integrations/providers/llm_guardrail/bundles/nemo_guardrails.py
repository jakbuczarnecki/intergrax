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
from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.opens import nemo_scan_colang


def _nemo_colang_path(options: GuardrailBackendOptions) -> str | None:
    if options.colang_config_path:
        return options.colang_config_path
    env_path = os.environ.get("INTERGRAX_NEMO_COLANG_CONFIG_PATH", "").strip()
    return env_path or None


def _nemo_scan(text: str, *, mode: str, colang_path: str | None) -> GuardrailScanResult | None:
    if not colang_path:
        return None
    vendor = nemo_scan_colang(text, mode=mode, colang_path=colang_path)
    if vendor is None:
        return None
    if vendor.get("skipped"):
        return GuardrailScanResult(
            allowed=True,
            detail=str(vendor.get("detail", "nemo_guardrails skipped")),
            audit_payload={"engine": "nemo_guardrails", "colang_path": colang_path, "mode": mode},
        )
    allowed = bool(vendor.get("allowed", True))
    if not allowed:
        return GuardrailScanResult(
            allowed=False,
            risk_level=GuardrailRiskLevel.HIGH,
            categories=("nemo_guardrails",),
            matched_rules=("nemo_guardrails:colang",),
            detail=str(vendor.get("detail", "nemo_guardrails Colang policy blocked")),
            audit_payload={"engine": "nemo_guardrails", "colang_path": colang_path, "mode": mode},
        )
    return GuardrailScanResult(
        allowed=True,
        detail=str(vendor.get("detail", "nemo_guardrails pass")),
        audit_payload={"engine": "nemo_guardrails", "colang_path": colang_path, "mode": mode},
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
