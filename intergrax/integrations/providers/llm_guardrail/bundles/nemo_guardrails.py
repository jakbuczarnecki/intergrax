# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    GuardrailRiskLevel,
    GuardrailScanResult,
    LlmGuardrailBackend,
)
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter
from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.opens import nemo_scan_colang


class NemoGuardrailOptions(BaseModel):
    """Nemo guardrails provider-owned configuration."""

    model_config = ConfigDict(extra="forbid")

    config_path: str | None = None


def _parse_nemo_options(
    provider_options: Mapping[str, Any] | None,
) -> NemoGuardrailOptions:
    if provider_options:
        return NemoGuardrailOptions.model_validate(provider_options)
    return NemoGuardrailOptions()


def _nemo_colang_path(options: NemoGuardrailOptions) -> str | None:
    if options.config_path:
        return options.config_path
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
    def __init__(
        self,
        *,
        slug: str,
        nemo_options: NemoGuardrailOptions,
    ) -> None:
        super().__init__(slug=slug)
        self._nemo_options = nemo_options

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = _nemo_scan(text, mode="input", colang_path=_nemo_colang_path(self._nemo_options))
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
        vendor = _nemo_scan(text, mode="output", colang_path=_nemo_colang_path(self._nemo_options))
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)


def create_nemo_guardrails_backend(
    *,
    provider_options: Mapping[str, Any] | None = None,
) -> LlmGuardrailBackend:
    return NemoGuardrailsAdapter(
        slug="nemo_guardrails",
        nemo_options=_parse_nemo_options(provider_options),
    )
