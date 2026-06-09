# © Artur Czarnecki. All rights reserved.

"""Vendor SDK boundary for LLM guardrail providers (M.12)."""

from __future__ import annotations

import os
from typing import Any

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailRiskLevel,
    GuardrailScanResult,
)


def llm_guard_scan_input(text: str) -> GuardrailScanResult | None:
    try:
        from llm_guard import scan_prompt
        from llm_guard.input_scanners import PromptInjection, Toxicity
    except ImportError:
        return None

    scanners = [PromptInjection(), Toxicity()]
    sanitized, results_valid, results_score = scan_prompt(scanners, text)
    blocked = any(not valid for valid in results_valid.values())
    categories = [name for name, valid in results_valid.items() if not valid]
    return GuardrailScanResult(
        allowed=not blocked,
        risk_level=GuardrailRiskLevel.HIGH if blocked else GuardrailRiskLevel.LOW,
        categories=tuple(categories),
        matched_rules=tuple(categories),
        sanitized_text=sanitized,
        detail="llm-guard scan_prompt",
        audit_payload={"engine": "llm_guard", "scores": dict(results_score)},
    )


def llm_guard_scan_output(text: str, *, prompt: str = "") -> GuardrailScanResult | None:
    try:
        from llm_guard import scan_output
        from llm_guard.output_scanners import Sensitive, Toxicity
        from llm_guard.vault import Vault
    except ImportError:
        return None

    vault = Vault()
    scanners = [Sensitive(vault), Toxicity()]
    sanitized, results_valid, results_score = scan_output(scanners, prompt or text, text)
    blocked = any(not valid for valid in results_valid.values())
    categories = [name for name, valid in results_valid.items() if not valid]
    return GuardrailScanResult(
        allowed=not blocked,
        risk_level=GuardrailRiskLevel.HIGH if blocked else GuardrailRiskLevel.LOW,
        categories=tuple(categories),
        matched_rules=tuple(categories),
        sanitized_text=sanitized,
        detail="llm-guard scan_output",
        audit_payload={"engine": "llm_guard", "scores": dict(results_score)},
    )


def presidio_scan_text(text: str) -> GuardrailScanResult | None:
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return None

    engine = AnalyzerEngine()
    results = engine.analyze(text=text, language="en")
    if not results:
        return GuardrailScanResult(
            allowed=True,
            sanitized_text=text,
            detail="presidio clean",
            audit_payload={"engine": "presidio"},
        )
    categories = tuple(sorted({item.entity_type for item in results}))
    return GuardrailScanResult(
        allowed=False,
        risk_level=GuardrailRiskLevel.MEDIUM,
        categories=categories,
        matched_rules=tuple(f"presidio:{item.entity_type}" for item in results),
        detail="presidio PII detected",
        audit_payload={"engine": "presidio", "count": len(results)},
    )


def guardrails_ai_validate(text: str) -> GuardrailScanResult | None:
    try:
        from guardrails import Guard
        from guardrails.hub import ToxicLanguage, DetectPII
    except ImportError:
        return None

    guard = Guard().use(ToxicLanguage(), DetectPII())
    outcome = guard.validate(text)
    if outcome.validation_passed:
        return GuardrailScanResult(
            allowed=True,
            sanitized_text=outcome.validated_output or text,
            detail="guardrails-ai pass",
            audit_payload={"engine": "guardrails_ai"},
        )
    return GuardrailScanResult(
        allowed=False,
        risk_level=GuardrailRiskLevel.HIGH,
        categories=("guardrails_ai",),
        matched_rules=("guardrails_ai:validation_failed",),
        detail=str(outcome.error or "guardrails-ai validation failed"),
        audit_payload={"engine": "guardrails_ai"},
    )


def http_guardrail_scan(
    *,
    slug: str,
    text: str,
    mode: str,
    env_prefix: str,
    default_path: str,
) -> GuardrailScanResult | None:
    api_key = os.environ.get(f"{env_prefix}_API_KEY", "").strip()
    base_url = os.environ.get(f"{env_prefix}_BASE_URL", "").strip()
    if not api_key and not base_url:
        return None
    try:
        import httpx
    except ImportError:
        return None

    url = (base_url or "https://api.example.com").rstrip("/") + default_path
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload: dict[str, Any] = {"text": text, "mode": mode}
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:  # noqa: BLE001 — vendor boundary
        return GuardrailScanResult(
            allowed=True,
            risk_level=GuardrailRiskLevel.LOW,
            detail=f"{slug} http scan skipped: {exc}",
            audit_payload={"engine": slug, "http_error": str(exc)},
        )
    allowed = bool(body.get("allowed", body.get("safe", True)))
    categories = tuple(str(item) for item in body.get("categories", body.get("blocked_categories", ())))
    return GuardrailScanResult(
        allowed=allowed,
        risk_level=GuardrailRiskLevel.HIGH if not allowed else GuardrailRiskLevel.LOW,
        categories=categories,
        sanitized_text=body.get("sanitized_text") or body.get("redacted_text"),
        detail=f"{slug} http scan",
        audit_payload={"engine": slug, "response": body},
    )
