# © Artur Czarnecki. All rights reserved.

"""LLM guardrail backend adapters (M-P12.*)."""

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    GuardrailScanResult,
    LlmGuardrailBackend,
)
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns
from intergrax.integrations.providers.llm_guardrail._vendor_opens import (
    guardrails_ai_validate,
    http_guardrail_scan,
    llm_guard_scan_input,
    llm_guard_scan_output,
    presidio_scan_text,
)


class BaseGuardrailAdapter:
    """Shared defaults for catalog guardrail backends."""

    def __init__(self, *, slug: str) -> None:
        self._slug = slug

    @property
    def slug(self) -> str:
        return self._slug

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        joined = " ".join(arguments.values())
        if "ignore previous instructions" in joined.lower():
            return GuardrailScanResult(
                allowed=False,
                categories=("tool_injection",),
                matched_rules=("tool_injection:argument",),
                detail=f"{self._slug} tool argument blocked",
                audit_payload={"tool_name": tool_name, "engine": self._slug},
            )
        return GuardrailScanResult(allowed=True, sanitized_text=joined)

    def health_check(self) -> bool:
        return True


class LlmGuardAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = llm_guard_scan_input(text)
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
        vendor = llm_guard_scan_output(text, prompt=prompt or "")
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)


class GuardrailsAiAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = guardrails_ai_validate(text)
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
        vendor = guardrails_ai_validate(text)
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)


class PresidioAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = presidio_scan_text(text)
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
        return self.scan_input(text, context=context)


class HttpGuardrailAdapter(BaseGuardrailAdapter):
    def __init__(self, *, slug: str, env_prefix: str, path: str) -> None:
        super().__init__(slug=slug)
        self._env_prefix = env_prefix
        self._path = path

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = http_guardrail_scan(
            slug=self._slug,
            text=text,
            mode="input",
            env_prefix=self._env_prefix,
            default_path=self._path,
        )
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
        vendor = http_guardrail_scan(
            slug=self._slug,
            text=text,
            mode="output",
            env_prefix=self._env_prefix,
            default_path=self._path,
        )
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)


class PatternOnlyAdapter(BaseGuardrailAdapter):
    """Nemo / Llama / Bedrock harness adapters — pattern fallback until dedicated bundles ship."""

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        return scan_patterns(text, mode="input", slug=self._slug)

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        return scan_patterns(text, mode="output", slug=self._slug)


def create_llm_guard_backend() -> LlmGuardrailBackend:
    return LlmGuardAdapter(slug="llm_guard")


def create_guardrails_ai_backend() -> LlmGuardrailBackend:
    return GuardrailsAiAdapter(slug="guardrails_ai")


def create_presidio_backend() -> LlmGuardrailBackend:
    return PresidioAdapter(slug="presidio")


def create_nemo_guardrails_backend() -> LlmGuardrailBackend:
    return PatternOnlyAdapter(slug="nemo_guardrails")


def create_openguardrails_backend() -> LlmGuardrailBackend:
    return HttpGuardrailAdapter(
        slug="openguardrails",
        env_prefix="INTERGRAX_OPENGUARDRAILS",
        path="/v1/guardrails/check",
    )


def create_lakera_backend() -> LlmGuardrailBackend:
    return HttpGuardrailAdapter(
        slug="lakera",
        env_prefix="INTERGRAX_LAKERA",
        path="/v1/guard",
    )


def create_azure_content_safety_backend() -> LlmGuardrailBackend:
    return HttpGuardrailAdapter(
        slug="azure_content_safety",
        env_prefix="INTERGRAX_AZURE_CONTENT_SAFETY",
        path="/contentsafety/text:analyze",
    )


def create_bedrock_guardrails_backend() -> LlmGuardrailBackend:
    return PatternOnlyAdapter(slug="bedrock_guardrails")


def create_llama_guard_backend() -> LlmGuardrailBackend:
    return PatternOnlyAdapter(slug="llama_guard")
