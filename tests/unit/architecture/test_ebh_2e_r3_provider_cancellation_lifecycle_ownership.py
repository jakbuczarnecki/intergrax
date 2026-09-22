# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R3 — provider cancellation & lifecycle seam ownership."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from collections.abc import Callable

import pytest

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationCancellationPort,
)
from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationResult,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from tests.unit.architecture.ebh_2e_external_structural_llm_adapter import (
    ExternalStructuralAdapter,
)
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationSpec,
    ProviderExternalOperationSeam,
)
from intergrax.llm_adapters.providers.registrations.external_operation_seams import (
    BEDROCK_EXTERNAL_OPERATION_CAPABILITIES,
    HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES,
    OLLAMA_LOCAL_EXTERNAL_OPERATION_CAPABILITIES,
    build_bedrock_external_operation_seam,
    build_claude_external_operation_seam,
    build_gemini_external_operation_seam,
    build_mistral_external_operation_seam,
    build_ollama_external_operation_seam,
    build_openai_external_operation_seam,
)
from intergrax.runtime.external_operations.provider_cancellation import (
    resolve_llm_provider_external_operation_seam,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESOLVER_PATH = _REPO_ROOT / "intergrax/runtime/external_operations/provider_cancellation.py"
_PROVIDER_SLUG_LITERALS = (
    "openai",
    "azure_openai",
    "groq",
    "vllm",
    "openrouter",
    "claude",
    "gemini",
    "vertex_gemini",
    "mistral",
    "aws_bedrock",
    "ollama",
)


def _resolver_source() -> str:
    return _RESOLVER_PATH.read_text(encoding="utf-8-sig")


def _resolver_import_modules() -> list[str]:
    tree = ast.parse(_resolver_source())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def test_ebh_2e_r3_runtime_resolver_has_no_provider_package_imports() -> None:
    offenders = [
        mod
        for mod in _resolver_import_modules()
        if mod.startswith("intergrax.llm_adapters.providers.")
    ]
    assert offenders == []


def test_ebh_2e_r3_runtime_resolver_has_no_llm_provider_import() -> None:
    modules = _resolver_import_modules()
    assert not any(mod.endswith("llm_provider") or mod.endswith(".llm_provider") for mod in modules)
    assert "LLMProvider" not in _resolver_source()


def test_ebh_2e_r3_runtime_resolver_has_no_provider_slug_literals() -> None:
    lowered = _resolver_source().lower()
    hits = [slug for slug in _PROVIDER_SLUG_LITERALS if f'"{slug}"' in lowered or f"'{slug}'" in lowered]
    assert hits == []


def test_ebh_2e_r3_runtime_resolver_has_no_openai_fallback() -> None:
    assert "openai_external_operation_ports" not in _resolver_source()
    assert "external_operation_capabilities_for_provider" not in _resolver_source()


class _ExternalCancellationPort:
    async def request_cancel(self, operation_id: str) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")


class _ExternalTerminationPort:
    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        return TerminationResult.physical_stop_confirmed()


class _R3ExternalStructuralAdapter(ExternalStructuralAdapter):
    provider = "ebh-r3-external-provider"


def _external_seam_factory() -> ProviderExternalOperationSeam:
    return ProviderExternalOperationSeam(
        cancellation=_ExternalCancellationPort(),
        termination=_ExternalTerminationPort(),
        stream_registry=ProviderStreamTransportRegistry(),
    )


def test_ebh_2e_r3_external_provider_registration_resolves_without_core_edit() -> None:
    LLMAdapterRegistry.reset_for_testing()
    provider_id = "ebh-r3-external-provider"
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=provider_id,
            factory=lambda **_kwargs: _R3ExternalStructuralAdapter(),
            external_operation_seam_factory=_external_seam_factory,
            external_operation_capabilities=ExternalOperationCapabilities(
                supports_native_cancel=True,
                supports_stream_abort=False,
                supports_remote_termination=True,
            ),
        )
    )
    cancel, termination, registry = resolve_llm_provider_external_operation_seam(provider_id)
    assert isinstance(cancel, ExternalOperationCancellationPort)
    assert registry is not None
    caps = LLMAdapterRegistry.external_operation_capabilities_for(provider_id)
    assert caps.supports_native_cancel is True
    assert caps.supports_stream_abort is False
    LLMAdapterRegistry.reset_for_testing()


def test_ebh_2e_r3_unknown_provider_gets_provider_neutral_default() -> None:
    LLMAdapterRegistry.reset_for_testing()
    cancel, termination, _registry = resolve_llm_provider_external_operation_seam(
        "totally-unknown-provider-slug"
    )
    caps = LLMAdapterRegistry.external_operation_capabilities_for("totally-unknown-provider-slug")
    assert caps.supports_native_cancel is False
    assert caps.supports_stream_abort is False
    assert caps.supports_remote_termination is False
    assert isinstance(cancel, ExternalOperationCancellationPort)


def test_ebh_2e_r3_openai_compatible_providers_share_registration_seam_not_runtime_branch() -> None:
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    openai_seam = build_openai_external_operation_seam()
    for slug in ("openai", "groq", "openrouter", "azure_openai"):
        cancel_a, term_a, _ = resolve_llm_provider_external_operation_seam(slug)
        cancel_b, term_b, _ = (
            openai_seam.cancellation,
            openai_seam.termination,
            openai_seam.stream_registry,
        )
        assert type(cancel_a) is type(cancel_b)
        assert type(term_a) is type(term_b)


@pytest.mark.parametrize(
    ("slug", "expected_caps", "seam_builder"),
    [
        ("openai", HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES, build_openai_external_operation_seam),
        ("claude", HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES, build_claude_external_operation_seam),
        ("gemini", HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES, build_gemini_external_operation_seam),
        ("mistral", HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES, build_mistral_external_operation_seam),
        ("aws_bedrock", BEDROCK_EXTERNAL_OPERATION_CAPABILITIES, build_bedrock_external_operation_seam),
        ("ollama", OLLAMA_LOCAL_EXTERNAL_OPERATION_CAPABILITIES, build_ollama_external_operation_seam),
    ],
)
def test_ebh_2e_r3_builtin_provider_capability_seam_consistency(
    slug: str,
    expected_caps: ExternalOperationCapabilities,
    seam_builder: Callable[[], ProviderExternalOperationSeam],
) -> None:
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    caps = LLMAdapterRegistry.external_operation_capabilities_for(slug)
    assert caps == expected_caps
    cancel, termination, _ = resolve_llm_provider_external_operation_seam(slug)
    reference = seam_builder()
    assert type(cancel) is type(reference.cancellation)
    assert type(termination) is type(reference.termination)


def test_ebh_2e_r3_r1_public_llm_adapter_unchanged() -> None:
    from intergrax.llm_adapters.contracts import llm_adapter as llm_adapter_module

    source = Path(llm_adapter_module.__file__).read_text(encoding="utf-8")
    assert "cancel" not in source
    assert "terminate" not in source
    assert "bind_external_operation_ports" not in source
