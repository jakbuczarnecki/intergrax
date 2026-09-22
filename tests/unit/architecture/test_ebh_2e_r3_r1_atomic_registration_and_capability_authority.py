# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R3-R1 — atomic registration & single capability authority."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

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
from intergrax.llm_adapters.llm_provider_registry import (
    LLMAdapterRegistrationError,
    LLMAdapterRegistry,
)
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterFactory,
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
from tests.unit.architecture.ebh_2e_external_structural_llm_adapter import (
    ExternalStructuralAdapter,
)
from tests.unit.llm_adapters.registry_state_test_support import (
    snapshot_registry_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRATION_CONTRACT = (
    _REPO_ROOT / "intergrax/llm_adapters/registry/registration_contract.py"
)
_REGISTRY_PATH = _REPO_ROOT / "intergrax/llm_adapters/llm_provider_registry.py"
_SEAM_BUILDERS_PATH = (
    _REPO_ROOT
    / "intergrax/llm_adapters/providers/registrations/external_operation_seams.py"
)
_RESOLVER_PATH = _REPO_ROOT / "intergrax/runtime/external_operations/provider_cancellation.py"


class _TestAdapter(ExternalStructuralAdapter):
    provider = "r3-r1-test"


def _factory(**_kwargs: object) -> _TestAdapter:
    return _TestAdapter()


class _CancelPort:
    async def request_cancel(self, operation_id: str) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")


class _TermPort:
    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        return TerminationResult.not_supported()


def _seam_factory() -> ProviderExternalOperationSeam:
    return ProviderExternalOperationSeam(
        cancellation=_CancelPort(),
        termination=_TermPort(),
        stream_registry=ProviderStreamTransportRegistry(),
    )


def _caps(**kwargs: bool) -> ExternalOperationCapabilities:
    return ExternalOperationCapabilities(
        supports_native_cancel=kwargs.get("supports_native_cancel", False),
        supports_stream_abort=kwargs.get("supports_stream_abort", False),
        supports_remote_termination=kwargs.get("supports_remote_termination", False),
    )


def _valid_lifecycle_spec(
    provider_id: str,
    *,
    factory: LLMAdapterFactory | None = None,
) -> LLMAdapterRegistrationSpec:
    return LLMAdapterRegistrationSpec(
        provider_id=provider_id,
        factory=factory or _factory,
        external_operation_seam_factory=_seam_factory,
        external_operation_capabilities=_caps(supports_native_cancel=True),
    )


def test_ebh_2e_r3_r1_provider_external_operation_seam_has_no_capabilities_field() -> None:
    tree = ast.parse(_REGISTRATION_CONTRACT.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ProviderExternalOperationSeam":
            field_names = [
                target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                for target in [stmt.target]
            ]
            assert "capabilities" not in field_names
            return
    pytest.fail("ProviderExternalOperationSeam not found")


def test_ebh_2e_r3_r1_seam_builders_do_not_pass_capabilities_kwarg() -> None:
    source = _SEAM_BUILDERS_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id != "ProviderExternalOperationSeam":
                continue
            for keyword in node.keywords:
                assert keyword.arg != "capabilities"


def test_ebh_2e_r3_r1_register_validates_before_mutating_factories() -> None:
    LLMAdapterRegistry.reset_for_testing()
    before = snapshot_registry_state()
    invalid = LLMAdapterRegistrationSpec(
        provider_id="r3-r1-invalid-new",
        factory=_factory,
        external_operation_seam_factory=_seam_factory,
        external_operation_capabilities=None,
    )
    with pytest.raises(LLMAdapterRegistrationError):
        LLMAdapterRegistry.register_from_spec(invalid)
    assert snapshot_registry_state() == before


def test_ebh_2e_r3_r1_invalid_override_leaves_registry_unchanged() -> None:
    LLMAdapterRegistry.reset_for_testing()
    provider_id = "r3-r1-override-victim"
    LLMAdapterRegistry.register_from_spec(_valid_lifecycle_spec(provider_id))
    before = snapshot_registry_state()

    def other_factory(**_kwargs: object) -> _TestAdapter:
        return _TestAdapter()

    invalid_override = LLMAdapterRegistrationSpec(
        provider_id=provider_id,
        factory=other_factory,
        external_operation_seam_factory=_seam_factory,
        external_operation_capabilities=None,
    )
    with pytest.raises(LLMAdapterRegistrationError):
        LLMAdapterRegistry.register_from_spec(invalid_override, override=True)
    assert snapshot_registry_state() == before


def test_ebh_2e_r3_r1_duplicate_registration_does_not_mutate() -> None:
    LLMAdapterRegistry.reset_for_testing()
    provider_id = "r3-r1-duplicate"
    LLMAdapterRegistry.register_from_spec(_valid_lifecycle_spec(provider_id))
    before = snapshot_registry_state()
    with pytest.raises(ValueError, match="already registered"):
        LLMAdapterRegistry.register_from_spec(_valid_lifecycle_spec(provider_id))
    assert snapshot_registry_state() == before


def test_ebh_2e_r3_r1_capabilities_without_seam_factory_rejected_atomically() -> None:
    LLMAdapterRegistry.reset_for_testing()
    before = snapshot_registry_state()
    spec = LLMAdapterRegistrationSpec(
        provider_id="r3-r1-caps-only",
        factory=_factory,
        external_operation_capabilities=_caps(supports_stream_abort=True),
    )
    with pytest.raises(LLMAdapterRegistrationError, match="seam_factory"):
        LLMAdapterRegistry.register_from_spec(spec)
    assert snapshot_registry_state() == before


def test_ebh_2e_r3_r1_valid_override_replaces_whole_registration() -> None:
    LLMAdapterRegistry.reset_for_testing()
    provider_id = "r3-r1-full-replace"
    LLMAdapterRegistry.register_from_spec(_valid_lifecycle_spec(provider_id))

    def replaced_factory(**_kwargs: object) -> _TestAdapter:
        return _TestAdapter()

    replaced_caps = _caps(
        supports_native_cancel=False,
        supports_stream_abort=True,
        supports_remote_termination=True,
    )
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=provider_id,
            factory=replaced_factory,
            external_operation_seam_factory=_seam_factory,
            external_operation_capabilities=replaced_caps,
        ),
        override=True,
    )
    assert LLMAdapterRegistry._factories[provider_id] is replaced_factory
    assert provider_id in LLMAdapterRegistry._external_operation_seam_factories
    assert LLMAdapterRegistry._external_operation_capabilities[provider_id] == replaced_caps


def test_ebh_2e_r3_r1_override_removing_seam_clears_lifecycle_projection() -> None:
    LLMAdapterRegistry.reset_for_testing()
    provider_id = "r3-r1-seam-removal"
    LLMAdapterRegistry.register_from_spec(_valid_lifecycle_spec(provider_id))
    assert provider_id in LLMAdapterRegistry._external_operation_seam_factories

    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(provider_id=provider_id, factory=_factory),
        override=True,
    )
    assert provider_id in LLMAdapterRegistry._factories
    assert provider_id not in LLMAdapterRegistry._external_operation_seam_factories
    assert provider_id not in LLMAdapterRegistry._external_operation_capabilities
    caps = LLMAdapterRegistry.external_operation_capabilities_for(provider_id)
    assert caps.supports_native_cancel is False
    assert caps.supports_stream_abort is False


def test_ebh_2e_r3_r1_reset_clears_all_logical_state() -> None:
    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.register_from_spec(_valid_lifecycle_spec("r3-r1-reset-me"))
    LLMAdapterRegistry.reset_for_testing()
    assert LLMAdapterRegistry._factories == {}
    assert LLMAdapterRegistry._external_operation_seam_factories == {}
    assert LLMAdapterRegistry._external_operation_capabilities == {}


def test_ebh_2e_r3_r1_builtin_bootstrap_restores_coherent_records() -> None:
    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    for slug in (
        "openai",
        "claude",
        "gemini",
        "mistral",
        "aws_bedrock",
        "ollama",
    ):
        assert slug in LLMAdapterRegistry._factories
        assert slug in LLMAdapterRegistry._external_operation_seam_factories
        assert slug in LLMAdapterRegistry._external_operation_capabilities


def test_ebh_2e_r3_r1_external_provider_still_registers() -> None:
    LLMAdapterRegistry.reset_for_testing()
    provider_id = "r3-r1-external-plugin"
    caps = _caps(
        supports_native_cancel=True,
        supports_stream_abort=False,
        supports_remote_termination=True,
    )
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=provider_id,
            factory=_factory,
            external_operation_seam_factory=_seam_factory,
            external_operation_capabilities=caps,
        )
    )
    cancel, _term, registry = resolve_llm_provider_external_operation_seam(provider_id)
    assert isinstance(cancel, ExternalOperationCancellationPort)
    assert registry is not None
    assert LLMAdapterRegistry.external_operation_capabilities_for(provider_id) == caps


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
def test_ebh_2e_r3_r1_builtin_capability_projection_from_registration(
    slug: str,
    expected_caps: ExternalOperationCapabilities,
    seam_builder: Callable[[], ProviderExternalOperationSeam],
) -> None:
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    assert LLMAdapterRegistry.external_operation_capabilities_for(slug) == expected_caps
    cancel, termination, _ = resolve_llm_provider_external_operation_seam(slug)
    reference = seam_builder()
    assert type(cancel) is type(reference.cancellation)
    assert type(termination) is type(reference.termination)


def test_ebh_2e_r3_r1_runtime_resolver_remains_provider_neutral() -> None:
    source = _RESOLVER_PATH.read_text(encoding="utf-8-sig")
    assert "openai_external_operation_ports" not in source
    assert "LLMProvider" not in source
    assert not any(
        slug in source
        for slug in ('"openai"', "'openai'", '"claude"', "'claude'")
    )


def test_ebh_2e_r3_r1_custom_provider_retained_after_builtin_bootstrap() -> None:
    LLMAdapterRegistry.reset_for_testing()
    custom_id = "r3-r1-custom-retained"
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(provider_id=custom_id, factory=_factory)
    )
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    assert custom_id in LLMAdapterRegistry.registered_providers()


def test_ebh_2e_r3_r1_register_from_spec_commit_after_validation_only() -> None:
    source = _REGISTRY_PATH.read_text(encoding="utf-8-sig")
    assert "_validate_registration_spec" in source
    assert "_commit_registration_spec" in source
    register_fn = ast.parse(source).body
    class_def = next(
        node
        for node in register_fn
        if isinstance(node, ast.ClassDef) and node.name == "LLMAdapterRegistry"
    )
    register_method = next(
        node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) and node.name == "register_from_spec"
    )
    body = register_method.body
    validate_idx = next(
        i
        for i, stmt in enumerate(body)
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "_validate_registration_spec"
    )
    commit_idx = next(
        i
        for i, stmt in enumerate(body)
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "_commit_registration_spec"
    )
    assert validate_idx < commit_idx
