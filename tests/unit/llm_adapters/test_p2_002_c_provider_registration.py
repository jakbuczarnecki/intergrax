# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from typing import Optional, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import (
    LLMAdapterDependencyError,
    LLMAdapterRegistrationError,
    LLMAdapterRegistry,
)
from intergrax.llm_adapters.providers.registrations._lazy_factory import (
    lazy_adapter_registration_spec,
)
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationSpec,
    OptionalDependencyRequirement,
)


pytestmark = pytest.mark.unit

_LLM_ADAPTERS_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "llm_adapters"
_PRODUCTION_SCAN_ROOTS = (
    _LLM_ADAPTERS_ROOT / "llm_provider_registry.py",
    _LLM_ADAPTERS_ROOT / "registry" / "registration_contract.py",
    _LLM_ADAPTERS_ROOT / "providers" / "registrations",
)

_SDK_ROOTS = (
    "openai",
    "anthropic",
    "ollama",
    "mistralai",
    "boto3",
    "cohere",
    "google",
)

_REPRESENTATIVE_PROVIDERS = (
    (LLMProvider.OPENAI, "openai", "llm-openai"),
    (LLMProvider.GEMINI, "google", "llm-gemini"),
    (LLMProvider.CLAUDE, "anthropic", "llm-anthropic"),
    (LLMProvider.OLLAMA, "ollama", "llm-ollama"),
    (LLMProvider.AWS_BEDROCK, "boto3", "llm-bedrock"),
)

_STATIC_GATE_FORBIDDEN = (
    "_BUILTIN_ADAPTERS",
    "_BUILTIN_OPTIONAL_DEPENDENCIES",
    "_ensure_builtin",
    "importlib.import_module",
    "attribute_access.optional",
    "adapter.__dict__",
    "type(adapter).__dict__",
    "getattr(",
    "setattr(",
    "__getattr__",
    "ensure_available",
)


class _SdkImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_roots: set[str]) -> None:
        self._blocked_roots = blocked_roots

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root in self._blocked_roots:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: object) -> None:
        name = getattr(module, "__name__", "unknown")
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)


class _FakeEnterpriseGatewayAdapter(LLMAdapter):
    provider = "fake_enterprise_gateway"
    model = "fake-model"

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.model = str(kwargs.get("model", self.model))
        self.validated = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def validate(self) -> None:
        super().validate()
        self.validated = True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="gateway-ok")


@pytest.fixture()
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    installed = LLMAdapterRegistry._builtin_registrations_installed
    try:
        yield
    finally:
        LLMAdapterRegistry._factories = snapshot
        LLMAdapterRegistry._builtin_registrations_installed = installed


def _clear_sdk_modules() -> dict[str, object]:
    removed: dict[str, object] = {}
    for name in list(sys.modules):
        if name.split(".", 1)[0] in {root.split(".", 1)[0] for root in _SDK_ROOTS}:
            removed[name] = sys.modules.pop(name)
    return removed


def _iter_production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_SCAN_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(sorted(root.rglob("*.py")))
    return files


def test_production_registration_path_static_gate() -> None:
    for path in _iter_production_python_files():
        source = path.read_text(encoding="utf-8")
        for token in _STATIC_GATE_FORBIDDEN:
            assert token not in source, f"{token} found in {path}"


def test_bootstrap_registration_does_not_import_vendor_sdks(
    _restore_registry_state,
) -> None:
    removed = _clear_sdk_modules()
    LLMAdapterRegistry._factories = {}
    LLMAdapterRegistry._builtin_registrations_installed = False
    blocker = _SdkImportBlocker({root.split(".", 1)[0] for root in _SDK_ROOTS})
    sys.meta_path.insert(0, blocker)
    try:
        providers = LLMAdapterRegistry.registered_providers()
        assert len(providers) == len(LLMProvider)
        for root in _SDK_ROOTS:
            assert not any(
                name == root or name.startswith(f"{root}.") for name in sys.modules
            )
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(removed)


def test_selected_provider_adapter_import_is_lazy(_restore_registry_state) -> None:
    adapter_module = "intergrax.llm_adapters.providers.openai_responses_adapter"
    sys.modules.pop(adapter_module, None)
    LLMAdapterRegistry._factories = {}
    LLMAdapterRegistry._builtin_registrations_installed = False

    providers = LLMAdapterRegistry.registered_providers()
    assert LLMProvider.OPENAI.value in providers
    assert adapter_module not in sys.modules


@pytest.mark.parametrize(("provider", "sdk_root", "extra"), _REPRESENTATIVE_PROVIDERS)
def test_provider_creation_surfaces_dependency_error_without_sdk(
    provider: LLMProvider,
    sdk_root: str,
    extra: str,
    _restore_registry_state,
) -> None:
    removed = _clear_sdk_modules()
    blocker = _SdkImportBlocker({sdk_root.split(".", 1)[0]})
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(LLMAdapterDependencyError, match=extra):
            LLMAdapterRegistry.create(provider, model="test-model")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(removed)


def test_unrelated_module_not_found_is_not_converted(_restore_registry_state) -> None:
    dependency = OptionalDependencyRequirement(
        import_names=("anthropic",),
        distribution_name="anthropic",
        extra_name="llm-anthropic",
    )

    def _raise_unrelated() -> type[LLMAdapter]:
        raise ModuleNotFoundError("No module named 'unrelated_pkg'", name="unrelated_pkg")

    provider = "test-unrelated-missing"
    LLMAdapterRegistry.register_from_spec(
        lazy_adapter_registration_spec(
            provider_id=provider,
            dependency=dependency,
            load_adapter_cls=_raise_unrelated,
        ),
        override=True,
    )

    with pytest.raises(ModuleNotFoundError, match="unrelated_pkg"):
        LLMAdapterRegistry.create(provider)


def test_optional_dependency_matches_declared_roots_only() -> None:
    dependency = OptionalDependencyRequirement(
        import_names=("openai",),
        distribution_name="openai",
        extra_name="llm-openai",
    )
    assert dependency.matches_missing_module(
        ModuleNotFoundError("No module named 'openai'", name="openai")
    )
    assert dependency.matches_missing_module(
        ModuleNotFoundError("No module named 'openai.types'", name="openai.types")
    )
    assert not dependency.matches_missing_module(
        ModuleNotFoundError("No module named 'requests'", name="requests")
    )


def test_fake_enterprise_gateway_pluginability(_restore_registry_state) -> None:
    provider = "fake_enterprise_gateway"

    def factory(**kwargs: object) -> LLMAdapter:
        return _FakeEnterpriseGatewayAdapter(**kwargs)

    LLMAdapterRegistry.register(
        LLMAdapterRegistrationSpec(provider_id=provider, factory=factory),
        override=True,
    )
    assert provider in LLMAdapterRegistry.registered_providers()

    adapter = LLMAdapterRegistry.create(provider, model="enterprise-model")
    assert isinstance(adapter, _FakeEnterpriseGatewayAdapter)
    assert adapter.model == "enterprise-model"
    assert adapter.validated is True


def test_register_accepts_registration_spec(_restore_registry_state) -> None:
    provider = "fake_enterprise_gateway"

    def factory(**kwargs: object) -> LLMAdapter:
        return _FakeEnterpriseGatewayAdapter(**kwargs)

    LLMAdapterRegistry.register(
        LLMAdapterRegistrationSpec(provider_id=provider, factory=factory),
        override=True,
    )
    adapter = LLMAdapterRegistry.create(provider)
    assert isinstance(adapter, _FakeEnterpriseGatewayAdapter)
