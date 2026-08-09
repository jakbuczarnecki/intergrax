from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_compatibility_module_imports_without_langchain_ollama() -> None:
    result = _run_python(
        """
        import importlib.abc
        import sys

        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "langchain_ollama" or fullname.startswith("langchain_ollama."):
                    raise ModuleNotFoundError("blocked " + fullname, name=fullname)
                return None

        sys.meta_path.insert(0, Blocked())
        from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
        print(LangChainOllamaAdapter.__name__)
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LangChainOllamaAdapter"


def test_missing_compatibility_extra_has_controlled_error() -> None:
    result = _run_python(
        """
        import importlib.abc
        import sys

        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "langchain_ollama" or fullname.startswith("langchain_ollama."):
                    raise ModuleNotFoundError("blocked " + fullname, name=fullname)
                return None

        sys.meta_path.insert(0, Blocked())
        from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
        try:
            LangChainOllamaAdapter(model="qwen2.5:14b")
        except RuntimeError as exc:
            message = str(exc)
            assert "LangChainOllamaAdapter" in message
            assert "llm-langchain-ollama" in message
            assert "uv sync --extra llm-langchain-ollama" in message
        else:
            raise AssertionError("missing optional extra was not reported")
        """
    )

    assert result.returncode == 0, result.stderr


def test_installed_compatibility_adapter_constructs_without_network() -> None:
    pytest.importorskip("langchain_ollama")
    adapter = LangChainOllamaAdapter(
        chat=SimpleNamespace(model="qwen2.5:14b"),
        model="qwen2.5:14b",
    )

    assert adapter.provider is LLMProvider.OLLAMA
    assert adapter.model == "qwen2.5:14b"


def test_core_default_imports_with_all_langchain_packages_blocked() -> None:
    result = _run_python(
        """
        import importlib.abc
        import sys

        blocked = ("langchain", "langchain_core", "langchain_ollama")

        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked or fullname.startswith(tuple(name + "." for name in blocked)):
                    raise ImportError("blocked " + fullname)
                return None

        sys.meta_path.insert(0, Blocked())
        from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
        from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
        from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
        from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
        from intergrax.multimedia.image_smart_loader import ImageSmartLoader

        adapter = LLMAdapterRegistry.create(
            LLMProvider.OLLAMA,
            client=object(),
            model="qwen2.5:14b",
        )
        assert LLMAdapterRegistry.registered_providers()
        assert LLMToolCall.__name__ == "LLMToolCall"
        assert NativeOllamaAdapter.__name__ == "NativeOllamaAdapter"
        assert ImageSmartLoader.__name__ == "ImageSmartLoader"
        assert isinstance(adapter, NativeOllamaAdapter)
        """
    )

    assert result.returncode == 0, result.stderr
