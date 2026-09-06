from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _clean_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_harness_import_does_not_load_local_ml() -> None:
    result = _clean_subprocess(
        """
import json
import sys
import intergrax.harness

print(json.dumps({
    name: name in sys.modules
    for name in ("torch", "sentence_transformers", "transformers")
}))
"""
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "torch": False,
        "sentence_transformers": False,
        "transformers": False,
    }


def test_default_embedding_bootstrap_imports_provider_only_on_selection() -> None:
    result = _clean_subprocess(
        """
import json
import sys
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_manager,
    create_default_embedding_engine,
)
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.embedding.registry.profile import EmbeddingProfile
from intergrax.rag.embedding.runtime.resolver import bind_embedding_provider

register_default_integrations(preset="full")
manager = create_default_embedding_manager()
before = {
    name: name in sys.modules
    for name in ("torch", "sentence_transformers", "transformers")
}
try:
    provider = bind_embedding_provider(
        integration_profile=IntegrationProfile(embedding_provider="hf"),
        embedding_profile=EmbeddingProfile(provider="hf", model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    engine = create_default_embedding_engine(provider=provider)
    selection = {
        "provider": provider.provider_name(),
        "engine_provider": engine.provider.provider_name(),
    }
except Exception as exc:
    selection = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
after = {
    name: name in sys.modules
    for name in ("torch", "sentence_transformers", "transformers")
}
print(json.dumps({
    "manager": type(manager).__name__,
    "before": before,
    "after": after,
    "selection": selection,
}))
"""
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["manager"] == "EmbeddingManager"
    assert payload["before"] == {
        "torch": False,
        "sentence_transformers": False,
        "transformers": False,
    }
    if "provider" in payload["selection"]:
        assert payload["selection"]["provider"] == "hf"
        assert payload["after"]["sentence_transformers"] is True
    else:
        assert payload["selection"]["error_type"] in {
            "EmbeddingProviderDependencyError",
            "ModuleNotFoundError",
            "ImportError",
        }


def test_whisper_is_owned_by_media_extra() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert not any(dependency.startswith("openai-whisper") for dependency in project["dependencies"])
    whisper_extra = project["optional-dependencies"]["media-whisper"]
    assert "openai-whisper>=20240930,<20250626" in whisper_extra
    assert "webvtt-py>=0.4,<1" in whisper_extra


def test_whisper_opens_import_is_lazy() -> None:
    result = _clean_subprocess(
        """
import json
import sys
from intergrax.integrations.providers.document_parser.whisper import opens

print(json.dumps({
    "module_loaded": "whisper" in sys.modules,
    "opener": callable(opens.transcribe_audio_file),
}))
"""
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "module_loaded": False,
        "opener": True,
    }


def test_whisper_missing_dependency_has_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.integrations.contracts.base import IntegrationDependencyError
    from intergrax.integrations.providers.document_parser.whisper import opens

    real_import = builtins.__import__

    def missing_whisper(name: str, *args: object, **kwargs: object) -> object:
        if name == "whisper":
            raise ModuleNotFoundError("No module named 'whisper'", name="whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_whisper)

    with pytest.raises(
        IntegrationDependencyError,
        match=r"Intergrax-ai\[media-whisper\]",
    ):
        opens._import_whisper()


def test_core_dependency_inventory_has_no_langchain_runtime() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    dependencies = project["dependencies"]
    assert not any(
        dependency.startswith(("langchain", "langgraph"))
        for dependency in dependencies
    )
