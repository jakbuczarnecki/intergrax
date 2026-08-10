from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderDependencyError,
    EmbeddingProviderRegistry,
    lazy_import_provider_factory,
)


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


def test_default_embedding_registry_imports_provider_only_on_selection() -> None:
    result = _clean_subprocess(
        """
import json
import sys
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_manager,
    create_default_registry,
)

manager = create_default_embedding_manager()
registry = create_default_registry()
before = {
    name: name in sys.modules
    for name in ("torch", "sentence_transformers", "transformers")
}
provider = registry.get("hf")
after = {
    name: name in sys.modules
    for name in ("torch", "sentence_transformers", "transformers")
}
print(json.dumps({
    "default": registry.default_provider(),
    "provider": provider.provider_name(),
    "manager": type(manager).__name__,
    "before": before,
    "after": after,
}))
"""
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["default"] == "hf"
    assert payload["provider"] == "hf"
    assert payload["manager"] == "EmbeddingManager"
    assert payload["before"] == {
        "torch": False,
        "sentence_transformers": False,
        "transformers": False,
    }
    assert payload["after"]["sentence_transformers"] is True


def test_missing_lazy_provider_dependency_has_controlled_error() -> None:
    factory = lazy_import_provider_factory(
        provider_id="missing",
        module_name="intergrax._missing_dep1_provider",
        class_name="Provider",
        dependency_name="dep1-missing-package",
    )

    registry = EmbeddingProviderRegistry()
    registry.register_factory("missing", factory)

    with pytest.raises(
        EmbeddingProviderDependencyError,
        match="Embedding provider 'missing' requires dependency 'dep1-missing-package'",
    ):
        registry.get("missing")


def test_whisper_distribution_exports_expected_api() -> None:
    import whisper

    assert callable(whisper.load_model)


def test_core_dependency_inventory_has_no_langchain_runtime() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    dependencies = project["dependencies"]
    assert not any(
        dependency.startswith(("langchain", "langgraph"))
        for dependency in dependencies
    )
