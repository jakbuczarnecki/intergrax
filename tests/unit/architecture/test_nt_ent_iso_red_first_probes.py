"""Red-first ENT-iso probes (NT-CFG-02/08, NT-OWN-02/03) without nested ENT semantic gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CANONICAL_ARTIFACTS = "intergrax/contracts/task_artifacts.py"
_SHIM_ARTIFACTS = "intergrax/applications/contracts/application_artifacts.py"
_CANONICAL_LLM = "intergrax/runtime/wiring/llm_resolver.py"
_DELEGATING_LLM = "intergrax/applications/_shared/llm_resolver.py"
_EMBEDDING_PROFILE = _REPO / "intergrax/rag/embedding/registry/profile.py"
_EMBEDDING_RESOLVER = _REPO / "intergrax/rag/embedding/runtime/resolver.py"
_ADAPTER_FACTORIES = frozenset({"create_adapter", "create_adapter_with_failover"})


def _class_definition_paths(class_name: str) -> list[str]:
    found: list[str] = []
    for path in (_REPO / "intergrax").rglob("*.py"):
        rel = path.relative_to(_REPO).as_posix()
        if "/tests/" in rel:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body):
            found.append(rel)
    return sorted(found)


def _llm_resolver_adapter_construction() -> dict[str, bool]:
    found: dict[str, bool] = {}
    for root in ("intergrax", "agents", "applications"):
        for path in (_REPO / root).rglob("*.py"):
            rel = path.relative_to(_REPO).as_posix()
            if "/tests/" in rel:
                continue
            text = path.read_text(encoding="utf-8")
            if "def resolve_llm_adapter" not in text:
                continue
            tree = ast.parse(text)
            defines = any(
                isinstance(node, ast.FunctionDef) and node.name == "resolve_llm_adapter" for node in ast.walk(tree)
            )
            if not defines:
                continue
            constructs = any(
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id in _ADAPTER_FACTORIES)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr in _ADAPTER_FACTORIES)
                )
                for node in ast.walk(tree)
            )
            found[rel] = constructs
    return found


def test_nt_cfg_02_missing_embedding_config_must_not_silently_select_vendor() -> None:
    """NT-CFG-02 / FRZ-CFG-02: unset embedding env must fail closed, not default a vendor slug."""
    text = _EMBEDDING_RESOLVER.read_text(encoding="utf-8")
    assert "return validate_embedding_provider_slug(default_embedding_provider_slug())" not in text, (
        "NT-CFG-02 missing or invalid required configuration must not silently select "
        "a vendor/backend when embedding provider is unset"
    )
    profile = _EMBEDDING_PROFILE.read_text(encoding="utf-8")
    assert 'os.getenv(f"{prefix}_PROVIDER", _DEFAULT_EMBEDDING_PROVIDER)' not in profile, (
        "NT-CFG-02 embedding profile must not silently default INTERGRAX_EMBEDDING_PROVIDER to a hard-coded vendor slug"
    )


def test_nt_cfg_08_embedding_profile_must_not_hardcode_provider_getenv_default() -> None:
    """NT-CFG-08 / FRZ-CFG-08: profile builder must not silently activate via getenv provider default."""
    text = _EMBEDDING_PROFILE.read_text(encoding="utf-8")
    assert "_DEFAULT_EMBEDDING_PROVIDER" not in text, (
        "NT-CFG-08 silent fallback: embedding profile must not carry a hard-coded provider constant"
    )
    assert 'os.getenv(f"{prefix}_PROVIDER")' in text or "os.getenv(f'{prefix}_PROVIDER')" in text, (
        "NT-CFG-08 regression gate: embedding provider must be read without a silent getenv default"
    )


def test_nt_own_02_application_artifact_ref_has_one_canonical_definition() -> None:
    """NT-OWN-02 / FRZ-OWN-02: one canonical ApplicationArtifactRef contract per responsibility."""
    paths = _class_definition_paths("ApplicationArtifactRef")
    assert paths == [_CANONICAL_ARTIFACTS], (
        f"NT-OWN-02 duplicate canonical contract: ApplicationArtifactRef defined in {paths!r}, "
        f"expected only {_CANONICAL_ARTIFACTS!r}"
    )
    shim = ast.parse((_REPO / _SHIM_ARTIFACTS).read_text(encoding="utf-8"))
    assert not [n for n in shim.body if isinstance(n, ast.ClassDef)], (
        "NT-OWN-02 application_artifacts shim must not redefine ApplicationArtifactRef"
    )


def test_nt_own_03_one_sanctioned_llm_adapter_composition_owner() -> None:
    """NT-OWN-03 / FRZ-OWN-03: exactly one module constructs LLM adapters at composition boundary."""
    definitions = _llm_resolver_adapter_construction()
    constructing = sorted(path for path, constructs in definitions.items() if constructs)
    assert constructing == [_CANONICAL_LLM], (
        "NT-OWN-03 multiple composition owners construct LLM adapters: " + ", ".join(constructing)
    )
    assert _DELEGATING_LLM in definitions and not definitions[_DELEGATING_LLM], (
        "NT-OWN-03 Tier-3 llm_resolver must delegate, not construct adapters"
    )
