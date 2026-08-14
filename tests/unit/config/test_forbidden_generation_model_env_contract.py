# © Artur Czarnecki. All rights reserved.

"""CONFIG-6 — regression guard against legacy generation model-selection env names."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.config.forbidden_generation_model_env import (
    FORBIDDEN_GENERATION_MODEL_ENV_NAMES,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_REGISTRY_REL = "intergrax/runtime/config/forbidden_generation_model_env.py"

_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "infra",
    _REPO_ROOT / "docs",
    _REPO_ROOT / ".github",
    _REPO_ROOT / "tests",
)

_SKIP_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "build",
    "dist",
    "runtime-context",
}

_ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".sh",
    ".bat",
    ".ps1",
    ".yml",
    ".yaml",
    ".env",
    ".example",
}

# Explicit allowlist for files that may reference forbidden generation-model env names
# for negative regression semantics only. No directory-level exemptions.
_ALLOWED_NEGATIVE_REFERENCE_FILES: frozenset[str] = frozenset(
    {
        "tests/unit/config/test_forbidden_generation_model_env_contract.py",
        "tests/unit/llm_adapters/test_llm_profile.py",
        "tests/unit/llm_adapters/test_native_ollama_adapter.py",
        "tests/unit/docs/test_public_reader_documents_contract.py",
        "tests/unit/docs/test_platform_configuration_reference.py",
        "tests/unit/docs/test_root_env_example.py",
        "tests/unit/applications/test_active_application_env_contracts.py",
        "applications/local_workspace_application/tests/model_runtime/test_model_runtime_proof.py",
    }
)


def _relative_posix(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def is_allowed_forbidden_generation_model_env_reference(rel_path: str) -> bool:
    if rel_path == _REGISTRY_REL:
        return True
    return rel_path in _ALLOWED_NEGATIVE_REFERENCE_FILES


def _should_skip(path: Path, repo_root: Path) -> bool:
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        return False
    return any(part in _SKIP_PARTS for part in rel_parts)


def _iter_scan_files(
    repo_root: Path,
    scan_roots: tuple[Path, ...],
) -> list[Path]:
    registry_file = (repo_root / _REGISTRY_REL).resolve()
    files: list[Path] = []
    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _should_skip(path, repo_root):
                continue
            if path.suffix not in _ALLOWED_SUFFIXES and path.name != ".env.example":
                continue
            if path.resolve() == registry_file:
                continue
            files.append(path)
    return files


def find_forbidden_generation_model_env_violations(
    *,
    repo_root: Path,
    env_name: str,
    scan_roots: tuple[Path, ...] | None = None,
) -> list[str]:
    violations: list[str] = []
    roots = scan_roots or _SCAN_ROOTS
    for path in _iter_scan_files(repo_root, roots):
        rel = _relative_posix(path, repo_root)
        text = path.read_text(encoding="utf-8", errors="replace")
        if env_name not in text:
            continue
        if is_allowed_forbidden_generation_model_env_reference(rel):
            continue
        violations.append(rel)
    return violations


@pytest.mark.parametrize("env_name", sorted(FORBIDDEN_GENERATION_MODEL_ENV_NAMES))
def test_forbidden_generation_model_env_not_present_in_active_repo(env_name: str) -> None:
    violations = find_forbidden_generation_model_env_violations(
        repo_root=_REPO_ROOT,
        env_name=env_name,
    )
    assert not violations, f"{env_name} found in: {', '.join(violations)}"


def test_guard_allows_allowlisted_negative_reference(tmp_path: Path) -> None:
    rel = "tests/unit/llm_adapters/test_llm_profile.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(
        'os.environ["INTERGRAX_DEFAULT_OLLAMA_MODEL"] = "legacy"\n'
        'assert adapter.model != "legacy"\n',
        encoding="utf-8",
    )
    violations = find_forbidden_generation_model_env_violations(
        repo_root=tmp_path,
        env_name="INTERGRAX_DEFAULT_OLLAMA_MODEL",
        scan_roots=(tmp_path / "tests",),
    )
    assert violations == []


def test_guard_rejects_positive_reference_in_source_file(tmp_path: Path) -> None:
    rel = "intergrax/llm_adapters/providers/ollama_adapter.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(
        'os.environ["INTERGRAX_DEFAULT_OLLAMA_MODEL"] = "legacy"\n'
        'assert adapter.model == "legacy"\n',
        encoding="utf-8",
    )
    violations = find_forbidden_generation_model_env_violations(
        repo_root=tmp_path,
        env_name="INTERGRAX_DEFAULT_OLLAMA_MODEL",
        scan_roots=(tmp_path / "intergrax",),
    )
    assert violations == [rel]


def test_guard_rejects_positive_reference_in_non_allowlisted_test_file(tmp_path: Path) -> None:
    rel = "tests/unit/llm_adapters/test_positive_legacy_reintroduction.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(
        'os.environ["INTERGRAX_DEFAULT_OLLAMA_MODEL"] = "legacy"\n'
        'adapter = LangChainOllamaAdapter()\n'
        'assert adapter.model == "legacy"\n',
        encoding="utf-8",
    )
    violations = find_forbidden_generation_model_env_violations(
        repo_root=tmp_path,
        env_name="INTERGRAX_DEFAULT_OLLAMA_MODEL",
        scan_roots=(tmp_path / "tests",),
    )
    assert violations == [rel]
