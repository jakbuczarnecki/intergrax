# © Artur Czarnecki. All rights reserved.

"""PDC-D3-R023 — ModelCatalog is the single context-window source; provider tables are fallbacks."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens
from intergrax.llm_adapters.registry.model_catalog import ModelCatalog, get_model_catalog

pytestmark = pytest.mark.unit

_PROVIDERS_DIR = Path(__file__).resolve().parents[4] / "intergrax" / "llm_adapters" / "providers"
_TABLE_NAME = re.compile(r"^_?[A-Z_]*CONTEXT_WINDOWS$")

_PROVIDER_OF_FILE = {
    "aws_bedrock_adapter.py": LLMProvider.AWS_BEDROCK,
    "claude_adapter.py": LLMProvider.CLAUDE,
    "cohere_native_adapter.py": LLMProvider.COHERE_NATIVE,
    "gemini_adapter.py": LLMProvider.GEMINI,
    "mistral_adapter.py": LLMProvider.MISTRAL,
    "ollama_adapter.py": LLMProvider.OLLAMA,
    "openai_responses_adapter.py": LLMProvider.OPENAI,
}


def _catalog(models: list[dict[str, Any]] | None = None, **kwargs: Any) -> ModelCatalog:
    payload: dict[str, Any] = {
        "models": models or [],
        "prefix_rules": kwargs.get("prefix_rules", []),
        "provider_defaults": kwargs.get("provider_defaults", {}),
        "fallback_default": kwargs.get("fallback_default", 4_321),
    }
    return ModelCatalog.from_mapping(payload)


# --------------------------------------------------------------------------- resolution order


def test_catalog_exact_record_beats_a_conflicting_provider_local_table() -> None:
    catalog = _catalog([{"model_id": "m-1", "context_window_tokens": 1_000}])
    tokens = resolve_context_window_tokens(LLMProvider.CLAUDE, "m-1", legacy_windows={"m-1": 9_999}, catalog=catalog)
    assert tokens == 1_000


def test_catalog_prefix_rule_beats_a_provider_local_table() -> None:
    catalog = _catalog(prefix_rules=[{"prefix": "m-", "context_window_tokens": 2_000}])
    tokens = resolve_context_window_tokens(LLMProvider.CLAUDE, "m-7", legacy_windows={"m-7": 9_999}, catalog=catalog)
    assert tokens == 2_000


def test_provider_local_table_is_only_a_fallback_for_models_the_catalog_does_not_know() -> None:
    catalog = _catalog(provider_defaults={"claude": 7_000})
    tokens = resolve_context_window_tokens(
        LLMProvider.CLAUDE, "only-local", legacy_windows={"only-local": 9_999}, catalog=catalog
    )
    assert tokens == 9_999
    # ... and it still loses to the catalog's provider default when the model is in no table.
    assert resolve_context_window_tokens(LLMProvider.CLAUDE, "nowhere", legacy_windows={}, catalog=catalog) == 7_000
    assert resolve_context_window_tokens(LLMProvider.GEMINI, "nowhere", legacy_windows={}, catalog=catalog) == 4_321


def test_operator_override_is_the_only_thing_above_the_catalog() -> None:
    catalog = _catalog([{"model_id": "m-1", "context_window_tokens": 1_000}])
    tokens = resolve_context_window_tokens(
        LLMProvider.CLAUDE, "m-1", profile_options={"context_window_tokens": 55}, catalog=catalog
    )
    assert tokens == 55


# --------------------------------------------------------------------------- the formerly bypassing paths


def test_bedrock_estimator_goes_through_the_catalog_not_its_private_prefix_ladder() -> None:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    adapter = BedrockChatAdapter(
        client=MagicMock(),
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        use_converse=True,
        region="us-east-1",
    )
    catalog = get_model_catalog()
    # Exact catalog record, not the removed local ladder.
    record = catalog.lookup_exact("anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert record is not None
    assert adapter._estimate_context_window(record.model_id) == record.context_window_tokens
    # A model in no table and no catalog entry resolves to the catalog's provider default
    # (the removed ladder answered 32_000 here).
    unknown = "vendor.unknown-model-v1:0"
    assert catalog.lookup_exact(unknown) is None and catalog.lookup_prefix(unknown) is None
    assert adapter._estimate_context_window(unknown) == catalog.provider_default("aws_bedrock")


def test_openai_responses_estimator_goes_through_the_catalog() -> None:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    adapter = OpenAIChatResponsesAdapter(client=MagicMock(), model="gpt-4o-mini")
    catalog = get_model_catalog()
    assert adapter._estimate_openai_context_window("gpt-4.1") == catalog.lookup_exact("gpt-4.1").context_window_tokens
    assert adapter._estimate_openai_context_window("gpt-5") == catalog.lookup_exact("gpt-5").context_window_tokens


# --------------------------------------------------------------------------- source gate


def _tables() -> list[tuple[str, str, dict[str, int]]]:
    found: list[tuple[str, str, dict[str, int]]] = []
    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
            if (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and isinstance(target, ast.Name)
                and _TABLE_NAME.match(target.id)
                and isinstance(node.value, ast.Dict)
                and node.value.keys
            ):
                found.append((path.name, target.id, ast.literal_eval(node.value)))
    return found


def test_tables_are_discovered() -> None:
    names = {file for file, _, _ in _tables()}
    assert set(_PROVIDER_OF_FILE) <= names


def test_no_provider_table_value_disagrees_with_or_is_shadowed_by_the_catalog() -> None:
    """Every local entry resolves to its own value via the catalog path: one number per model."""
    problems: list[str] = []
    for filename, table_name, table in _tables():
        provider = _PROVIDER_OF_FILE.get(filename)
        if provider is None:
            continue
        for model, local in table.items():
            resolved = resolve_context_window_tokens(provider, model, legacy_windows=table)
            if resolved != local:
                problems.append(f"{filename}:{table_name}[{model!r}] local={local} catalog-resolved={resolved}")
    assert not problems, "; ".join(problems)


def test_exact_catalog_records_always_win_over_every_local_table() -> None:
    catalog = get_model_catalog()
    for filename, _, table in _tables():
        provider = _PROVIDER_OF_FILE.get(filename)
        if provider is None:
            continue
        for model in table:
            record = catalog.lookup_exact(model)
            if record is not None:
                inflated = {model: record.context_window_tokens + 1}
                assert resolve_context_window_tokens(provider, model, legacy_windows=inflated) == (
                    record.context_window_tokens
                )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def test_provider_tables_are_never_read_outside_the_catalog_resolution_call() -> None:
    """A table may only appear as ``legacy_windows=`` (or the OpenAI-compat ``context_windows=``) argument."""
    offenders: list[str] = []
    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _TABLE_NAME.match(node.attr):
                parent = parents[node]
                ok = isinstance(parent, ast.keyword) and parent.arg in {"legacy_windows", "context_windows"}
                if isinstance(parent, ast.Assign):  # `windows = ... self._CONTEXT_WINDOWS` handoff variable
                    ok = path.name == "openai_chat_completions_adapter.py"
                if isinstance(parent, ast.IfExp):
                    ok = path.name == "openai_chat_completions_adapter.py"
                if not ok:
                    offenders.append(f"{path.name}:{node.lineno} reads {node.attr} directly")
    assert not offenders, "; ".join(offenders)


def test_estimator_methods_delegate_to_the_single_resolver() -> None:
    expected = {
        "aws_bedrock_adapter.py": "_estimate_context_window",
        "gemini_adapter.py": "_estimate_gemini_context_window",
        "mistral_adapter.py": "_estimate_mistral_context_window",
        "ollama_adapter.py": "_estimate_ollama_context_window_from_model",
        "openai_responses_adapter.py": "_estimate_openai_context_window",
    }
    for filename, method in expected.items():
        tree = ast.parse((_PROVIDERS_DIR / filename).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method)
        calls = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "resolve_context_window_tokens" in calls, f"{filename}:{method} bypasses the catalog"
