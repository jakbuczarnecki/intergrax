# © Artur Czarnecki. All rights reserved.

"""CE-02-R1F mandatory base message classification (Tier-0, no Nexus)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.context.budget.mandatory_base_messages import (
    estimate_mandatory_base_message_tokens,
    mandatory_base_message_indices,
)
from intergrax.context.budget.mandatory_reserve import estimate_mandatory_reserve_tokens
from intergrax.context.contracts import ContextFragment, ContextFragmentSource
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_mandatory_reserve_system_and_user_turn() -> None:
    messages = [
        ChatMessage(role="system", content="sys-instructions"),
        ChatMessage(role="user", content="current question"),
    ]
    reserve = estimate_mandatory_reserve_tokens(
        base_messages=messages,
        collected_fragments=(),
        count_text=len,
    )
    assert reserve == len("sys-instructions") + len("current question")


def test_multiple_system_instructions_all_mandatory() -> None:
    messages = [
        ChatMessage(role="system", content="primary"),
        ChatMessage(role="system", content="secondary"),
        ChatMessage(role="user", content="task"),
    ]
    indices = mandatory_base_message_indices(messages)
    assert indices == frozenset({0, 1, 2})
    reserve = estimate_mandatory_base_message_tokens(messages, count_text=len)
    assert reserve == len("primary") + len("secondary") + len("task")


def test_current_user_turn_mandatory_not_prior_user() -> None:
    messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="old"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="current"),
    ]
    indices = mandatory_base_message_indices(messages)
    assert 3 in indices
    assert 1 not in indices
    assert 2 not in indices


def test_mandatory_fragment_added_once_no_double_count() -> None:
    base = [
        ChatMessage(role="system", content="x" * 10),
        ChatMessage(role="user", content="y" * 10),
    ]
    fragment = ContextFragment(
        fragment_id="mandatory-frag",
        source=ContextFragmentSource.RAG,
        source_id="doc",
        content="fragment body",
        token_estimate=300,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
    )
    reserve = estimate_mandatory_reserve_tokens(
        base_messages=base,
        collected_fragments=(fragment,),
        count_text=lambda text: len(text) // 4,
    )
    assert reserve == (10 // 4) + (10 // 4) + 300


def test_custom_count_text_affects_reserve() -> None:
    messages = [ChatMessage(role="user", content="abcd")]
    reserve_default = estimate_mandatory_reserve_tokens(
        base_messages=messages,
        collected_fragments=(),
        count_text=lambda text: len(text),
    )
    reserve_custom = estimate_mandatory_reserve_tokens(
        base_messages=messages,
        collected_fragments=(),
        count_text=lambda text: len(text) * 10,
    )
    assert reserve_default == 4
    assert reserve_custom == 40


def _assert_module_has_no_nexus_imports(relative_path: str) -> None:
    module_path = _REPO_ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.runtime.nexus"), (
                f"{relative_path} imports {node.module}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax.runtime.nexus"), (
                    f"{relative_path} imports {alias.name}"
                )


def test_context_budget_modules_do_not_import_nexus_runtime() -> None:
    budget_dir = _REPO_ROOT / "intergrax" / "context" / "budget"
    for path in sorted(budget_dir.glob("*.py")):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        _assert_module_has_no_nexus_imports(relative)
