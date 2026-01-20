# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.chat_agent import (
    default_chat_router_system,
    default_chat_router_user,
)
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.chat_agent import ChatRouterConfig


pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _assert_non_empty_str(value: str) -> None:
    assert isinstance(value, str)
    assert value.strip()


# ----------------------------------------------------------------------
# Registry level contracts
# ----------------------------------------------------------------------

def test_chat_router_yaml_registry_contains_prompts() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    assert registry.resolve_localized("chat_router")
    assert registry.resolve_localized("chat_router_user")


# ----------------------------------------------------------------------
# Router system prompt
# ----------------------------------------------------------------------

def test_chat_router_system_prompt_formatting() -> None:
    cfg = ChatRouterConfig(
        tools_description="TOOLS_DESC",
        general_description="GENERAL_DESC",
    )
    cfg.ensure_prompts()

    txt = default_chat_router_system(
        tools_enabled=True,
        tools_count=3,
        router_cfg=cfg,
        routing_context="CTX",
    )

    _assert_non_empty_str(txt)

    assert "ENABLED" in txt
    assert "AVAILABLE=3" in txt
    assert "TOOLS_DESC" in txt
    assert "GENERAL_DESC" in txt
    assert "Context: CTX" in txt


def test_chat_router_system_prompt_disabled_tools() -> None:
    cfg = ChatRouterConfig(
        tools_description="T",
        general_description="G",
    )
    cfg.ensure_prompts()

    txt = default_chat_router_system(
        tools_enabled=False,
        tools_count=0,
        router_cfg=cfg,
        routing_context=None,
    )

    assert "DISABLED" in txt
    assert "AVAILABLE=0" in txt


# ----------------------------------------------------------------------
# Router user prompt
# ----------------------------------------------------------------------

def test_chat_router_user_prompt_formatting() -> None:
    txt = default_chat_router_user(
        question="Q?",
        rag_catalog_txt="RAG",
        tools_catalog_txt="TOOLS",
    )

    _assert_non_empty_str(txt)

    assert "User query:\nQ?" in txt
    assert "RAG components:\nRAG" in txt
    assert "Available tools:\nTOOLS" in txt

    # few-shot examples must be part of YAML content
    assert '"q": "What is the weather in Warsaw?"' in txt
    assert '"route": "TOOLS"' in txt
    assert '"route": "RAG"' in txt
    assert '"route": "GENERAL"' in txt


def test_chat_router_user_prompt_contains_json_contract() -> None:
    txt = default_chat_router_user(
        question="x",
        rag_catalog_txt="y",
        tools_catalog_txt="z",
    )

    assert "Output STRICT JSON ONLY" in txt
    assert '{"route":"RAG","rag_component":"intergrax_docs"}' in txt
