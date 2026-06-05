# © Artur Czarnecki. All rights reserved.

"""YAML contracts for deprecated ChatAgent router prompts (assets only, no runtime module)."""

from __future__ import annotations

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

pytestmark = pytest.mark.unit


def _assert_non_empty_str(value: str) -> None:
    assert isinstance(value, str)
    assert value.strip()


def _format_chat_router_system(
    *,
    tools_enabled: bool,
    tools_count: int,
    tools_description: str,
    general_description: str,
    routing_context: str | None,
) -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    template = (registry.resolve_localized("chat_router").system or "").rstrip("\n")
    txt = template.format(
        tools_state="ENABLED" if tools_enabled else "DISABLED",
        tools_count=tools_count,
        tools_description=tools_description,
        general_description=general_description,
    )
    if routing_context:
        txt += f"\nContext: {routing_context}"
    return txt


def _format_chat_router_user(
    *,
    question: str,
    rag_catalog_txt: str,
    tools_catalog_txt: str,
) -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    template = (registry.resolve_localized("chat_router_user").user_template or "").rstrip("\n")
    return template.format(
        question=question,
        rag_catalog_txt=rag_catalog_txt,
        tools_catalog_txt=tools_catalog_txt,
    )


def test_chat_router_yaml_registry_contains_prompts() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    assert registry.resolve_localized("chat_router")
    assert registry.resolve_localized("chat_router_user")


def test_chat_router_system_prompt_formatting() -> None:
    registry = YamlPromptRegistry.create_default(load=True)
    tools_description = (registry.resolve_localized("chat_router_tool").system or "").rstrip("\n")
    general_description = (registry.resolve_localized("chat_router_general").system or "").rstrip("\n")

    txt = _format_chat_router_system(
        tools_enabled=True,
        tools_count=3,
        tools_description=tools_description,
        general_description=general_description,
        routing_context="CTX",
    )

    _assert_non_empty_str(txt)
    assert "ENABLED" in txt
    assert "AVAILABLE=3" in txt
    assert tools_description in txt
    assert general_description in txt
    assert "Context: CTX" in txt


def test_chat_router_system_prompt_disabled_tools() -> None:
    txt = _format_chat_router_system(
        tools_enabled=False,
        tools_count=0,
        tools_description="T",
        general_description="G",
        routing_context=None,
    )

    assert "DISABLED" in txt
    assert "AVAILABLE=0" in txt


def test_chat_router_user_prompt_formatting() -> None:
    txt = _format_chat_router_user(
        question="Q?",
        rag_catalog_txt="RAG",
        tools_catalog_txt="TOOLS",
    )

    _assert_non_empty_str(txt)
    assert "User query:\nQ?" in txt
    assert "RAG components:\nRAG" in txt
    assert "Available tools:\nTOOLS" in txt
    assert '"q": "What is the weather in Warsaw?"' in txt
    assert '"route": "TOOLS"' in txt
    assert '"route": "RAG"' in txt
    assert '"route": "GENERAL"' in txt


def test_chat_router_user_prompt_contains_json_contract() -> None:
    txt = _format_chat_router_user(
        question="x",
        rag_catalog_txt="y",
        tools_catalog_txt="z",
    )

    assert "Output STRICT JSON ONLY" in txt
    assert '{"route":"RAG","rag_component":"intergrax_docs"}' in txt
