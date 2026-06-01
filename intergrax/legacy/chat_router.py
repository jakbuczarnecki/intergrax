# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Legacy chat router config (extracted from deprecated ChatAgent, Phase Q-X.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

Route = Literal["rag", "tools", "general"]


@dataclass
class ChatRouterConfig:
    """LLM router configuration (descriptive, not hard rules)."""

    use_llm_router: bool = True
    router_max_tokens: int = 256
    router_temperature: float = 0.0
    tools_description: str = ""
    general_description: str = ""
    allow_override: bool = True

    def ensure_prompts(self) -> None:
        registry = YamlPromptRegistry.create_default(load=True)

        def system(id_: str) -> str:
            return (registry.resolve_localized(id_).system or "").rstrip("\n")

        if not self.general_description:
            self.general_description = system("chat_router_general")
        if not self.tools_description:
            self.tools_description = system("chat_router_tool")


def default_chat_router_system(
    tools_enabled: bool,
    tools_count: int,
    router_cfg: ChatRouterConfig,
    routing_context: Optional[str],
) -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    template = (registry.resolve_localized("chat_router").system or "").rstrip("\n")
    txt = template.format(
        tools_state="ENABLED" if tools_enabled else "DISABLED",
        tools_count=tools_count,
        tools_description=router_cfg.tools_description,
        general_description=router_cfg.general_description,
    )
    if routing_context:
        txt += f"\nContext: {routing_context}"
    return txt


def default_chat_router_user(
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
