# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_notion_wiki_knowledge", "register_notion_integration"]

def __getattr__(name: str):
    if name == "register_notion_integration":
        from intergrax.integrations.providers.wiki_knowledge.notion.register import register_notion_integration
        return register_notion_integration
    if name == "create_notion_wiki_knowledge":
        from intergrax.integrations.providers.wiki_knowledge.notion.bundle import create_notion_wiki_knowledge
        return create_notion_wiki_knowledge
    raise AttributeError(name)
