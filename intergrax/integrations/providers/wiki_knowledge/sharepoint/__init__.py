# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_sharepoint_wiki_knowledge", "register_sharepoint_integration"]

def __getattr__(name: str):
    if name == "register_sharepoint_integration":
        from intergrax.integrations.providers.wiki_knowledge.sharepoint.register import register_sharepoint_integration
        return register_sharepoint_integration
    if name == "create_sharepoint_wiki_knowledge":
        from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import create_sharepoint_wiki_knowledge
        return create_sharepoint_wiki_knowledge
    raise AttributeError(name)
