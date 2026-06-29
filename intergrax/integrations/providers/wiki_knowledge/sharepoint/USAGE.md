# Sharepoint (sharepoint)

Category: `wiki_knowledge`

## Single public entrypoint

- **`SharepointWikiKnowledgeIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SharepointWikiKnowledgeIntegration`.
- Contract factory: `create_sharepoint_wiki_knowledge_integration()`.
