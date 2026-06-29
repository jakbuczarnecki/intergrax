# Confluence (confluence)

Category: `wiki_knowledge`

## Single public entrypoint

- **`ConfluenceWikiKnowledgeIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ConfluenceWikiKnowledgeIntegration`.
- Contract factory: `create_confluence_wiki_knowledge_integration()`.
