# Notion (notion)

Category: `wiki_knowledge`

## Single public entrypoint

- **`NotionWikiKnowledgeIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `NotionWikiKnowledgeIntegration`.
- Contract factory: `create_notion_wiki_knowledge_integration()`.
