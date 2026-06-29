# Wikipedia (wikipedia)

Category: `wiki_knowledge`

## Single public entrypoint

- **`WikipediaWikiKnowledgeIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `WikipediaWikiKnowledgeIntegration`.
- Contract factory: `create_wikipedia_wiki_knowledge_integration()`.
