# RAG skill bundle

**Bundle id:** `rag` · **Status:** STABLE · **Plugin:** `RagSkillPlugin`

Composable packs for harness RAG retrieval and document ingestion. Register via `register_rag_skill_bundle()` or enable bundle `rag` on `SkillProfile`.

## Skills in this bundle

| skill_id | Guide |
|----------|-------|
| `rag.hybrid_qa` | [rag.hybrid_qa/USAGE.md](rag.hybrid_qa/USAGE.md) |
| `rag.document_ingest` | [rag.document_ingest/USAGE.md](rag.document_ingest/USAGE.md) |

## Tier-3 preset

`rag_skill_profile()` · also included in `lkw_skill_profile()`, `lab_skill_profile()`, `legal_skill_profile()`, `platform_skill_profile()`.

## Registration

```python
from intergrax.skills.providers.rag.bundle import register_rag_skill_bundle
register_rag_skill_bundle()
```
