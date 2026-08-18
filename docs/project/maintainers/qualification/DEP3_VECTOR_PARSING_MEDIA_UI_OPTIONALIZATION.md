# DEP-3 — Vector, parsing, media, UI and MCP optionalization

Status: `READY_FOR_REVIEW`

## Scope and ownership

DEP-3 changes dependency ownership and lazy import boundaries only. RAG algorithms,
retrieval/indexing contracts, tenant/scope behavior, ranking, chunking and embedding
algorithms were not changed.

Direct owner evidence:

- Vector SDKs are imported only from `intergrax/integrations/providers/vector_store/*/opens.py`.
- `python-docx`, `openpyxl`/`xlrd`, PyMuPDF and OCR libraries are used by their
  document-parser opener modules.
- `yt-dlp`, VTT and Whisper are used by the media/parser opener modules;
  OpenCV is used by `intergrax/multimedia/video_loader.py` and the OpenCV vision adapter.
- Pillow is used by `intergrax/multimedia/image_smart_loader.py`.
- BeautifulSoup and Trafilatura are used by `intergrax/websearch/fetcher/extractor.py`.
- Streamlit has no canonical `intergrax` runtime import.
- FastMCP is loaded through `intergrax/applications/_shared/mcp_import_guard.py`;
  there are no direct `mcp` imports in the Intergrax runtime.
- `requests-cache` has no production import and was removed as an unused declaration.

## Core dependency surface

Before DEP-3, the core declaration also contained:

```text
chromadb==1.4.1
qdrant-client>=1.9
pinecone>=3.0
beautifulsoup4>=4.12
trafilatura>=1.8
python-docx>=1.1
openpyxl>=3.1
xlrd>=2.0
pytesseract>=0.3
pillow>=11.0
PyMuPDF>=1.23
requests-cache>=1.2
yt-dlp>=2024.0
webvtt-py>=0.4
opencv-python-headless==4.9.0.80
streamlit>=1.39
fastmcp>=3.3.1
mcp>=1.0
```

After DEP-3, core retains only the proven default owners from this area:

```text
numpy==1.26.4
pandas>=2.1.4,<3.0.0
chardet>=5.2,<6
```

The existing core owners `openai`, `boto3` and `tiktoken` remain unchanged.
`chardet` remains core because it is the fallback used by canonical native text
ingest; UTF-8 and UTF-8 BOM ingestion remain dependency-free beyond core.

## New and affected extras

All new extras are `EFFECTIVE`:

```text
vector-chroma   = chromadb==1.4.1
vector-qdrant   = qdrant-client>=1.9
vector-pinecone = pinecone>=3.0

parsing-web    = beautifulsoup4, trafilatura
parsing-office = python-docx, openpyxl, xlrd, docx2txt, langchain-community
parsing-pdf    = PyMuPDF, langchain-community
parsing-ocr    = pytesseract, pillow

media-youtube  = yt-dlp
media-video    = opencv-python-headless, webvtt-py
media-image    = pillow
media-ocr      = pytesseract, pillow
media-whisper  = openai-whisper, webvtt-py

ui-streamlit   = streamlit
mcp            = fastmcp
```

`langchain-community` and `docx2txt` in the office/PDF extras are required by
the existing parser loader implementations; they remain optional and do not
reintroduce LangChain into core. `fastmcp` pulls `mcp` transitively, so `mcp`
is no longer declared directly.

No new extra is `PARTIAL` or `BLOCKED_BY_CORE_OWNER`. Core-owned `pandas` and
`chardet` are intentionally not duplicated in parser extras.

## Lazy boundaries and controlled errors

- Chroma, Qdrant and Pinecone missing SDKs raise
  `IntegrationConfigurationError` naming `vector-chroma`, `vector-qdrant` or
  `vector-pinecone`.
- Python-docx, office engine, PyMuPDF, OCR, YouTube, video/OpenCV and MCP
  paths now name their owning extra in the controlled dependency error.
- Core imports:
  `import intergrax`, `import intergrax.runtime.nexus`, and
  `import intergrax.harness` load none of the optional capability modules.
- The in-memory vector path remains native and does not require a vector SDK.

## Clean-core qualification

Fresh Python 3.12 default install, with no extra:

```text
distributions:       45
site-packages:       171.46 MiB
Harness import:      1.875 s median (5 fresh subprocesses)
```

DEP-2 baseline:

```text
distributions:       179
site-packages:       ~764.75 MiB
Harness import:      ~2.35 s median
```

Passed clean-core checks:

```text
KnowledgeDocument
core integration/provider registration
in-memory vector manager
native text ingest
HarnessApplication construction
NexusLoop construction
```

The following distributions were absent from the fresh default environment:
ChromaDB, Qdrant client, Pinecone, Streamlit, FastMCP/MCP, yt-dlp, OpenCV,
Pytesseract, PyMuPDF, python-docx, openpyxl, xlrd, Trafilatura and BeautifulSoup.

## Extra matrix

Fresh `default + one extra` environments, no SaaS calls and no model downloads:

```text
vector-chroma   PASS — chromadb; opener import and SDK probe
vector-qdrant   PASS — qdrant-client; opener import and SDK probe
vector-pinecone PASS — pinecone; opener import and SDK probe
parsing-web     PASS — beautifulsoup4/trafilatura; HTML fixture
parsing-office  PASS — python-docx/openpyxl/xlrd; CSV fixture
parsing-pdf     PASS — PyMuPDF/langchain-community; generated PDF fixture
parsing-ocr     PASS — Pillow/pytesseract; runtime probe
media-youtube   PASS — yt-dlp; availability probe
media-video     PASS — OpenCV/webvtt; runtime import probe
media-image     PASS — Pillow; image module probe
media-ocr       PASS — Pillow/pytesseract; runtime probe
media-whisper   PASS — openai-whisper/webvtt; provider import probe
ui-streamlit    PASS — streamlit import
mcp             PASS — fastmcp and transitive mcp import
```

## Cross-extra matrix

```text
media-whisper + media-youtube   PASS — both availability probes
parsing-office + parsing-pdf    PASS — both SDK/parser surfaces
mcp + Harness                   PASS — MCP guard and HarnessApplication
vector-qdrant + rag-local-embeddings
                                PASS — resolver dry-run; no model download
```

Whisper does not pull YouTube or OpenCV. YouTube is selected independently;
VTT is included with Whisper because the existing Whisper provider exposes VTT
generation, and with video because video frame extraction consumes VTT.

## Tests and audits

```text
DEP-3 focused capability tests: 35 passed
websearch/media/MCP regression selection: passed
fresh clean-core smoke: passed
fresh extra matrix: 14/14 passed
cross-extra matrix: passed
```

The pre-existing vector bootstrap test
`test_create_vectorstore_from_integration_falls_back_to_inmemory` still fails
because it calls the current in-memory contract without the required explicit
`tenant_id`. It is outside DEP-3 and was not changed under the RAG fence.

Required audit commands:

```text
uv run python scripts/docs/validate_langchain_inventory.py
uv run python scripts/maintenance/check_langchain_boundary.py
uv lock --check
git diff --check
```

Expected invariants remain: zero core LangChain/LangGraph, default
`NativeOllamaAdapter`, canonical `KnowledgeDocument`, effective
`rag-local-embeddings`, `media-whisper` and `llm-*`.

## Unrelated concurrent work

Unstaged unrelated changes were observed in:

```text
infra/docker/postgresql/docker-compose.yml
intergrax/integrations/providers/vector_store/pgvector/opens.py
intergrax/integrations/providers/vector_store/pgvector/rag_store.py
```

They are preserved and excluded from DEP-3 staging.
