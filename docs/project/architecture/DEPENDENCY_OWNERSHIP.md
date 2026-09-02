# Dependency ownership

This is the compact ownership foundation for the DEP roadmap. It classifies
dependencies by the runtime capability that owns them; a provider
implementation does not make its dependency a default-core dependency.

## Categories

- `CORE_FOUNDATION` - contracts, configuration, serialization, and primitives
  required by the canonical core import.
- `CORE_SERVER` - the default HTTP/runtime server surface unconditionally
  required by the core server.
- `PROVIDER_OPTIONAL` - vendor SDKs used only by a selected provider.
- `LOCAL_ML_OPTIONAL` - local model runtimes and model-loading libraries.
- `VECTOR_OPTIONAL` - external vector-store clients.
- `PARSER_OPTIONAL` - format-specific parsing libraries.
- `MEDIA_OPTIONAL` - audio, video, and image integrations.
- `UI_OPTIONAL` - interactive UI and presentation frameworks.
- `INTEGRATION_OPTIONAL` - external service and platform integrations.
- `DEV_QUALITY` - test, lint, type-check, and development tooling.
- `COMPATIBILITY_OPTIONAL` - legacy or compatibility adapters.

## Core invariant

A dependency belongs to the default/core installation only when:

1. a canonical core import requires it; or
2. the default runtime unconditionally requires it.

Use of a dependency by a provider-specific implementation alone does not
justify default-core ownership. Later DEP tasks must move such dependencies
behind the provider's explicit selection/import boundary and preserve a
controlled missing-dependency error.

## DEP-4 version policy

Direct runtime dependencies use one of three visible policies:

- `EXACT_PIN` - reserved for reproducibility-sensitive packages such as the
  pinned NumPy baseline and Chroma provider.
- `BOUNDED_MAJOR` - a supported lower bound plus an upper boundary before the
  next incompatible major family.
- `UNBOUNDED_MAJOR` - permitted only for packages outside the designated
  high-risk runtime/optional policy set.

FastAPI, Starlette, Uvicorn and HTTPX are bounded below major `1` as one
compatibility family. Pydantic is bounded below major `3`; it must not enter
the resolver unqualified. FastMCP is qualified on major `3` and bounded below
major `4`; its `mcp` dependency remains transitive and is not declared
directly.

## Core allowlist and optional capability principle

The default dependency list is an explicit reviewed allowlist in
`scripts/maintenance/check_dependency_ownership.py`. Every core entry must
have a `CORE_FOUNDATION` or `CORE_SERVER` owner. Capability packages such as
provider SDKs, vector clients, local ML, parsers, media, UI, FastMCP and
LangChain/LangGraph belong in a named extra unless a documented temporary
core owner is approved.

Intentional shared declarations (`openai`, `boto3`, `tiktoken` and the Harness
authoring server set) remain allowed because their core owner is independently
qualified. `pandas` is owned by `parsing-office`; `chardet` remains core for
the canonical native text-loader fallback.

## Governance gate and change procedure

The offline gate at
`scripts/maintenance/check_dependency_ownership.py` parses `pyproject.toml`
and rejects unapproved core entries, forbidden optional-family reintroduction,
LangChain/LangGraph in core or `llm-all`, direct `mcp`, accidental core/extra
duplicate ownership, and unbounded high-risk declarations. It runs in the
lightweight governance smoke job in `.github/workflows/unit-tests.yml`.

A dependency change must update its owner/extra and version bound together,
then pass the checker, targeted declaration tests, `uv lock --check`, and the
relevant fresh resolver matrix. Adding a new core owner requires an explicit
change to the allowlist and qualification evidence; the checker has no
baseline-update or global-ignore mode.

## DEP-2 decisions

DEP-2 makes the native LLM extras and `rag-local-embeddings` actual selection
boundaries:

- `torch`, `sentence-transformers`, and `transformers` are
  `LOCAL_ML_OPTIONAL` and owned by `rag-local-embeddings`.
- `openai-whisper` is `MEDIA_OPTIONAL` and owned by `media-whisper`. DEP-2
  moves this one media dependency early because its transitive Torch ownership
  blocked the clean-core local-ML invariant; no other DEP-3 media or parser
  ownership moves are included.
- `anthropic`, `mistralai`, `ollama`, `google-genai`, and `cohere` are
  `PROVIDER_OPTIONAL` and owned by their native `llm-*` extras.
- `mypy-boto3-bedrock-runtime` has no runtime owner after the Bedrock adapter
  uses one local minimal protocol. `boto3` remains `CORE_SERVER` /
  `INTEGRATION_OPTIONAL` because AWS and S3 integration paths import it.
- `openai` remains core because native embedding and vector-related paths use
  it directly. The OpenAI-family extras therefore remain explicit provider
  metadata but are blocked from becoming sole ownership boundaries until that
  core owner is addressed.
- `tiktoken` remains `CORE_FOUNDATION` because the base LLM adapter performs
  canonical token accounting through it.

`llm-all` contains native SDKs only; LangChain remains confined to
`llm-langchain-ollama` and the existing RAG compatibility extras.
