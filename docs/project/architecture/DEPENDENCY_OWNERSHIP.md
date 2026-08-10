# Dependency ownership

This is the compact ownership foundation for the DEP roadmap. It classifies
dependencies by the runtime capability that owns them; a provider
implementation does not make its dependency a default-core dependency.

## Categories

- `CORE_FOUNDATION` — contracts, configuration, serialization, and primitives
  required by the canonical core import.
- `CORE_SERVER` — the default HTTP/runtime server surface unconditionally
  required by the core server.
- `PROVIDER_OPTIONAL` — vendor SDKs used only by a selected provider.
- `LOCAL_ML_OPTIONAL` — local model runtimes and model-loading libraries.
- `VECTOR_OPTIONAL` — external vector-store clients.
- `PARSER_OPTIONAL` — format-specific parsing libraries.
- `MEDIA_OPTIONAL` — audio, video, and image integrations.
- `UI_OPTIONAL` — interactive UI and presentation frameworks.
- `INTEGRATION_OPTIONAL` — external service and platform integrations.
- `DEV_QUALITY` — test, lint, type-check, and development tooling.
- `COMPATIBILITY_OPTIONAL` — legacy or compatibility adapters.

## Core invariant

A dependency belongs to the default/core installation only when:

1. a canonical core import requires it; or
2. the default runtime unconditionally requires it.

Use of a dependency by a provider-specific implementation alone does not
justify default-core ownership. Later DEP tasks must move such dependencies
behind the provider's explicit selection/import boundary and preserve a
controlled missing-dependency error.

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
