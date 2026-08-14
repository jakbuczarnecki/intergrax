# Intergrax Platform Configuration

Authoritative reference for **platform-wide** Intergrax configuration: what each
option does, whether it is required, its default, accepted values, related
settings, and ownership.

This document describes **active runtime contracts**. Typed settings and
profiles win over comments in `.env.example`. Application-specific settings
are out of scope — see [What is not in this document?](#what-is-not-in-this-document).

**Related:** [LLM adapters](../../architecture/LLM_ADAPTERS.md) (provider catalog) ·
[Harness environment](HARNESS_ENVIRONMENT.md) (lab stack / OTLP presets) ·
root [`.env.example`](../../../../.env.example) (copyable template)

---

## How configuration works

Intergrax reads configuration from the **process environment**. The library does
not load `.env` by itself — your application entrypoint, Compose file, or
operator shell must export variables (many hosts use `python-dotenv`).

### Precedence

```text
1. Explicit constructor / composition-root overrides in the application host
2. Typed profiles built from environment
   (LLMProfile, EmbeddingProfile, IntegrationProfile, GlobalSettings, …)
3. Platform profile defaults when the env var is unset
4. Provider adapter fallbacks when a selected model is omitted
```

An application may wrap the same platform keys (for example a host factory that
calls `llm_profile_from_env()`) or replace them with a typed `LLMProfile` in
code. That is an **application override**, not a second platform contract.

### Four kinds of “default”

These are not interchangeable. Every value below is labelled as one of:

| Kind | Meaning |
|------|---------|
| **Platform profile default** | Value the typed platform profile uses when the env var is unset |
| **Provider adapter fallback** | Value a provider adapter uses when the profile did not pass a model or URL |
| **Example value** | Copyable illustration in `.env.example` — not a runtime default |
| **Application override** | Host- or product-specific setting (not documented here) |

### Secrets

Treat API keys, tokens, signing secrets, and webhook URLs as **secrets**:

- Never commit real secrets. `.env` is local/runtime configuration and is gitignored.
- [`.env.example`](../../../../.env.example) contains placeholders and comments only.
- Prefer a secrets store (`llm/<provider>/api_key`) in production; env vars are
  the local/dev path.

---

## Quick reference

Compact catalog of platform selection and infrastructure keys. Provider
connection/auth keys follow in the detailed sections.

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `INTERGRAX_DEFAULT_LANGUAGE` | Default language for prompts and extracted content | Platform profile: `pl` | No |
| `INTERGRAX_DEFAULT_LOCALE` | Default locale fallback | Platform profile: `pl-PL` | No |
| `INTERGRAX_DEFAULT_REGION` | Default region/market fallback | Platform profile: `pl-PL` | No |
| `INTERGRAX_DEFAULT_TIMEZONE` | Default timezone fallback | Platform profile: `Europe/Warsaw` | No |
| `INTERGRAX_DEFAULT_USER_TURNS_CONSOLIDATION_INTERVAL` | Session memory consolidation interval (user turns) | Platform profile: none | No |
| `INTERGRAX_DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS` | Consolidation cooldown | Platform profile: none | No |
| `INTERGRAX_LLM_PROVIDER` | Generation LLM adapter/provider | Platform profile: `ollama` | No (has default) |
| `INTERGRAX_LLM_MODEL` | Generation model id | Platform profile: none | Conditional |
| `INTERGRAX_EMBEDDING_PROVIDER` | Embedding adapter/provider | Platform profile: `ollama` | No (has default) |
| `INTERGRAX_EMBEDDING_MODEL` | Embedding model id | Platform profile: none | Conditional |
| `INTERGRAX_LLM_TENANT_MAX_TOKENS` | Per-tenant generation token budget | Platform profile: `0` (disabled) | No |
| `INTERGRAX_DOCLING_MODE` | Document parser mode | Platform profile: `local` | No |
| `INTERGRAX_DOCLING_SERVER_URL` | Docling server origin | Platform profile: `http://localhost:8000` | Conditional |
| `INTERGRAX_DOCLING_SERVER_PATH` | Docling server path | Platform profile: `/parse` | Conditional |
| `INTERGRAX_DOCLING_TIMEOUT` | Docling timeout (seconds) | Platform profile: `120` | No |
| `INTERGRAX_INTEGRATION_<CATEGORY>` | Integration provider slug for a category | none (unset = no env selection) | Conditional |
| `INTERGRAX_NOTIFICATION_BACKEND` | Runtime notification backend | Platform profile: `log` | No |
| `INTERGRAX_INTERACTION_SURFACE` | Inbound interaction surface | Platform profile: `auto` | No |
| `INTERGRAX_INBOUND_VERIFIER` | Inbound signature verifier | Platform profile: `none` | Conditional |
| `INTERGRAX_ENV` | Host environment name when the app prefix is unset | Platform profile: `dev` | No |
| `INTERGRAX_HARNESS_API_KEY` | Optional harness HTTP API key | none | Conditional |
| `INTERGRAX_SQLITE_DATA_DIR` | Directory for default SQLite store files | Platform profile: `build` | No |
| `INTERGRAX_SHADOW_ROOT` | Shadow workspace root | Platform profile: `build/shadow_workspaces` | No |
| `INTERGRAX_SANDBOX_ROOT` | Sandbox session root | Platform profile: `build/sandbox_sessions` | No |
| `INTERGRAX_SCHEDULER_POLL_SECONDS` | Long-running scheduler poll interval | Platform profile: none; adapter fallback `30` | No |
| `INTERGRAX_CELERY_BROKER_URL` | Worker/broker URL | none | Conditional |
| `INTERGRAX_EXPORT_JOURNAL` | Export run-journal OTLP snapshot on task completion | Platform profile: enabled (`1`) | No |
| `INTERGRAX_EXPORT_PARSER_TRACE` | Export document-parser traces to the observability backend | Platform profile: off | No |

---

## Global defaults

Owner: **platform**. Used as framework-wide fallbacks when no user/org override
is present.

### INTERGRAX_DEFAULT_LANGUAGE

Purpose: Default language for prompts, instructions, and extracted content.

Owner: Platform.

Default: Platform profile default: `pl`.

Required: No.

Accepted values: Language code string (for example `pl`, `en`).

Example:

```text
INTERGRAX_DEFAULT_LANGUAGE=en
```

Related settings: `INTERGRAX_DEFAULT_LOCALE`, `INTERGRAX_DEFAULT_REGION`.

Used by: Global platform defaults.

### INTERGRAX_DEFAULT_LOCALE

Purpose: Default locale when no user or organisation override is available.

Owner: Platform.

Default: Platform profile default: `pl-PL`.

Required: No.

Example:

```text
INTERGRAX_DEFAULT_LOCALE=en-US
```

### INTERGRAX_DEFAULT_REGION

Purpose: Default country/market context. Fallback only.

Owner: Platform.

Default: Platform profile default: `pl-PL`.

Required: No.

Example:

```text
INTERGRAX_DEFAULT_REGION=en-US
```

### INTERGRAX_DEFAULT_TIMEZONE

Purpose: Default timezone when no user or organisation value is configured.

Owner: Platform.

Default: Platform profile default: `Europe/Warsaw`.

Required: No.

Accepted values: IANA timezone name.

Example:

```text
INTERGRAX_DEFAULT_TIMEZONE=UTC
```

### INTERGRAX_DEFAULT_USER_TURNS_CONSOLIDATION_INTERVAL

Purpose: How often (in user turns) mid-session memory consolidation runs.

Owner: Platform.

Default: Platform profile default: none (unset or non-digit → treated as disabled).

Required: No.

Accepted values: Digits only.

Example:

```text
INTERGRAX_DEFAULT_USER_TURNS_CONSOLIDATION_INTERVAL=8
```

### INTERGRAX_DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS

Purpose: Minimum seconds between consolidations.

Owner: Platform.

Default: Platform profile default: none.

Required: No.

Accepted values: Digits only.

Example:

```text
INTERGRAX_DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS=60
```

---

## Generation LLM

Canonical selection is a **pair**. The provider slug chooses the adapter; the
model string chooses the model. Connection and auth settings are independent:
switching `INTERGRAX_LLM_PROVIDER` does not by itself change API keys or host
URLs, and those connection settings do not select the model.

```text
INTERGRAX_LLM_PROVIDER  →  which adapter/provider
INTERGRAX_LLM_MODEL     →  which model id
```

Supported built-in provider slugs and adapter behaviour:
[LLM adapters — provider selection](../../architecture/LLM_ADAPTERS.md#provider-selection).
Do not duplicate that catalog here.

An application host may pass an explicit `LLMProfile` in code. That is an
**application override** of this platform pair.

### INTERGRAX_LLM_PROVIDER

Purpose: Selects the generation LLM adapter (Ollama, OpenAI, vLLM, …).

Owner: Platform.

Default: Platform profile default: `ollama`.

Required: No — the profile default applies when unset.

Accepted values: Built-in slugs:

`openai`, `gemini`, `ollama`, `mistral`, `claude`, `azure_openai`,
`aws_bedrock`, `groq`, `vllm`, `together`, `fireworks`, `openrouter`,
`deepseek`, `xai`, `llama_cpp`, `cohere`, `cohere_native`, `vertex_gemini`,
`azure_ai_inference`

Custom slugs are valid only after they are registered on the LLM adapter
registry.

Example:

```text
INTERGRAX_LLM_PROVIDER=ollama
```

Related settings: `INTERGRAX_LLM_MODEL`; provider connection/auth keys below.

Used by: Generation / conversation LLM selection for the platform runtime.

Notes: Independent from embedding selection (`INTERGRAX_EMBEDDING_*`).

### INTERGRAX_LLM_MODEL

Purpose: Selects the generation model id for the chosen provider.

Owner: Platform.

Default: Platform profile default: **none**. If omitted, adapters use their
own code-level constructor default (for example Ollama `llama3.1:latest`).

Required: Conditional. Set it whenever you need a specific model.

Accepted values: Free string (provider-specific model id). No platform model enum.

Example:

```text
INTERGRAX_LLM_MODEL=llama3.1:latest
```

Related settings: `INTERGRAX_LLM_PROVIDER`.

Used by: Generation LLM. Passed into the adapter as `model` when set.

Provider-specific `INTERGRAX_DEFAULT_*_MODEL` environment variables are **not**
supported for generation model selection. Use `INTERGRAX_LLM_PROVIDER` and
`INTERGRAX_LLM_MODEL` only.

### INTERGRAX_LLM_TENANT_MAX_TOKENS

Purpose: Hard per-tenant cap on cumulative generation tokens.

Owner: Platform.

Default: Platform profile default: `0` (disabled).

Required: No.

Accepted values: Non-negative integer. `0` disables the quota.

Example:

```text
INTERGRAX_LLM_TENANT_MAX_TOKENS=2000000
```

Used by: Generation LLM governance.

---

## Generation LLM — provider connection and auth

These settings connect to a provider. They do **not** select `INTERGRAX_LLM_PROVIDER`
or `INTERGRAX_LLM_MODEL`. Set only what the chosen provider needs.

Secrets are marked **(secret)**.

### Local / self-hosted

#### OLLAMA_HOST

Purpose: Conventional Ollama HTTP base URL used by Ollama clients (including
the embedding client). Not a field on `LLMProfile`.

Owner: Provider connection.

Default: Provider adapter / client fallback: typically `http://127.0.0.1:11434`.

Required: No for local default; yes if Ollama is not on that URL.

Example:

```text
OLLAMA_HOST=http://127.0.0.1:11434
```

Related settings: `INTERGRAX_LLM_PROVIDER=ollama`, `INTERGRAX_EMBEDDING_PROVIDER=ollama`.

#### INTERGRAX_DEFAULT_VLLM_BASE_URL

Purpose: OpenAI-compatible base URL for the vLLM **chat** server.

Owner: Provider connection.

Default: Provider adapter fallback: `http://127.0.0.1:8000/v1`. Example (Docker
host map): `http://127.0.0.1:8100/v1`.

Required: Conditional — when using `vllm` and the server is not on the adapter default.

Example:

```text
INTERGRAX_DEFAULT_VLLM_BASE_URL=http://127.0.0.1:8100/v1
```

#### VLLM_API_KEY

Purpose: Optional API key for vLLM. **(secret)**

Owner: Provider connection.

Default: none. Adapter treats the key as optional; local servers often use `EMPTY`.

Required: No for typical local vLLM.

#### INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL

Purpose: OpenAI-compatible base URL for the llama.cpp **chat** server.

Owner: Provider connection.

Default: Provider adapter fallback: `http://127.0.0.1:8080/v1`. Example (Docker
host map): `http://127.0.0.1:8102/v1`.

Required: Conditional.

#### LLAMA_CPP_API_KEY

Purpose: Optional API key for llama.cpp. **(secret)**

Owner: Provider connection.

Default: none. Adapter treats the key as optional.

Required: No for typical local llama.cpp.

### Cloud API keys

| Variable | Provider slug(s) | Default | Required |
|----------|------------------|---------|----------|
| `OPENAI_API_KEY` **(secret)** | `openai` (also OpenAI embeddings) | none | Yes when that provider is used |
| `ANTHROPIC_API_KEY` **(secret)** | `claude` | none | Yes when used |
| `GOOGLE_API_KEY` **(secret)** | `gemini` | none | Yes when used |
| `MISTRAL_API_KEY` **(secret)** | `mistral` | none | Yes when used |
| `GROQ_API_KEY` **(secret)** | `groq` | none | Yes when used |
| `TOGETHER_API_KEY` **(secret)** | `together` | none | Yes when used |
| `FIREWORKS_API_KEY` **(secret)** | `fireworks` | none | Yes when used |
| `OPENROUTER_API_KEY` **(secret)** | `openrouter` | none | Yes when used |
| `DEEPSEEK_API_KEY` **(secret)** | `deepseek` | none | Yes when used |
| `XAI_API_KEY` **(secret)** | `xai` | none | Yes when used |
| `COHERE_API_KEY` **(secret)** | `cohere`, `cohere_native` (also Cohere rerank) | none | Yes when used |
| `AZURE_AI_INFERENCE_API_KEY` **(secret)** | `azure_ai_inference` | none | Yes when used |

Optional OpenAI-compatible base URL overrides (adapter fallbacks are the public
API origins): `INTERGRAX_DEFAULT_GROQ_BASE_URL`,
`INTERGRAX_DEFAULT_TOGETHER_BASE_URL`, `INTERGRAX_DEFAULT_FIREWORKS_BASE_URL`,
`INTERGRAX_DEFAULT_OPENROUTER_BASE_URL`, `INTERGRAX_DEFAULT_DEEPSEEK_BASE_URL`,
`INTERGRAX_DEFAULT_XAI_BASE_URL`, `INTERGRAX_DEFAULT_COHERE_BASE_URL`,
`INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_BASE_URL` (required for `azure_ai_inference`).

### Azure OpenAI

| Variable | Purpose | Default | Required when `azure_openai` |
|----------|---------|---------|------------------------------|
| `AZURE_OPENAI_API_KEY` **(secret)** | Azure OpenAI SDK key | none | Yes |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT` | Azure resource endpoint | none | Yes |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION` | API version | none | Yes |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT` | Deployment name (adapter fallback if model omitted) | none | Yes |

### AWS Bedrock

| Variable | Purpose | Default | Required when `aws_bedrock` |
|----------|---------|---------|------------------------------|
| `INTERGRAX_DEFAULT_AWS_REGION` | AWS region | none | Yes |
| `INTERGRAX_LLM_MODEL` | Bedrock model id | none | Yes |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Use Bedrock Converse API when truthy | none (off) | No |

Auth uses the standard AWS credential chain (`AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY`, SSO, or instance role) — not Intergrax-prefixed keys.

### Vertex Gemini

| Variable | Purpose | Default | Required when `vertex_gemini` |
|----------|---------|---------|-------------------------------|
| `INTERGRAX_VERTEX_PROJECT` | GCP project | none | Yes |
| `INTERGRAX_DEFAULT_VERTEX_LOCATION` | Vertex location | `us-central1` | No |
| `INTERGRAX_LLM_MODEL` | Gemini model id on Vertex | Adapter code default: `gemini-2.5-flash` when omitted | No |

Uses Application Default Credentials (no API key env).

---

## Embeddings

Canonical selection is a **pair**, independent from generation LLM selection.

```text
INTERGRAX_EMBEDDING_PROVIDER  →  which embedding adapter
INTERGRAX_EMBEDDING_MODEL     →  which embedding model
```

Accepted providers: `ollama`, `openai`, `hf`, `vllm`, `llama_cpp`.

The following names are **not** current supported model-selection configuration.
Do not set them; they are not read as the embedding selection contract:

- `INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL`
- `INTERGRAX_DEFAULT_VLLM_EMBED_MODEL`
- `INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL`
- `INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL`
- `INTERGRAX_DEFAULT_HF_EMBED_MODEL`
- `INTERGRAX_RAG_EMBEDDING_PROVIDER`

### INTERGRAX_EMBEDDING_PROVIDER

Purpose: Selects the embedding provider used to convert text into vectors.

Owner: Platform.

Default: Platform profile default: `ollama`.

Required: No — the profile default applies when unset.

Accepted values: `ollama` · `openai` · `hf` · `vllm` · `llama_cpp`.

Example:

```text
INTERGRAX_EMBEDDING_PROVIDER=ollama
```

Related settings: `INTERGRAX_EMBEDDING_MODEL`; provider connection settings below.

Used by: RAG / document embedding.

Notes: Changing the generation LLM does not change this provider.

### INTERGRAX_EMBEDDING_MODEL

Purpose: Selects the embedding model id for the chosen provider.

Owner: Platform.

Default: Platform profile default: **none**. If omitted, the selected provider
uses its **provider adapter fallback**:

| Provider | Adapter fallback model |
|----------|------------------------|
| `ollama` | `nomic-embed-text` |
| `openai` | `text-embedding-3-small` |
| `hf` | `sentence-transformers/all-MiniLM-L6-v2` |
| `vllm` | `BAAI/bge-small-en-v1.5` |
| `llama_cpp` | `default` |

`.env.example` uses the **example value** `nomic-embed-text` for local Ollama.
That example matches the Ollama adapter fallback; it is still not the platform
profile default (the profile default remains none).

Required: Conditional. Set it when the adapter fallback is not the model you want.

Example:

```text
INTERGRAX_EMBEDDING_MODEL=nomic-embed-text
```

Related settings: `INTERGRAX_EMBEDDING_PROVIDER`.

Used by: RAG / document embedding.

### Embedding provider connection

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `OLLAMA_HOST` | Ollama embed client host | Client fallback `http://127.0.0.1:11434` | Conditional |
| `OPENAI_API_KEY` **(secret)** | OpenAI embeddings (official client) | none | Yes when `openai` |
| `INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL` | vLLM **embedding** server URL | none, then falls back to `INTERGRAX_DEFAULT_VLLM_BASE_URL` | Conditional |
| `VLLM_API_KEY` **(secret)** | vLLM embed key | Adapter fallback `EMPTY` | No locally |
| `INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL` | llama.cpp **embedding** server URL | none, then falls back to `INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL` | Conditional |
| `LLAMA_CPP_API_KEY` **(secret)** | llama.cpp embed key | Adapter fallback `EMPTY` | No locally |

`hf` loads a local sentence-transformers model; no Intergrax API-key contract.

Example (separate vLLM embed server):

```text
INTERGRAX_EMBEDDING_PROVIDER=vllm
INTERGRAX_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL=http://127.0.0.1:8101/v1
```

---

## Document parsing (Docling)

Owner: **platform integration** (document parser). Used when the document-parser
integration is Docling.

### INTERGRAX_DOCLING_MODE

Purpose: How Intergrax parses documents (local Docling, remote server, or off).

Owner: Platform integration.

Default: Platform profile default: `local`.

Required: No.

Accepted values: `none` · `local` · `server`. Invalid values fail at startup.

Example:

```text
INTERGRAX_DOCLING_MODE=local
```

Related settings: `INTERGRAX_DOCLING_SERVER_URL`, `INTERGRAX_DOCLING_SERVER_PATH`,
`INTERGRAX_DOCLING_TIMEOUT`, `INTERGRAX_INTEGRATION_DOCUMENT_PARSER`.

### INTERGRAX_DOCLING_SERVER_URL

Purpose: Origin of a Docling parse server.

Owner: Platform integration.

Default: Platform profile default: `http://localhost:8000`.

Required: Conditional — when `INTERGRAX_DOCLING_MODE=server`.

Example:

```text
INTERGRAX_DOCLING_SERVER_URL=http://localhost:8000
```

### INTERGRAX_DOCLING_SERVER_PATH

Purpose: HTTP path on the Docling server.

Owner: Platform integration.

Default: Platform profile default: `/parse`.

Required: Conditional — when mode is `server`.

Example:

```text
INTERGRAX_DOCLING_SERVER_PATH=/parse
```

### INTERGRAX_DOCLING_TIMEOUT

Purpose: Parser timeout in seconds.

Owner: Platform integration.

Default: Platform profile default: `120`.

Required: No.

Example:

```text
INTERGRAX_DOCLING_TIMEOUT=120
```

---

## Integration provider selection

Owner: **platform integration**. Pattern:

```text
INTERGRAX_INTEGRATION_<CATEGORY>=<catalog-slug>
```

`<CATEGORY>` is an `IntegrationCategory` value in uppercase, for example
`RELATIONAL_STORE`. Unset categories are left empty unless the application
supplies an `IntegrationProfile` in code (**application override**).

Categories:

`relational_store`, `document_store`, `key_value_cache`, `message_bus`,
`object_storage`, `vector_store`, `search_provider`, `notification_channel`,
`conversation_channel`, `collaboration_suite`, `issue_tracker`,
`wiki_knowledge`, `observability_backend`, `browser_automation`,
`cloud_platform`, `secrets_store`, `graph_store`, `document_parser`,
`rerank_provider`, `feature_flag`, `ci_cd`, `security_scanner`,
`sandbox_host`, `identity_provider`, `speech_provider`,
`workflow_orchestrator`, `vision_serving`, `ml_inference_host`,
`model_serving_runtime`, `billing_meter`, `crm`, `llm_guardrail`,
`external_work`

Default: none (no env selection). Lab composition often sets slugs in code
(`sqlite`, `docling`, `log`, …) — that is composition, not an env default.

Required: Conditional — required for a category only if the running host
resolves that category from env rather than from a coded profile.

Example:

```text
INTERGRAX_INTEGRATION_RELATIONAL_STORE=sqlite
INTERGRAX_INTEGRATION_DOCUMENT_PARSER=docling
INTERGRAX_INTEGRATION_VECTOR_STORE=chroma
INTERGRAX_INTEGRATION_RERANK_PROVIDER=cohere_rerank
INTERGRAX_INTEGRATION_SEARCH_PROVIDER=google_cse
INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND=langfuse
```

Per-slug connection options (Redis URL, Qdrant endpoint, SendGrid, …) live with
each integration provider. This catalog documents the **selection** contract and
the widely used credential keys below. A full per-slug env inventory is not
duplicated here.

### Search and rerank credentials

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `INTERGRAX_GOOGLE_CSE_API_KEY` **(secret)** | Google CSE key (preferred) | none | Yes when Google CSE is used |
| `GOOGLE_CSE_API_KEY` **(secret)** | Legacy alias for the CSE key | none | Fallback if the Intergrax-prefixed key is unset |
| `INTERGRAX_GOOGLE_CSE_CX` | Google CSE engine id (preferred) | none | Yes when Google CSE is used |
| `GOOGLE_CSE_CX` | Legacy alias for the engine id | none | Fallback |
| `INTERGRAX_BING_API_KEY` **(secret)** | Bing Web Search key (preferred) | none | Yes when Bing is used |
| `BING_SEARCH_V7_API_KEY` **(secret)** | Legacy alias | none | Fallback |
| `COHERE_API_KEY` **(secret)** | Cohere rerank (and Cohere LLM) | none | Yes when Cohere rerank/LLM is used |
| `INTERGRAX_DEFAULT_COHERE_RERANK_MODEL` | Cohere rerank adapter fallback model | Adapter fallback: `rerank-english-v3.0` | No |
| `JINA_API_KEY` **(secret)** | Jina rerank | none | Yes when Jina rerank is used |
| `INTERGRAX_DEFAULT_JINA_RERANK_MODEL` | Jina rerank adapter fallback model | Adapter fallback: `jina-reranker-v1-base-en` | No |

Rerank **model** variables are adapter fallbacks, analogous to
`INTERGRAX_DEFAULT_*_MODEL` for generation. Provider **selection** is
`INTERGRAX_INTEGRATION_RERANK_PROVIDER`.

---

## Persistence / runtime stores

Owner: **runtime infrastructure**. Default files are created under
`INTERGRAX_SQLITE_DATA_DIR` (platform profile default: `build`).

### INTERGRAX_SQLITE_DATA_DIR

Purpose: Directory that holds default SQLite filenames for runtime stores.

Owner: Runtime infrastructure.

Default: Platform profile default: `build`.

Required: No.

Example:

```text
INTERGRAX_SQLITE_DATA_DIR=/var/lib/intergrax
```

Related settings: per-store `INTERGRAX_*_DB` path overrides below.

### Per-store SQLite paths

Each path may be set to a full file path. If unset, the file lives under the
data directory with the listed filename.

| Variable | Store | Default filename under data dir |
|----------|-------|----------------------------------|
| `INTERGRAX_RELATIONAL_DB` | Relational / app SQLite | `intergrax.db` |
| `INTERGRAX_TRACE_DB` | Run trace | `intergrax_trace.db` |
| `INTERGRAX_RUNTIME_EVENTS_DB` | Runtime events | `intergrax_runtime_events.db` |
| `INTERGRAX_TASK_CHECKPOINTS_DB` | Task checkpoints | `intergrax_task_checkpoints.db` |
| `INTERGRAX_TASK_MEMORY_DB` | Task memory | `intergrax_task_memory.db` |
| `INTERGRAX_HUMAN_DECISIONS_DB` | Human decisions (HITL) | `intergrax_human_decisions.db` |
| `INTERGRAX_EXPERIMENTS_DB` | Experiment registry | `intergrax_experiments.db` |
| `INTERGRAX_IDEMPOTENCY_DB` | Idempotency ledger | `intergrax_idempotency.db` |
| `INTERGRAX_SESSION_DB` | Sessions | `intergrax_session.db` |
| `INTERGRAX_ORGANIZATION_DB` | Organisation records | `intergrax_organization.db` |
| `INTERGRAX_USER_PROFILE_DB` | User profiles | `intergrax_user_profile.db` |

Required: No. Example:

```text
INTERGRAX_TRACE_DB=/var/lib/intergrax/trace.db
```

---

## Notifications

Owner: **platform** runtime notifications (log-only by default; no network).

### INTERGRAX_NOTIFICATION_BACKEND

Purpose: Selects where runtime notifications are sent.

Owner: Platform.

Default: Platform profile default: `log`. Unknown values fall back to `log`.

Required: No.

Accepted values: `log` · `webhook` · `slack` · `teams` · `pagerduty` · `opsgenie`.

Example:

```text
INTERGRAX_NOTIFICATION_BACKEND=log
```

Related settings: webhook/Slack/Teams URLs; PagerDuty routing key.

### INTERGRAX_WEBHOOK_URL

Purpose: Destination URL for the generic webhook backend. **(secret)** if the
URL embeds a token.

Owner: Provider connection.

Default: none.

Required: Conditional — when backend is `webhook`.

Example:

```text
INTERGRAX_WEBHOOK_URL=https://hooks.example.com/intergrax
```

### INTERGRAX_SLACK_WEBHOOK_URL

Purpose: Slack incoming-webhook URL for the Slack notification backend. **(secret)**

Owner: Provider connection.

Default: none.

Required: Conditional — when backend is `slack`.

### INTERGRAX_TEAMS_WEBHOOK_URL

Purpose: Microsoft Teams webhook URL. **(secret)**

Owner: Provider connection.

Default: none.

Required: Conditional — when backend is `teams`.

### PagerDuty

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `INTERGRAX_PAGERDUTY_ROUTING_KEY` **(secret)** | Events API routing key | none | Yes when backend is `pagerduty` |
| `INTERGRAX_PAGERDUTY_API_KEY` / `INTERGRAX_PAGERDUTY_TOKEN` **(secret)** | Aliases if routing key is unset | none | Fallback |
| `INTERGRAX_PAGERDUTY_URL` | Events API base URL | `https://events.pagerduty.com` | No |

---

## Inbound interactions

Owner: **platform**. Application hosts may also expose `*_INTERACTION_SURFACE`
under an application prefix; that is an **application override** of the same
idea. The platform env name is `INTERGRAX_INTERACTION_SURFACE`.

### INTERGRAX_INTERACTION_SURFACE

Purpose: Selects how inbound messages (lab JSON, Slack, Teams, slash commands)
are turned into tasks.

Owner: Platform.

Default: Platform profile default: `auto` (try Slack, Teams, slash command, then lab JSON).

Required: No. Unknown values fall back to `auto`.

Accepted values: `auto` · `lab` · `lab_json` · `slack` · `slash_command` · `teams`.

Example:

```text
INTERGRAX_INTERACTION_SURFACE=auto
```

### INTERGRAX_INBOUND_VERIFIER

Purpose: Verifies inbound HTTP signatures for Slack or Teams.

Owner: Platform.

Default: Platform profile default: `none`. Unknown values fall back to `none`.

Required: Conditional — use `slack` or `teams` for production webhooks.

Accepted values: `none` · `slack` · `teams`.

Example:

```text
INTERGRAX_INBOUND_VERIFIER=slack
```

Related settings: signing secrets below.

### INTERGRAX_SLACK_SIGNING_SECRET

Purpose: Slack signing secret for request verification. **(secret)**

Owner: Provider connection.

Default: none.

Required: Conditional — when `INTERGRAX_INBOUND_VERIFIER=slack` and verification is enabled.

### INTERGRAX_SLACK_VERIFY_SIGNATURE

Purpose: Enables Slack signature checks (opt-in; disabled by default so lab
intake works without a secret).

Owner: Platform.

Default: Platform profile default: off.

Required: No.

Accepted values: truthy / falsy env strings. Example: `true` to enable.

### INTERGRAX_TEAMS_SECURITY_TOKEN

Purpose: Teams security token for inbound verification. **(secret)**

Owner: Provider connection.

Default: none.

Required: Conditional — when verifier is `teams` and verification is enabled.

### INTERGRAX_TEAMS_VERIFY_SIGNATURE

Purpose: Enables Teams signature checks (opt-in; disabled by default).

Owner: Platform.

Default: Platform profile default: off.

Required: No.

---

## Workspace and sandbox

Owner: **runtime infrastructure**.

### INTERGRAX_SHADOW_ROOT

Purpose: Root directory for per-tenant/task shadow workspaces.

Owner: Runtime infrastructure.

Default: Platform profile default: `build/shadow_workspaces`.

Required: No.

Example:

```text
INTERGRAX_SHADOW_ROOT=/tmp/intergrax_shadow
```

### INTERGRAX_SANDBOX_ROOT

Purpose: Root directory for sandbox sessions.

Owner: Runtime infrastructure.

Default: Platform profile default: `build/sandbox_sessions`.

Required: No.

Example:

```text
INTERGRAX_SANDBOX_ROOT=/tmp/intergrax_sandbox
```

---

## Host environment, auth, scheduler, workers

### INTERGRAX_ENV

Purpose: Deployment environment name used when an application-prefixed
`BACKEND_ENV` is unset. `staging` is accepted and mapped to `stage`.

Owner: Platform (shared fallback for application hosts).

Default: Platform profile default: `dev`.

Required: No.

Accepted values: `dev` · `stage` · `prod` (plus `staging` → `stage`).

Example:

```text
INTERGRAX_ENV=dev
```

Notes: Application hosts typically prefer `<APP>_BACKEND_ENV`. Prefix alone is
not an absolute law — this platform key is the fallback.

### INTERGRAX_HARNESS_API_KEY

Purpose: Optional API key protecting harness HTTP/MCP surfaces. **(secret)**

Owner: Platform.

Default: none (auth disabled when unset). Stage/prod lab hosts require it.

Required: Conditional.

Example:

```text
INTERGRAX_HARNESS_API_KEY=replace-me
```

### INTERGRAX_SCHEDULER_POLL_SECONDS

Purpose: Poll interval for the long-running / HITL resume scheduler.

Owner: Runtime infrastructure.

Default: Platform profile default: none (host may pass `None`). Provider/runtime
adapter fallback when constructing the scheduler: `30` seconds.

Required: No.

Example:

```text
INTERGRAX_SCHEDULER_POLL_SECONDS=30
```

### INTERGRAX_CELERY_BROKER_URL

Purpose: Broker URL for platform worker / Celery-backed execution.

Owner: Runtime infrastructure.

Default: none. Some workers also accept ecosystem `CELERY_BROKER_URL` or
modality-specific `INTERGRAX_MODALITY_CELERY_BROKER_URL`.

Required: Conditional — when a worker queue is wired.

Example:

```text
INTERGRAX_CELERY_BROKER_URL=redis://localhost:6379/0
```

Notes: `INTERGRAX_USE_WORKER_QUEUE` is **not** an active runtime contract (not
read by platform code). Use host composition to enable a queued execution adapter.

---

## Observability export

Owner: **platform** where the flag is Intergrax-owned. Vendor backends are
selected with `INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND`.

### INTERGRAX_EXPORT_JOURNAL

Purpose: Export a unified run-journal OTLP snapshot when a task completes.

Owner: Platform.

Default: Platform profile default: **enabled** (`1`). Disabled when the value
is `0`, `false`, `no`, or `off`.

Required: No.

Example:

```text
INTERGRAX_EXPORT_JOURNAL=1
```

### INTERGRAX_EXPORT_PARSER_TRACE

Purpose: Export document-parser pipeline traces to the configured observability
backend (for example Langfuse).

Owner: Platform.

Default: Platform profile default: off (empty / not in the enabled set).

Required: No.

Accepted values: enable with `1` / `true` / `yes` / `on`.

Example:

```text
INTERGRAX_EXPORT_PARSER_TRACE=1
```

### Langfuse connection

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `LANGFUSE_PUBLIC_KEY` **(secret)** | Langfuse public key | none | Yes when exporting to Langfuse |
| `LANGFUSE_SECRET_KEY` **(secret)** | Langfuse secret key | none | Yes when exporting to Langfuse |
| `INTERGRAX_LANGFUSE_BASE_URL` | Langfuse API origin | `https://cloud.langfuse.com` | No |

Related: `INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND=langfuse`.

Lab OTLP presets and Compose ports: [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md).

---

## What is not in this document?

This is the **platform** catalog. It does not document product or application
hosts.

`INTERGRAX_*` usually means shared platform configuration. Application prefixes
such as:

- `LOCAL_WORKSPACE_*`
- `LEGAL_*`
- `RESEARCH_*`
- `LAB_*`

belong to individual applications and will be documented separately (CONFIG-3).

Prefix is a **convention**, not an absolute architectural law. Exceptions already
exist, for example:

- `INTERGRAX_ENV` is a platform fallback used by application hosts.
- `INTERGRAX_SCHEDULER_POLL_SECONDS` is platform-owned even when a lab host also
  has `LAB_INCLUDE_SCHEDULER`.
- `OLLAMA_HOST`, `OPENAI_API_KEY`, and similar vendor names are provider
  connection settings without an `INTERGRAX_` prefix.
- Some application hosts still read `INTERGRAX_LLM_*` because they consume the
  platform profile rather than inventing a second selection pair.

Also out of scope here:

- Proof-only `LKW_*` / `INTERGRAX_LKW_*` variables
- GitHub development tooling (`GH_TOKEN`)
- Application backend host/port/route/CORS/auth maps
- Application-specific Slack companion configuration

---

## Relationship to `.env.example`

| Artifact | Role |
|----------|------|
| **This document** | Canonical explanation of supported **platform** options |
| Root [`.env.example`](../../../../.env.example) | Copyable practical template of common platform examples |

`.env.example` is not automatically authoritative. Where the template and
runtime disagree, **runtime wins**. Application, proof, and development-tooling
keys belong in their own templates and docs, not in the root platform example.

---

## See also

- [LLM adapters](../../architecture/LLM_ADAPTERS.md) — provider catalog and model-selection architecture
- [Harness environment](HARNESS_ENVIRONMENT.md) — lab stack, OTLP, integration presets
- [Documentation map](../DOCUMENTATION_MAP.md) — where to read next
