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
