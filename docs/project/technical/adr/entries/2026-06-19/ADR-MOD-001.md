# ADR-MOD-001: Speech provider slug identity via Integration Library (no enum)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-19 |
| **Deciders** | Harness platform (operator-approved idea audit) |
| **Related** | [`architecture/MODALITY.md`](../../architecture/MODALITY.md) · [`plan/MODALITY.md`](../../plan/MODALITY.md) MOD-SPEECH-ARCH.* · [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) |

## Context

Plane C speech (TTS/STT) was delivered in two parallel paths:

1. **Integration Library** — `speech_provider` category, `SpeechProviderBackend` Protocol, manifest + factory registration (`elevenlabs`, `deepgram`), Tier-3 `IntegrationProfile` resolution, `wire_integration_tool_context()` → `speech.*` tools.
2. **`speech_adapters`** — closed `SpeechProvider` enum (`stub`, `elevenlabs` only), `SpeechAdapterRegistry` builtins, `SpeechProfile` enum coercion, `speech_provider_for_slug()` hardcoded mapping (`deepgram` → `STUB`).

Integration canon states catalog identity is a string **slug** — not a central enum — and third-party providers **must not** extend a core enum. The enum path violates **SYS-INV-10** (one canonical path per universal concern) and blocks extension without editing platform source.

Operator constraint: **remove legacy immediately** — no deprecation phases or dual-path compatibility shims.

## Decision

1. **Single source of truth** for speech vendors is the Integration Library `speech_provider` category and `SpeechProviderBackend` Protocol.
2. **Delete** `SpeechProvider` enum and all enum-coercion in `SpeechProfile`, `SpeechAdapter`, `speech_provider_for_slug()`, and enum-typed `validate()` checks — in the same implementation wave as slug migration (no transitional aliases).
3. **`SpeechProfile`** (when used for env/lab defaults) holds `provider_slug: str` resolved against the integration catalog, or accepts a pre-built `SpeechProviderBackend` / `IntegrationBinding` instance — same binding model as `IntegrationProfile`.
4. **`IntegrationSpeechAdapter`** remains the bridge from `SpeechProviderBackend` to tool I/O contracts; it records `provider_slug: str` from the manifest, not an enum member.
5. **`wire_modality_extras()`** MUST NOT build a parallel enum-based backend when `IntegrationProfile` already resolved `speech_provider`; integration wiring wins.
6. **External extension** — new vendors register via `IntegrationManifest` + factory or `IntegrationPlugin`; optional `SpeechAdapterRegistry.register(slug, factory)` for in-process adapters only when not using the integration catalog path.

**Rejected:**

- Keeping `SpeechProvider` enum as deprecated alias — violates operator no-transitional-phase policy.
- Adding enum members for each new vendor (`deepgram`, `azure_speech`, …) — contradicts open catalog policy.
- Third parallel registry outside Integration Library for SaaS speech vendors.

## Consequences

### Positive

- Speech vendors extend identically to other integration categories (185+ slug pattern).
- `deepgram` and future slugs work through `SpeechProfile` and Tier-3 wiring without platform enum edits.
- One tool path: `IntegrationProfile` → `SpeechProviderBackend` → `speech.synthesize` / `speech.transcribe`.

### Negative

- Breaking change for code importing `SpeechProvider` enum — must migrate to slug strings or `IntegrationProfile` in the same PR series.
- `speech_adapters` shrinks to bridge + optional local adapters; documentation and tests must be updated atomically.

## Compliance

- Tier boundaries preserved — vendors stay Tier-0; agents use `ToolRuntime` only.
- **SYS-INV-10** restored — integration catalog is the canonical vendor path for speech SaaS.
- **SYS-INV-17** unchanged — no vendor SDK in Tier-2 agents.
- Architecture + plan domain pairs updated; implementation tracked under MOD-SPEECH-ARCH.*.

## Implementation notes

- Remove: `intergrax/speech_adapters/contracts/speech_provider.py` enum (replace with slug constant module or delete file).
- Refactor: `speech_integration_bridge.py`, `registry/profile.py`, `registry/speech_adapter_registry.py`, `applications/_shared/modality_wiring.py`, `applications/_shared/integration_tool_wiring.py`.
- Tests: `tests/unit/speech_adapters`, `tests/unit/applications/test_p6_integration_tool_wiring.py`, speech tool provider tests.
- Verify: `uv run pytest tests/unit/speech_adapters/ tests/unit/applications/test_p6_integration_tool_wiring.py tests/unit/tools/providers/test_modality_tools.py -q` · `python scripts/maintenance/check_harness_adr.py`.
