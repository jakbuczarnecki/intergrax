# ADR-PDC-03 — Declared model streaming/cancel capabilities with fail-closed mismatch

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-03 |
| **Status** | **Accepted** (OD-PDC-ADR-01, 2026-11-12) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D3 Model Runtime |
| **Requirements** | PDC-D3-R006, PDC-D3-R022, PDC-D3-R015 (and PDC-D3-R011) — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-CTR-01, FRZ-REL-04, FRZ-REL-06, FRZ-CFG-02, FRZ-REG-09 |

## Context

- `LLM_ADAPTERS.md` conformance row #5 (streaming parity) is Partial; Ollama adapters do not implement `stream_with_tools` (base raises `NotImplementedError`).
- `supports_streaming()` defaults to False and six adapters that implement `stream_messages` still report False; no `supports_tool_streaming` / cancel capability flag exists.
- Cancellation is a per-provider `ProviderExternalOperationSeam` beside the ABI; Ollama only closes the transport.

## Decision (proposed)

- Capabilities are declared, not forced: extend the existing `supports_*` flag family with `supports_tool_streaming` and expose the cancellation seam outcome (`terminated` / `transport_closed` / `not_supported`) as a typed capability read through `LLMAdapterRegistry`.
- A conformance gate asserts that each flag matches the implemented methods (no adapter streams while reporting False).
- The runtime checks requested features against declared capabilities before the call and fails closed with a typed error; it never silently downgrades from streamed tool calls to buffered output.
- The ModelCatalog stays the single source for context-window data; provider-local tables become fallbacks resolved through the catalog (PDC-D3-R023 ownership via P3-01).

- **OD-PDC-SCOPE-03 = A (authorized exclusion):** the LLM provider plugin registration layer (LLM-PROVIDER-PLUGIN-1) is outside the freeze scope; replaceability stays on `LLMAdapterRegistry` registrations (EBH-5).

## Alternatives rejected

- Forcing full streaming parity in every provider before freeze (not required for provider-neutral architecture; Path B).
- A second capability registry outside `LLMAdapterRegistry` / `ModelCatalog`.

## Verification

`tests/qualification/pdc_x/d3/`: flag/method conformance, unsupported tool-streaming fails closed, typed cancel outcome per provider (fake providers), capability mismatch denied before the call.

## Status note

Accepted by OD-PDC-ADR-01 (Operator, 2026-11-12, recorded under T-IGX-036-E). This authorizes PDC-P4 design. Product merges still follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8), which is pending.
