# ADR-GR-001: LLM guardrail integration plane (M-P12)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-09 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) §47 · [`plan/INTEGRATIONS.md`](../../plan/INTEGRATIONS.md) M-P12 |

## Context

Tier-3 hosts need vendor-agnostic LLM input/output scanning without duplicating UAEP policy or placing vendor SDKs in Tier-2 agents. Multiple guardrail backends (LLM Guard, Guardrails AI, Presidio, cloud APIs) must share one catalog slot and compose with existing security middleware.

## Decision

- Introduce `IntegrationCategory.LLM_GUARDRAIL` and `LlmGuardrailBackend` with `scan_input` / `scan_output` returning `GuardrailScanResult`.
- Add `IntegrationProfile.llm_guardrail` binding resolved like other category slots.
- Wire guardrail slug into `RuntimeConfig.metadata` via `security_runtime_bridge` and Tier-3 `guardrail_runtime_bridge` — **no** Nexus fork.
- Ship catalog adapters for all slugs via `register_all.py` + `_factory.py` — vendor SDK in `_vendor_opens.py` with pattern fallback; HTTP adapters for cloud/gateway slugs.
- Preset `harness_guardrail_stack(primary, semantic)` for lab/legal strict profiles.

Rejected: per-agent guardrail SDK imports; duplicate scan paths outside security bridge.

## Consequences

### Positive

- Single extension point for guardrail vendors aligned with Integration Library.
- Tier-3 can enable scanning via profile only.

### Negative

- Vendor SDKs (`llm-guard`, `guardrails-ai`, `nemoguardrails`) are **manual install** — conflict with pinned docling/torch; harness uses pattern fallback in CI.
- NeMo/Llama/Bedrock full vendor depth remains pattern/HTTP until dedicated bundles ship.

## Compliance

- Tier boundaries preserved — vendors only under `integrations/providers/llm_guardrail/`.
- Linked architecture §47 and plan register M-P12 updated.

## Implementation notes

- `intergrax/integrations/contracts/llm_guardrail.py`
- `intergrax/integrations/providers/llm_guardrail/_factory.py`, `_adapters.py`, `_vendor_opens.py`
- `intergrax/applications/_shared/application_guardrail_middleware.py`, `guardrail_wiring.py`, `guardrail_assembly_resolver.py`
- `scripts/maintenance/check_harness_guardrail_wiring.py`
- `tests/unit/integrations/test_llm_guardrail_contract.py`, `tests/unit/applications/test_harness_guardrail_wiring.py`
