---
id: IJ-2026-06-19-004
date: 2026-06-19
tiers:
  - tier-0
  - tier-3
scope: LLM_ADAPTERS
plan_ref:
  - M-LLM-X.8.1
  - M-LLM-X.8.2
  - M-LLM-X.8.3
  - M-LLM-X.14.1
  - M-LLM-X.14.2
  - M-LLM-X.14.3
  - M-LLM-X.14.4
  - M-LLM-X.14.5
  - M-LLM-X.14.6
  - M-LLM-X.14.7
  - M-LLM-X.14.8
  - LLM-AUDIT-21
  - LLM-AUDIT-22
  - LLM-AUDIT-23
  - LLM-AUDIT-24
  - LLM-AUDIT-25
  - LLM-AUDIT-26
status: completed
commit: pending
adr: none — extends ADR-LLM-002 gateway merge and registry slug validation; no new ADR
---

# M-LLM-X.8 + X-14 — LLM enterprise domain closeout

## Operator request

Implement enterprise domain maturity backlog after routing L5 (X-13): gateway metadata, ACP budget bridge, plugin provider DX, secondary LLM evaluating wrap, domain audit register closeout, and journal.

## Summary

Delivered **M-LLM-X.14** (8 items) and **M-LLM-X.8** closeout: `OpenRouterModelMetadataClient` + session merge in `resolve_context_window_tokens`; `ModelCatalogMissDiagV1`; enum-free `LLMProfile.provider`; ACP `tokens_used_from_usage` fix; opt-in `llm_routing_evaluating_secondary` wrap for tool planner / websearch / critic; multi-step routing soak; `TokenizerPlugin` Protocol stub; scaffold USAGE comment. Closes **LLM-AUDIT-21…26**. Whole **LLM_ADAPTERS** domain re-scored **L4 enterprise** (routing **L5** maintained).

## Project impact

Tier-3 hosts can opt into live gateway context windows, custom provider slugs, secondary-surface mid-run routing, and honest catalog miss diagnostics without bypassing bundled `ModelCatalog`. Domain audit register is fully **Done** for LLM-AUDIT-1…26.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/LLM_ADAPTERS.md` § Enterprise domain maturity register |
| Plan | `docs/plan/LLM_ADAPTERS.md` Wave M-LLM-X-8 · M-LLM-X-14 |
| ADR | ADR-LLM-002 (gateway merge); no new ADR |
| Audit | LLM-AUDIT-21…26 **Done** |

## Changed artifacts

- `intergrax/llm_adapters/registry/gateway_metadata/` — OpenRouter client + session merge
- `intergrax/llm_adapters/registry/catalog_miss_diag.py` — `ModelCatalogMissDiagV1`
- `intergrax/llm_adapters/registry/context_window.py`, `profile.py`, `secrets.py`
- `intergrax/llm_adapters/contracts/tokenizer_plugin.py`
- `intergrax/llm_adapters/routing/context_bridge.py` — ACP usage token mapping
- `intergrax/applications/_shared/llm_routing_runtime_bridge.py` — secondary evaluating wrap
- `intergrax/applications/contracts/environment_profile/bundles.py` — `llm_routing_evaluating_secondary`
- Tests: `test_gateway_metadata.py`, `test_custom_provider_and_acp_usage.py`, `test_acp_routing_context_provider.py`, `test_secondary_evaluating_wrap.py`, `test_multi_step_routing_soak.py`
- Docs: `docs/architecture/LLM_ADAPTERS.md`, `docs/plan/LLM_ADAPTERS.md`, `USAGE.md`, hub, journal

## Verification

```bash
uv run pytest tests/unit/llm_adapters/ tests/acceptance/llm_routing/ -m "gate and not no_ci" -q
python scripts/check_llm_routing_tier_boundary.py
python scripts/check_llm_routing_context_wiring.py
python scripts/check_docs_domain_pairs.py
```

Result: 158 passed; LLM routing gates OK.

## Risks and follow-ups

- Vendor-native tokenizer plugins beyond Protocol stub (product opt-in).
- Live OpenRouter HTTP fetch in production hosts (`fetch_gateway_metadata=True`).
- `check_implementation_journal.py` reports pre-existing legacy entry gaps unrelated to this episode.
