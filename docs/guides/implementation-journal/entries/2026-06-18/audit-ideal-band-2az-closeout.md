# AUDIT-IDEAL Band 2az closeout — 88/88 Done

**Date:** 2026-06-18  
**Scope:** Close remaining 7 AUDIT-IDEAL rows (6.2–6.4, 6.6–6.7, 14.4–14.5) after §6.1av MAINT complete.  
**Domain pairs:** `LLM_ADAPTERS`, `RAG`, `PLATFORM_FOUNDATION`, `CONTEXT_ENGINEERING`, `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`

## Completed items

| ID | Implementation | Tests / gates |
|----|----------------|---------------|
| AUDIT-IDEAL-6.3 | `CatalogCapabilityAdapter` enriches adapters from YAML catalog | `test_catalog_capabilities.py` |
| AUDIT-IDEAL-6.4 | `count_message_tokens(..., adapter=)` delegates tokenizer to adapter | `test_count_message_tokens.py` |
| AUDIT-IDEAL-6.2 | Live model routing on product hosts (AHI prod path) | `check_live_model_routing_wiring.py`, depth gate |
| AUDIT-IDEAL-6.6 | `StepLLMRouter` async bridge over `LLMAdapter.generate_messages` | depth gate, `test_llm_router.py` |
| AUDIT-IDEAL-6.7 | `LLMProfile.validate_runtime()` wired into doctor + CI | `check_llm_profile_runtime.py` |
| AUDIT-IDEAL-14.4 | Hierarchical dual-index bootstrap gate | `check_rag_hierarchical_bootstrap.py` |
| AUDIT-IDEAL-14.5 | Poisoning filter on `perform_rag_retrieve` catalog path | `check_rag_catalog_poisoning_defense.py` |

## Changed files

- `intergrax/llm_adapters/registry/catalog_capabilities.py`
- `intergrax/llm_adapters/llm_provider_registry.py`
- `intergrax/runtime/nexus/context/context_preflight.py`
- `scripts/check_llm_profile_runtime.py`
- `scripts/check_rag_hierarchical_bootstrap.py`
- `scripts/check_rag_catalog_poisoning_defense.py`
- `scripts/check_audit_ideal_gates.py`
- `intergrax/cli/doctor.py`
- `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py`
- `docs/plan/AUDIT_IDEAL_2026.md`, `LLM_ADAPTERS.md`, `PLATFORM_FOUNDATION.md`
- `docs/guides/audit/results/2026-06-18/RUN_SUMMARY.md`

## Verification

```bash
uv run pytest -m "gate and not no_ci" -q
python scripts/check_audit_ideal_gates.py
python scripts/check_plan_scorecard_sync.py
python scripts/check_implementation_journal.py
```

## Architectural impact

No new Tier-0 mechanisms — composition and gate wiring only. Tier boundaries preserved.

## ADR

**No ADR needed** — gate tests and doctor hooks; no contract or semantics change beyond documented closeout.

## Risks

- `check_model_catalog_coverage.py` remains a separate M-LLM-X.7.3 maintenance item (not blocking AUDIT-IDEAL).

## Suggested next step

Return to **§6.1 gate maintenance** per `PLATFORM_FOUNDATION.md` default queue.
