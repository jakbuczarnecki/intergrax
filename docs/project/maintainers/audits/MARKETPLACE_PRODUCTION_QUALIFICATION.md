# ME-17 — Marketplace Production Qualification

**Branch:** `development`  
**Qualification commit:** (see git log after merge)  
**Verdict:** `MARKETPLACE PRODUCTION QUALIFIED` (pending independent GitHub audit)

## Architecture inventory (summary)

| Area | Contract surface | Default impl | Pluginable | Isolation | Failure policy | Concurrency | Qualified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Catalog federation | `CapabilityCatalogSource` | `FederatedCapabilityCatalog` | Yes | Per-source | STRICT / ALLOW_PARTIAL | Read-safe | Yes |
| Visibility | `MarketplaceVisibility` + evaluator | `MarketplaceVisibilityEvaluator` | Yes | Tenant/org | Fail closed | Read-safe | Yes |
| Search / rank / govern / recommend | Catalog + marketplace pipelines | Default strategies | Yes | Query context | Typed / propagate | Deterministic | Yes |
| Handoff | `CapabilityHandoffEnvelope` | Delivery + orchestrator | Adapters | Tenant + admission | Typed consumer errors | Idempotent admission | Yes |
| Observability | `MarketplaceDiagnosticObserver` | In-memory / noop | Yes | Correlation scoped | BEST_EFFORT / STRICT | Thread-safe sink | Yes |
| Cache | `CapabilityCatalogSnapshotCache` | Bounded in-memory | Yes | Keyed federation | Fallback / propagate | Bounded cache tests | Yes |
| Machine API | `MachineCapabilityAcquisition*` | `MachineCapabilityAcquisitionService` | Policy plugins | Single snapshot | Fail closed | ME-12 proofs | Yes |
| Agent / Tool / Skill verticals | Lifecycle handoff contracts | Domain bridges | Providers | Host profile | Domain typed | Concurrent acquire tests | Yes |
| Mixed A+T+S | ME-16 composition | Proof stack | Custom providers | Org/tenant | Readiness gates | ME-16-C1 | Yes |

Evidence: `tests/integration/marketplace/test_me17_marketplace_production_qualification.py` + existing ME-RB/ME-5–ME-16 suites (`422 passed` marketplace bundle).

## Performance baseline (local, synthetic, no network)

| Catalog size | snapshot (s) | discovery (s) | rank/governance (s) |
| --- | ---: | ---: | ---: |
| 100 | 0.00029 | 0.00281 | 0.00070 |
| 1,000 | 0.00294 | 0.21235 | 0.03161 |

No SLA claimed — numbers only.

## Known regression — `UserProfileLtmVectorProjection` / `workspace_id`

**Root cause:** `build_user_profile_manager` constructed `UserProfileLtmVectorProjection` without required `workspace_id` after constructor hardening.  
**Fix:** `workspace_id=None` in `intergrax/applications/_shared/memory_vector_wiring.py` (aligned with `UserProfileManager` defaults).  
**Result:** ME-14 C1 tool acquisition / execution host tests green.

## Production hardening in ME-17

- `MarketplaceCatalogService`: eager metadata validation (`duplicate source_id`, duplicate listing identity) at construction; canonical join validation on first federated read (unchanged semantics, ME-12 single-snapshot preserved).

## Remaining gaps

| Priority | Item |
| --- | --- |
| P2 | Canonical Tool trust authority (ME-14 E2E skip when trust fixture unavailable) |
| P3 | Optional 10k catalog perf sample |

## Regression bundle

```text
uv run pytest tests/unit/marketplace/ tests/integration/marketplace/ tests/unit/contracts/marketplace/
→ 422 passed, 1 skipped
```
