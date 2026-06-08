**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix H — Architecture coverage matrix (Intergrax canon + ideal harness)

**Purpose:** ensure the implementation plan explicitly covers all harness-scope requirements from:

- `intergrax_runtime_architecture.md` (canonical Intergrax runtime architecture)
- `IDEAL_HARNESS_AI_ARCHITECTURE.md` (target/benchmark architecture)

**Rule:** For harness work, this matrix must have **zero `Uncovered` rows**.

### H.1 Coverage status legend

- **Done** — capability implemented and verified by existing phases/tests.
- **Partial closeout** — contracts/governance Done; runtime enforcement gaps scheduled in Phase V-REM.
- **Planned (Phase V-REM)** — explicitly scheduled in Phase V-REM (`V-REM-*` IDs).
- **Deferred (product scope)** — intentionally outside harness-only scope (Band 3 / §6.3).
- **Uncovered** — gap; MUST be added to plan before related implementation proceeds.

### H.2 Harness architecture domains — required coverage

| Domain (harness scope) | Intergrax canon anchor | Ideal harness anchor | Plan coverage | Status |
|------------------------|------------------------|----------------------|---------------|--------|
| Strategic objective + harness-first hierarchy | canon §2, §5.1, §51, §53.1 | ideal §0, §1, §26 | §0, §4.0, Phase V governance | **Done** |
| Tier model and runtime boundaries | canon §5.1, §7.0–§7.4, §42 | ideal §3, §26 | §0.2, §2 map, Phases L/Q+/U, **FAUDIT-TIER.\*** | **Done** — reference manifest catalog in `intergrax/applications/reference/` + CI gate |
| Unified execution runtime (UAEP, lifecycle, interrupts, policy) | canon §42.* | ideal §3.3, §3.4, §5, §8 | §2 map, Phase U, gate suites | **Done** |
| Context engineering core | canon §28.1, §42.35 | ideal §16 | Phase R (Done) + V-CE.* | **Done** |
| Capability graph dependencies + impact analysis | canon §53.2 | ideal §19 + capability graph expectations | V-CG.* | **Done** |
| Agent lifecycle governance (cert/promo/deprec/retire/owner) | canon §15, §53.3 | ideal §17 | V-ALG.* | **Done** |
| Prompt engineering architecture | canon §53.5 | ideal §20 | V-PE.* | **Done** |
| Evaluation and benchmarking operations | canon §53.6 | ideal §18 | V-EVAL.* + A.4 | **Done** |
| Architecture metrics and debt governance | canon §53.7 | ideal §21 + architecture metrics expectations | V-AM.* | **Done** |
| Security/data governance (agent-native threats) | canon §42.37, §53.8 | ideal §23 | Phase U (baseline) + V-SEC.* | **Done** |
| Cost/resource governance | canon §53.9 | ideal §24 | V-COST.* | **Done** |
| Multi-agent coordination pattern catalog | canon §42.43, §53.10 | ideal §6 + §25 | V-MA.* | **Done** |
| Knowledge graph evolution path (Graph-RAG) | canon §53.11 | ideal §3.7.1 + §25 | V-KG.* | **Done** |
| **Adaptive Harness Intelligence (L4 runtime closed loop)** | canon §54 | ideal §25 | **Phase W-ADAPT** · AHIA | **Done** (Band 2y, 70/70) — L4 runtime closed; observe/recommend/apply/verify per AHIA |
| Observability and runtime traceability | canon §33, §42.24 · [`OBSERVABILITY_ARCHITECTURE.md`](OBSERVABILITY_ARCHITECTURE.md) | ideal §11 | Phases OBS + OBS-DEPTH.* + **Phase OBS-BUS** | **L4 Done** — spine, typed payloads, emitter, emission coverage, journal export; gate: `check_observability_gates.py` |
| Registry-driven extensibility (agent/tool/skill/policy/prompt/eval) | canon §7.1.5.1–§7.1.8, §15, §53.2 | ideal §19 | Phase R/U + V-CG/V-PE/V-EVAL + **P-Ext** | **Done** — plugin catalogs production-ready; marketplace UI out of scope |
| Product agents and new product apps | canon §7.4, §52 | ideal §26 | §6.3 only | **Deferred (product scope)** |

### H.3 Completion policy for “architecture-complete harness”

Harness architecture can be considered complete against both architecture documents only when:

1. All harness-scope rows in H.2 are `Done` (no `Partial closeout`, no `Planned`, no `Uncovered`).
2. `Deferred (product scope)` rows remain intentionally isolated to Band 3 (§6.3).
3. Phase V-REM complete and parent V-* Partial rows closed.
4. Phase V KPI thresholds and L3/L4 evidence gates are satisfied.
5. Canon + plan + docs index are synchronized in the same change window.

### H.4 Change control rule

Any future addition to either architecture document that introduces a new harness-scope
domain MUST be reflected in:

- this matrix (Appendix H),
- a concrete Phase V-REM (or successor phase) deliverable ID,
- priority ladder (§4) and “what next” (§6) if it changes execution order.

---
