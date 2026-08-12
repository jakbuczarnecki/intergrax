# Platform Plugin Developer Documentation Audit

**Task:** PLATFORM-PLUGIN-DOCS-1  
**Status:** READY_FOR_REVIEW  
**Date:** 2026-08-12  
**Branch:** `development`  
**Required ancestors verified:** `f7b6eedf` (PLATFORM-PLUGIN-9 closeout) · `00144b4d` (PLATFORM-PLUGIN-AUDIT-1)

---

## 1. Executive assessment

### Question

> Does an external or application developer have enough accurate documentation to understand, implement, register, configure, qualify, test and run every intentionally public Intergrax plugin surface without reading internal source code?

**Answer: No.**

After PLATFORM-PLUGIN-1..9 closeout and PLATFORM-PLUGIN-AUDIT-1, the documentation stack is **architecturally strong** at Level 1 and **uneven** at Levels 2–3. A developer can implement **Tools** end-to-end from documentation alone (external wheel + host-embedded). **Integrations, Skills, RAG (chunker/retriever/reranker), and Vendor Knowledge** are **partially** implementable with domain guides plus cross-platform material. **Context, Memory stores, Policy rules, and Tool invocation patterns** lack sufficient standalone developer paths. **Security defenses** are partially documented but omit production-relevant conflict semantics.

| Layer | Complete? | Summary |
|-------|-----------|---------|
| Level 1 — Platform architecture | **Mostly yes** | [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) is current, frozen, and cross-references audit limitations honestly |
| Level 2 — Cross-platform developer guide | **No** | [`EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md) is strong for Tools/Integrations/Skills and cross-cutting PLUGIN-3..8 topics but is **not** a complete entry point (no decision tree; 4 surfaces without dedicated sections) |
| Level 3 — Domain-specific developer docs | **No** | 1 of 12 surfaces fully documented; 7 partial; 4 missing as standalone developer journeys |

### Surface summary (12 audited)

| Verdict | Count | Surfaces |
|---------|-------|----------|
| **COMPLETE** | 1 | Tools |
| **PARTIAL** | 7 | Integrations, Skills, RAG chunkers, RAG retrievers, RAG rerankers, Vendor Knowledge, Security defenses |
| **MISSING** | 4 | Context, Memory stores, Policy rules, Tool invocation patterns |

### Findings by severity

| Severity | Count | IDs |
|----------|-------|-----|
| HIGH | 5 | PLUGIN-DOC-F001..F005 |
| MEDIUM | 8 | PLUGIN-DOC-F006..F013 |
| LOW | 4 | PLUGIN-DOC-F014..F017 |
| INFO | 2 | PLUGIN-DOC-F018..F019 |

### Can implement all 12 surfaces from docs alone?

**No.** Tools yes; Integrations/Skills/RAG/VK with caveats (read multiple docs, some fixture-only examples); Context/Memory/Policy/Tool-invocation patterns require source or tests.

---

## 2. Level 1 — Platform architecture assessment

**Canonical document:** [`docs/project/architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

### Coverage vs required topics

| Topic | Status | Evidence |
|-------|--------|----------|
| Terminology | **Complete** | §4 |
| Architecture / responsibility split | **Complete** | §2, §7 |
| Taxonomy (PEP/IP/HCE/IEP) | **Complete** | §5 |
| Package/capability model | **Complete** | §8, §21 |
| Manifest | **Complete** | §11 |
| Discovery | **Complete** | §9 — includes default-off (F003) |
| Registration | **Complete** | §10 |
| External vs local delivery | **Complete** | §20.3 matrix — gaps acknowledged |
| Configuration / secrets / DI | **Complete** | §12–§13, §12.3 matrix |
| Lifecycle | **Partial** | §14 vocabulary only — F016 honestly stated |
| Compatibility | **Complete** | §15 — F001 limitation in §20.4 |
| Conflict handling | **Complete** | §17 — domain variance documented |
| Qualification | **Complete** | §18 — F002 semantic vs attestation |
| Trust/security | **Complete** | §16 — F017 trusted in-process |
| Failure model | **Complete** | §22 — TARGET vs CURRENT |
| Observability | **Partial** | §19 TARGET list; F006 no unified inventory |
| Public APIs | **Complete** | §20.1, §20.3 |
| DO-NOT-UNIFY | **Complete** | §23 |
| Limitations / production model | **Complete** | §27, §20.4, cross-ref to AUDIT-1 |

### Verdict

**Architecture documentation is complete and current** for platform coordination after PLUGIN-9/AUDIT-1. It correctly does **not** claim domain implementation sufficiency — §20.3 records per-surface author gaps. It is **not** a substitute for domain developer guides.

### Gaps at Level 1

- No single navigation index from architecture hub to all 12 domain author paths (routes to EXTENSION_AUTHOR_GUIDE + VK guide only).
- Maintainer roadmap [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) line 56 still says Context author guide marks "partial rollout" — stale vs PLUGIN-9 fix (INFO).

---

## 3. Level 2 — Cross-platform developer guide assessment

**Canonical document:** [`docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md)

### Required capabilities

| Capability | Status | Evidence |
|------------|--------|----------|
| Decision tree: which surface? | **MISSING** | No § or appendix; only summary table in header |
| External vs local embedded choice | **Partial** | §16 dual-mode Tools only; §15.2 mentions pattern generically |
| Generic package structure | **Partial** | §13 manifest + §5 EP groups; §16.1 Tools package |
| Manifest | **Complete** | §13 |
| EP declaration | **Complete** | §5 (integrations/tools/skills/RAG); VK/RAG have own guides |
| Discovery enablement | **Complete** | §1, §81 — `INTERGRAX_DISCOVER_PLUGINS` |
| Qualification | **Complete** | §15 |
| Config/DI | **Complete** | §14 |
| Testing | **Partial** | §7 validation commands; domain-specific patterns thin |
| Production checklist | **Partial** | §15 + §16 steps; not per-surface |
| Debugging/troubleshooting | **Partial** | Scattered; RAG guide §17 is best example |
| Reference examples | **Partial** | Tools only has installable external + local examples |

### Sections present vs 12 surfaces

| Surface | Dedicated § in EXTENSION_AUTHOR_GUIDE |
|---------|--------------------------------------|
| Integrations | §2 |
| Tools | §3, §4b, §16 |
| Skills | §4 |
| Context | Table row only — **no §** |
| Memory stores | §9 (thin) |
| RAG | EP rows in §5; canon → RAG_EXTENSION_GUIDE |
| Vendor Knowledge | Not in guide — separate VK author guide |
| Security defenses | §12 (checklist) |
| Policy rules | §10 (thin) |
| Tool invocation patterns | **Absent** |

### Verdict

**Not complete as the mandated developer entry point.** Works well for **Integrations + Tools + Skills** bootstrap and cross-cutting Platform Plugin topics (manifest, qualification, dual-mode Tools). Fails the stated Level 2 goal for surface selection and for half the public surfaces.

---

## 4. Full 12-surface × D1–D16 matrix

**Legend:** COMPLETE · PARTIAL · MISSING · N/A

**Authoritative domain docs used:**

| # | Surface | Primary domain doc(s) | Secondary |
|---|---------|----------------------|-----------|
| 1 | Integrations | EXTENSION_AUTHOR_GUIDE §2 · INTEGRATIONS.md | PLATFORM_PLUGINS §20.3 |
| 2 | Tools | EXTENSION_AUTHOR_GUIDE §3,§16 · TOOLS.md | examples/platform_plugins/* |
| 3 | Skills | EXTENSION_AUTHOR_GUIDE §4 · SKILLS.md | skills/examples/custom_pack |
| 4 | Context | CONTEXT_ENGINEERING.md | context/plugin.py (contract) |
| 5 | Memory stores | EXTENSION_AUTHOR_GUIDE §9 | tests/fixtures/plugin_packages/memory_store_plugin |
| 6 | RAG chunkers | RAG_EXTENSION_GUIDE §4 | RAG.md |
| 7 | RAG retrievers | RAG_EXTENSION_GUIDE §5 | RAG.md |
| 8 | RAG rerankers | RAG_EXTENSION_GUIDE §6 | RAG.md |
| 9 | Vendor Knowledge | VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md | KNOWLEDGE_SOURCE_INTEGRATIONS.md |
| 10 | Security defenses | EXTENSION_AUTHOR_GUIDE §12 | UNIFIED_EXECUTION_RUNTIME §42.45 |
| 11 | Policy rules | EXTENSION_AUTHOR_GUIDE §10 | AGENT_CREATION_GUIDE Appendix H |
| 12 | Tool invocation patterns | TOOLS.md · ADR-TOOL-003 | tool_invocation_pattern.py |

### 1 — Integrations (`intergrax.integrations`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | PARTIAL | §2 + INTEGRATIONS.md (provider architecture, not third-party plugin framing) |
| D2 Public contract | COMPLETE | `IntegrationPlugin` · `intergrax.integrations.core.plugin` · §2 |
| D3 Minimal implementation | COMPLETE | `intergrax/integrations/examples/custom_memory_kv/` |
| D4 External package | COMPLETE | §5 EP + manifest pattern |
| D5 Local embedded | COMPLETE | `register_integration_plugin()` §2 |
| D6 Configuration | PARTIAL | `IntegrationProfile` shown; preset/bundle selection thin for externals |
| D7 Secrets | PARTIAL | `env_prefix` in PLATFORM_PLUGINS §12.3; not in §2 walkthrough |
| D8 DI | COMPLETE | `IntegrationProfile.resolve(category)` |
| D9 Registration/discovery | COMPLETE | `bootstrap_catalogs` §1 |
| D10 Qualification | COMPLETE | §15 |
| D11 Runtime invocation | COMPLETE | `profile.resolve` example §2 |
| D12 Lifecycle | PARTIAL | No cleanup contract documented |
| D13 Failure behavior | PARTIAL | `on_conflict` §5; not full domain failure matrix |
| D14 Testing | PARTIAL | §7 commands; `test_external_integration_entry_point.py` not linked |
| D15 Production checklist | PARTIAL | §15 generic; integration-specific concerns in INTEGRATIONS.md not consolidated |
| D16 Troubleshooting | PARTIAL | No dedicated discovery troubleshooting |

**Overall: PARTIAL**

### 2 — Tools (`intergrax.tools`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | COMPLETE | §3 + TOOLS.md |
| D2 Public contract | COMPLETE | `ToolPlugin` · `intergrax.tools.core.plugin` |
| D3 Minimal implementation | COMPLETE | `tools/examples/custom_echo` + reference wheel |
| D4 External package | COMPLETE | §16.1 · `examples/platform_plugins/intergrax_reference_tool_plugin/` |
| D5 Local embedded | COMPLETE | §16.2 · `local_embedded_tool_extension/` + scaffold |
| D6 Configuration | COMPLETE | `ToolProfile` §1 matrix |
| D7 Secrets | COMPLETE | §14 + ToolWiringContext pattern |
| D8 DI | COMPLETE | `ToolWiringContext` §3 |
| D9 Registration/discovery | COMPLETE | §1, §16 |
| D10 Qualification | COMPLETE | §15–§16 |
| D11 Runtime invocation | COMPLETE | `RuntimeToolInvoker` · E2E test cited |
| D12 Lifecycle | PARTIAL | No shutdown/cleanup contract |
| D13 Failure behavior | PARTIAL | Conflict policy §5; tool execution errors in audit matrix only |
| D14 Testing | COMPLETE | E2E + contract suite linked §16 |
| D15 Production checklist | COMPLETE | §16 steps 8–9 |
| D16 Troubleshooting | PARTIAL | Discovery env documented; no consolidated FAQ |

**Overall: COMPLETE** (reference surface)

### 3 — Skills (`intergrax.skills`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | COMPLETE | §4 + SKILLS.md |
| D2 Public contract | COMPLETE | `SkillPlugin` |
| D3 Minimal implementation | PARTIAL | `skills/examples/custom_pack` — in-repo, not installable package |
| D4 External package | COMPLETE | §5 EP |
| D5 Local embedded | COMPLETE | `register_skill_plugin()` §4 |
| D6 Configuration | PARTIAL | `SkillProfile` in §1 matrix; less detail than Tools |
| D7 Secrets | PARTIAL | §14 generic only |
| D8 DI | PARTIAL | Via tools/runtime; not walkthrough |
| D9 Registration/discovery | COMPLETE | `bootstrap_catalogs` |
| D10 Qualification | COMPLETE | §15 |
| D11 Runtime invocation | PARTIAL | `SkillResolver` mentioned SKILLS.md; not in guide |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | PARTIAL | Conflict §5 only |
| D14 Testing | PARTIAL | §7 generic |
| D15 Production checklist | PARTIAL | §15 generic |
| D16 Troubleshooting | MISSING | |

**Overall: PARTIAL**

### 4 — Context (`intergrax.context`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | PARTIAL | CONTEXT_ENGINEERING.md §CE plugin table only |
| D2 Public contract | PARTIAL | `ContextPlugin` in `context/plugin.py`; not documented in author guide |
| D3 Minimal implementation | MISSING | No example package or copyable walkthrough |
| D4 External package | PARTIAL | EP group in PLATFORM_PLUGINS §20; not in EXTENSION_AUTHOR_GUIDE §5 |
| D5 Local embedded | PARTIAL | `register_context_plugin()` named in table; no procedure |
| D6 Configuration | PARTIAL | `ContextProfile` referenced architecture only |
| D7 Secrets | MISSING | |
| D8 DI | PARTIAL | `ContextPluginRegistry` architecture only |
| D9 Registration/discovery | PARTIAL | `bootstrap_context_catalog()` in architecture |
| D10 Qualification | PARTIAL | "domain-owned rollout" note only |
| D11 Runtime invocation | MISSING | No assemble/CE path for custom plugin |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | MISSING | |
| D14 Testing | PARTIAL | `test_context_catalog_bootstrap.py` not linked for authors |
| D15 Production checklist | MISSING | |
| D16 Troubleshooting | MISSING | |

**Overall: MISSING** (developer journey)

### 5 — Memory stores (`intergrax.memory_stores`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | PARTIAL | §9 table only |
| D2 Public contract | PARTIAL | Duck-typed factories; no formal Protocol doc |
| D3 Minimal implementation | PARTIAL | Test fixture only — not author-facing |
| D4 External package | PARTIAL | EP group named; no pyproject example in guide |
| D5 Local embedded | MISSING | No documented helper; host factory kwargs only in §20.3 |
| D6 Configuration | MISSING | `MemoryProfile` not in §9 |
| D7 Secrets | MISSING | |
| D8 DI | PARTIAL | "host passes kwargs" architecture §12.3 |
| D9 Registration/discovery | PARTIAL | `bootstrap_memory_stores` documented but **misleading** (count-only — F010) |
| D10 Qualification | PARTIAL | §15 generic |
| D11 Runtime invocation | MISSING | How host selects factory at runtime |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | MISSING | |
| D14 Testing | PARTIAL | Fixture exists; not linked as author pattern |
| D15 Production checklist | MISSING | |
| D16 Troubleshooting | MISSING | |

**Overall: MISSING**

### 6 — RAG chunkers (`intergrax.rag.chunkers`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | COMPLETE | RAG_EXTENSION_GUIDE §1, §4 |
| D2 Public contract | COMPLETE | `BaseChunkingStrategy` |
| D3 Minimal implementation | COMPLETE | Inline code §4 + §15 skeleton |
| D4 External package | COMPLETE | §15 pyproject |
| D5 Local embedded | MISSING | Explicitly not documented (architecture §20.3) |
| D6 Configuration | COMPLETE | `RagProfile.chunking_strategy_id` |
| D7 Secrets | N/A | |
| D8 DI | COMPLETE | No-arg constructor; bootstrap semantics §1 |
| D9 Registration/discovery | COMPLETE | Opt-in discovery documented |
| D10 Qualification | COMPLETE | §16 checklist |
| D11 Runtime invocation | COMPLETE | Profile selection → bootstrap |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | COMPLETE | §1 conflicts + PluginLoadError |
| D14 Testing | COMPLETE | §16 test links |
| D15 Production checklist | COMPLETE | §16 |
| D16 Troubleshooting | COMPLETE | §17 |

**Overall: PARTIAL** (external-EP path strong; no local parity)

### 7 — RAG retrievers (`intergrax.rag.retrievers`)

| D | Score | Notes |
|---|-------|-------|
| D1–D16 | Same pattern as chunkers | §5 authoring · `BaseRetriever` / `BaseRetrieverPlugin` |
| D5 Local embedded | MISSING | |
| D3 | COMPLETE | §5 + §15 |

**Overall: PARTIAL**

### 8 — RAG rerankers (`intergrax.rag.rerankers`)

| D | Score | Notes |
|---|-------|-------|
| D1–D16 | Same pattern as chunkers | §6 authoring · `BaseReranker` / `BaseRerankerPlugin` |
| D5 Local embedded | MISSING | |
| D8 DI | COMPLETE | `create(embedding_manager=...)` documented |

**Overall: PARTIAL**

### 9 — Vendor Knowledge (`intergrax.vendor_knowledge.providers`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | COMPLETE | VK guide §1 |
| D2 Public contract | COMPLETE | §2 `VendorKnowledgeProviderContribution` |
| D3 Minimal implementation | PARTIAL | §20 minimal example; `acme_reference` is test artifact |
| D4 External package | COMPLETE | §5–§7 |
| D5 Local embedded | PARTIAL | Host builder composition §1 diagram — not Tier-0 catalog |
| D6 Configuration | COMPLETE | `KnowledgeSourceBinding` §12 |
| D7 Secrets | COMPLETE | §10 connection factory |
| D8 DI | COMPLETE | Contribution catalog model |
| D9 Registration/discovery | COMPLETE | §11 — separate from Tier-0 discovery (F018) |
| D10 Qualification | COMPLETE | §21–§22 |
| D11 Runtime invocation | COMPLETE | Search/Ask §15 |
| D12 Lifecycle | PARTIAL | §17 restart/rehydration |
| D13 Failure behavior | COMPLETE | §18 |
| D14 Testing | COMPLETE | §21 reference tests |
| D15 Production checklist | COMPLETE | §19, §25 |
| D16 Troubleshooting | PARTIAL | Error §18; no discovery FAQ |

**Overall: PARTIAL** (strongest domain guide after Tools/RAG; reference package not installable)

### 10 — Security defenses (`intergrax.security_defenses`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | PARTIAL | §12 checklist |
| D2 Public contract | PARTIAL | `SecurityDefensePlugin` — checklist not full protocol |
| D3 Minimal implementation | PARTIAL | CI fixture only |
| D4 External package | PARTIAL | EP group named; no pyproject walkthrough |
| D5 Local embedded | MISSING | `bootstrap_security_providers` only |
| D6 Configuration | PARTIAL | `ApplicationSecurityProfile` ids |
| D7 Secrets | N/A | |
| D8 DI | PARTIAL | `HookContext` rules §12 |
| D9 Registration/discovery | PARTIAL | bootstrap path; override semantics **omitted** (F005) |
| D10 Qualification | PARTIAL | §15 generic |
| D11 Runtime invocation | PARTIAL | Middleware pipeline described |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | PARTIAL | Fail mode §12; **duplicate override=True not documented** |
| D14 Testing | PARTIAL | Fixture cited |
| D15 Production checklist | PARTIAL | §12 author checklist |
| D16 Troubleshooting | MISSING | |

**Overall: PARTIAL**

### 11 — Policy rules (`intergrax.policy_rules`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | PARTIAL | §10 brief |
| D2 Public contract | PARTIAL | `PolicyRuleHandler` Protocol — no method semantics in guide |
| D3 Minimal implementation | MISSING | No handler example |
| D4 External package | PARTIAL | EP group named only |
| D5 Local embedded | MISSING | Bundle bootstrap only |
| D6 Configuration | PARTIAL | YAML path + inline rules |
| D7 Secrets | N/A | |
| D8 DI | MISSING | |
| D9 Registration/discovery | PARTIAL | `register_policy_rule_plugins` one-liner |
| D10 Qualification | PARTIAL | §15 generic |
| D11 Runtime invocation | MISSING | No `PolicyEngine` evaluate path |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | MISSING | |
| D14 Testing | MISSING | |
| D15 Production checklist | MISSING | |
| D16 Troubleshooting | MISSING | |

**Overall: MISSING**

### 12 — Tool invocation patterns (`intergrax.tool_invocation_patterns`)

| D | Score | Notes |
|---|-------|-------|
| D1 Purpose | PARTIAL | TOOLS.md orchestration layer; no author framing |
| D2 Public contract | PARTIAL | `ToolInvocationPattern` in source + ADR-TOOL-003 |
| D3 Minimal implementation | PARTIAL | Shipped patterns internal only |
| D4 External package | MISSING | No EP/pyproject author doc |
| D5 Local embedded | MISSING | Mode config only per §20.3 |
| D6 Configuration | PARTIAL | `ToolInvocationMode` in TOOLS.md |
| D7 Secrets | N/A | |
| D8 DI | PARTIAL | `execute(state, invoker, planner, …)` in architecture |
| D9 Registration/discovery | MISSING | `load_tool_invocation_pattern` not in author docs |
| D10 Qualification | MISSING | |
| D11 Runtime invocation | PARTIAL | TOOLS.md runtime path |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | MISSING | |
| D14 Testing | PARTIAL | `test_tool_invocation_registry.py` not linked |
| D15 Production checklist | MISSING | |
| D16 Troubleshooting | MISSING | |

**Overall: MISSING**

### Matrix roll-up (D1–D16 cell counts across 12 surfaces)

| Score | Cells (of 192) | % |
|-------|----------------|---|
| COMPLETE | 52 | 27% |
| PARTIAL | 89 | 46% |
| MISSING | 47 | 25% |
| N/A | 4 | 2% |

---

## 5. Reference example matrix

| Surface | Example quality | Path / evidence |
|---------|-----------------|-----------------|
| Integrations | Public copyable (in-repo) | `intergrax/integrations/examples/custom_memory_kv/` |
| Tools | **External installable reference package** | `examples/platform_plugins/intergrax_reference_tool_plugin/` |
| Tools | Public local embedded | `examples/platform_plugins/local_embedded_tool_extension/` |
| Skills | Public copyable (in-repo) | `intergrax/skills/examples/custom_pack/` |
| Context | **No example** | Builtin EP only |
| Memory stores | **Test fixture only** | `tests/fixtures/plugin_packages/memory_store_plugin/` |
| RAG chunkers | Guide skeleton (not separate package) | RAG_EXTENSION_GUIDE §15 |
| RAG retrievers | Guide skeleton | RAG_EXTENSION_GUIDE §15 |
| RAG rerankers | Guide skeleton | RAG_EXTENSION_GUIDE §15 |
| Vendor Knowledge | **Test/reference artifact** | `tests/reference_plugins/vendor_knowledge/acme_reference/` |
| Security defenses | **Test fixture only** | `tests/fixtures/plugin_packages/intergrax_security_defense_fixture/` |
| Policy rules | **No example** | YAML lab file only (`harness_lab.yaml`) |
| Tool invocation patterns | **Internal implementation** | `intergrax/runtime/nexus/tools/tool_invocation_pattern.py` |

### Summary

| Category | Surfaces |
|----------|----------|
| External installable reference package | **1** (Tools) |
| Public copyable in-repo examples | **3** (Integrations, Tools local, Skills) |
| Guide-embedded skeleton (no wheel) | **3** (RAG ×3) |
| Fixture/test only | **3** (Memory, Security, VK) |
| No usable third-party example | **3** (Context, Policy, Tool invocation) |

---

## 6. Discoverability and navigation assessment

### Intended path

`README` / docs architecture → Platform Plugins → Extension Author Guide → domain guide → reference example

### Current routing

| Entry | Routes to Platform Plugins? | Routes to Extension Author Guide? | Routes to domain guides? |
|-------|----------------------------|--------------------------------|--------------------------|
| [`docs/project/README.md`](../../README.md) | **Yes** §Explore | **No** direct link | Partial (Integrations hub) |
| [`DOCUMENTATION_MAP.md`](../../technical/DOCUMENTATION_MAP.md) | **No** explicit Platform Plugins row | Via guides/README only | RAG/VK not indexed for plugins |
| [`guides/README.md`](../../technical/guides/README.md) | **No** | **Yes** | VK/RAG not listed |
| [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) | — | **Yes** header link | §20.3 matrix |
| [`EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md) | **Yes** §13 | — | Partial (RAG/VK external) |

### Issues

| Issue | Severity |
|-------|----------|
| No single "Platform Plugin developer" path in DOCUMENTATION_MAP | MEDIUM |
| EXTENSION_AUTHOR_GUIDE not linked from guides/README as **start here for extensions** | LOW |
| Context/Memory/Policy/Tool-invocation lack links from §20.3 matrix to author procedures | HIGH |
| Maintainer plan PLATFORM_PLUGINS.md stale Context status | INFO |
| Fixed in this audit: broken relative links in EXTENSION_AUTHOR_GUIDE §9–12 (`architecture/`, `guides/` prefixes) | LOW (corrected) |
| MEMORY.md §11.5 reference was invalid — corrected to §5.3 | LOW (corrected) |

### Conflicting docs

- **Resolved:** Context "Planned" vs public EP — PLUGIN-9 fixed table to "Public EP" (closeout §Documentation gaps fixed).
- **Remaining:** Maintainer roadmap still mentions partial Context rollout (stale).

### Dead / misleading links (pre-fix)

- `EXTENSION_AUTHOR_GUIDE.md` §9: `architecture/MEMORY.md` → fixed to `../../architecture/MEMORY.md`; removed nonexistent §11.5.
- `EXTENSION_AUTHOR_GUIDE.md` §10, §12: `guides/AGENT_CREATION_GUIDE.md` and `architecture/UNIFIED_EXECUTION_RUNTIME.md` → fixed.

---

## 7. Accuracy and drift findings

Cross-check against runtime/API and PLATFORM-PLUGIN-AUDIT-1 findings:

| Audit ID | Documented accurately? | Where | Doc gap? |
|----------|------------------------|-------|----------|
| F001 Platform version authority | **Yes** at L1 | PLATFORM_PLUGINS §20.4 | **Yes** — not repeated in domain guides |
| F002 Semantic qualification | **Yes** at L1/L2 | §18, EXTENSION_AUTHOR_GUIDE §15 | Partial in domain guides |
| F003 Discovery default-off | **Yes** | §9, EXTENSION_AUTHOR_GUIDE §1 | Easy to miss in domain-only reads |
| F004 Fail-fast loaders | **Yes** at L1 | §22 TARGET vs CURRENT | RAG guide mentions PluginLoadError; others thin |
| F005 Security defense override | **Yes** at L1 §17 | Audit doc | **No** in EXTENSION_AUTHOR_GUIDE §12 |
| F006 No unified inventory | **Yes** at L1 §19 | Audit doc | No author troubleshooting |
| F007 Process-global catalogs | **Yes** at L1 | Audit doc | Not in author guide |
| F010 Memory bootstrap semantics | **Partial** | §9 says bootstrap; **does not** state count-only | **Doc bug** |
| F011 Loose error typing | N/A for authors | Audit §17 | Minor |
| F012 Uneven local DX | **Yes** | §20.3 | By design acknowledged |
| F016 Lifecycle vocabulary-only | **Yes** | §14 | Honest |
| F017 Trusted in-process only | **Yes** | §16 | Complete |

### Code vs doc verification (sampled)

| Claim | Verified |
|-------|----------|
| `bootstrap_memory_stores` returns counts only | **Yes** — `memory_bootstrap.py` lines 38–62 |
| `ContextPlugin` protocol methods | **Yes** — `context/plugin.py` |
| Tools reference wheel EP `intergrax.tools:reference_prefix_echo` | **Yes** — reference package pyproject |
| RAG EP groups (3 only) | **Yes** — RAG_EXTENSION_GUIDE §1 matches discovery constants |
| VK excluded from Tier-0 discovery.py | **Yes** — AUDIT F018 |

---

## 8. Missing document inventory

| ID | Missing artifact | Surfaces | Priority |
|----|------------------|----------|----------|
| MD-01 | `CONTEXT_PLUGIN_AUTHOR_GUIDE.md` (or EXTENSION_AUTHOR_GUIDE §) | Context | P1 |
| MD-02 | `MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md` | Memory stores | P1 |
| MD-03 | `POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md` | Policy rules | P1 |
| MD-04 | `TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md` (or TOOLS satellite) | Tool invocation | P1 |
| MD-05 | Extension surface **decision tree** in EXTENSION_AUTHOR_GUIDE | All | P1 |
| MD-06 | `SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md` (expand §12) | Security | P2 |
| MD-07 | External installable RAG reference package | RAG ×3 | P2 |
| MD-08 | External installable VK reference package (beyond test tree) | VK | P2 |
| MD-09 | DOCUMENTATION_MAP Platform Plugin developer route | Navigation | P2 |
| MD-10 | Per-surface troubleshooting index | All | P3 |
| MD-11 | INTEGRATIONS.md third-party author routing section | Integrations | P3 |

---

## 9. Findings (PLUGIN-DOC-F###)

### HIGH

#### PLUGIN-DOC-F001

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Surface** | Context |
| **Evidence** | No dedicated author §; CONTEXT_ENGINEERING.md lists APIs but no minimal implementation, EP pyproject, or runtime path |
| **Missing/incorrect content** | Full D3/D4/D11/D14–D16 developer journey |
| **Impact** | Developer cannot implement `ContextPlugin` from docs alone |
| **Required documentation artifact** | MD-01 |
| **Production API change required?** | No |

#### PLUGIN-DOC-F002

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Surface** | Memory stores |
| **Evidence** | EXTENSION_AUTHOR_GUIDE §9 only; fixture-only example; `bootstrap_memory_stores` count-only (`memory_bootstrap.py`) |
| **Missing/incorrect content** | Factory selection wiring, `MemoryProfile`, pyproject EP example, bootstrap semantics clarification |
| **Impact** | Developer cannot wire memory store plugins from docs alone; bootstrap API misleading |
| **Required documentation artifact** | MD-02 |
| **Production API change required?** | No (doc must reflect current API); optional future: register-to-catalog helper |

#### PLUGIN-DOC-F003

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Surface** | Tool invocation patterns |
| **Evidence** | Absent from EXTENSION_AUTHOR_GUIDE; TOOLS.md runtime-focused; AUDIT-1 matrix "Not documented" |
| **Missing/incorrect content** | EP declaration, `load_tool_invocation_pattern`, minimal pattern, qualification |
| **Impact** | Third-party pattern authors must read Tier-1 Nexus source |
| **Required documentation artifact** | MD-04 |
| **Production API change required?** | No |

#### PLUGIN-DOC-F004

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Surface** | Policy rules |
| **Evidence** | EXTENSION_AUTHOR_GUIDE §10 (~15 lines); no `PolicyRuleHandler` implementation example |
| **Missing/incorrect content** | Handler protocol semantics, minimal handler, runtime evaluation path, tests |
| **Impact** | Custom policy handlers not implementable from docs |
| **Required documentation artifact** | MD-03 |
| **Production API change required?** | No |

#### PLUGIN-DOC-F005

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Surface** | All (Level 2) |
| **Evidence** | Task requires decision tree; EXTENSION_AUTHOR_GUIDE has summary table only |
| **Missing/incorrect content** | "Which extension surface do I need?" decision tree with when-not-to-use |
| **Impact** | New developers cannot self-route to correct surface |
| **Required documentation artifact** | MD-05 |
| **Production API change required?** | No |

### MEDIUM

#### PLUGIN-DOC-F006

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | Context, Memory, RAG×3, Security, Policy, Tool invocation (9/12) |
| **Evidence** | PLATFORM_PLUGINS §20.3, AUDIT F012 |
| **Missing/incorrect content** | Local embedded delivery procedures parity with Tools |
| **Impact** | Application developers lack documented host-embedded path |
| **Required documentation artifact** | Per-surface guides or explicit "external-EP-only" statements |
| **Production API change required?** | No for doc; RUNTIME_CAPABILITY_GAP for scaffold parity |

#### PLUGIN-DOC-F007

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | All |
| **Evidence** | AUDIT F002; documented L1/L2 only |
| **Missing/incorrect content** | Domain guides should state qualification is host-trusted semantic metadata |
| **Impact** | Misinterpretation as cryptographic attestation |
| **Required documentation artifact** | Standard disclaimer block in domain guides |
| **Production API change required?** | RUNTIME_CAPABILITY_GAP (enterprise attestation) |

#### PLUGIN-DOC-F008

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | All external packages |
| **Evidence** | AUDIT F001; PLATFORM_PLUGINS §20.4 |
| **Missing/incorrect content** | Host must supply explicit `platform_version` — not in domain quickstarts |
| **Impact** | Inconsistent compatibility checks across hosts |
| **Required documentation artifact** | Compatibility subsection in each external-package guide |
| **Production API change required?** | RUNTIME_CAPABILITY_GAP (canonical version API) |

#### PLUGIN-DOC-F009

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | Memory stores |
| **Evidence** | AUDIT F010; `bootstrap_memory_stores` returns counts |
| **Missing/incorrect content** | §9 implies bootstrap registers plugins |
| **Impact** | Operators expect catalog registration that does not occur |
| **Required documentation artifact** | MD-02 §bootstrap semantics |
| **Production API change required?** | No |

#### PLUGIN-DOC-F010

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | Security defenses |
| **Evidence** | AUDIT F005; `defense_plugin_loader.py` `override=True` |
| **Missing/incorrect content** | Duplicate `plugin_id` silently overridden by EP |
| **Impact** | Authors cannot diagnose defense registration conflicts |
| **Required documentation artifact** | MD-06 failure/D13 section |
| **Production API change required?** | Optional HARDENING (configurable policy) |

#### PLUGIN-DOC-F011

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | RAG, VK, Skills |
| **Evidence** | Reference example matrix §5 |
| **Missing/incorrect content** | No installable external wheel examples except Tools |
| **Impact** | Higher friction validating packaging/discovery |
| **Required documentation artifact** | MD-07, MD-08 |
| **Production API change required?** | No |

#### PLUGIN-DOC-F012

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | Integrations |
| **Evidence** | INTEGRATIONS.md provider-focused; third-party path only via EXTENSION_AUTHOR_GUIDE |
| **Missing/incorrect content** | Architecture hub routing to third-party plugin author path |
| **Impact** | Architects miss EXTENSION_AUTHOR_GUIDE |
| **Required documentation artifact** | MD-11 |
| **Production API change required?** | No |

#### PLUGIN-DOC-F013

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Surface** | Navigation |
| **Evidence** | DOCUMENTATION_MAP.md lacks Platform Plugin developer route |
| **Missing/incorrect content** | Indexed path: Platform Plugins → Extension Author → domain guides |
| **Impact** | Developers bury-led through architecture only |
| **Required documentation artifact** | MD-09 |
| **Production API change required?** | No |

### LOW

#### PLUGIN-DOC-F014

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Surface** | Navigation |
| **Evidence** | EXTENSION_AUTHOR_GUIDE §9–12 had wrong relative paths |
| **Missing/incorrect content** | Fixed in this audit (see Changed files) |
| **Impact** | Broken links from author guide |
| **Required documentation artifact** | — (fixed) |
| **Production API change required?** | No |

#### PLUGIN-DOC-F015

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Surface** | Memory stores |
| **Evidence** | §9 cited MEMORY.md §11.5 (section does not exist) |
| **Missing/incorrect content** | Fixed to §5.3 |
| **Impact** | Dead anchor |
| **Required documentation artifact** | — (fixed) |
| **Production API change required?** | No |

#### PLUGIN-DOC-F016

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Surface** | All |
| **Evidence** | AUDIT F016; lifecycle enum without runtime tracking |
| **Missing/incorrect content** | Author-facing note that lifecycle states are vocabulary-only |
| **Impact** | Expectation of unload/reload APIs |
| **Required documentation artifact** | EXTENSION_AUTHOR_GUIDE lifecycle note |
| **Production API change required?** | RUNTIME_CAPABILITY_GAP |

#### PLUGIN-DOC-F017

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Surface** | All |
| **Evidence** | AUDIT F006 |
| **Missing/incorrect content** | Troubleshooting "what plugins are active" without unified inventory |
| **Impact** | Longer incident resolution |
| **Required documentation artifact** | MD-10 per-domain inventory workarounds |
| **Production API change required?** | RUNTIME_CAPABILITY_GAP |

### INFO

#### PLUGIN-DOC-F018

| Field | Value |
|-------|-------|
| **Severity** | INFO |
| **Surface** | Context |
| **Evidence** | `maintainers/plans/PLATFORM_PLUGINS.md` line 56 |
| **Missing/incorrect content** | Update stale "partial rollout" maintainer note |
| **Impact** | Maintainer confusion only |
| **Required documentation artifact** | Maintainer plan sync |
| **Production API change required?** | No |

#### PLUGIN-DOC-F019

| Field | Value |
|-------|-------|
| **Severity** | INFO |
| **Surface** | All |
| **Evidence** | PLATFORM_PLUGINS §16 |
| **Missing/incorrect content** | Trusted in-process model well documented |
| **Impact** | Positive — limitations not hidden |
| **Required documentation artifact** | None |
| **Production API change required?** | No |

---

## 10. RUNTIME_CAPABILITY_GAP list

Documentation cannot fully close these without runtime/API work (per AUDIT-1 / enterprise roadmap):

| Gap | Audit ref | Blocks doc completion? |
|-----|-----------|------------------------|
| Canonical platform version authority API | F001 | Partial — docs can require host discipline; API gap remains |
| Qualification provenance / attestation | F002 | Partial — docs must label semantic qualification |
| Unified plugin inventory / operator surface | F006 | Yes for D16 troubleshooting at platform level |
| Runtime lifecycle state tracking | F016 | Yes for D12 beyond vocabulary |
| Local registration scaffold parity (9/12 surfaces) | F012 | Partial — can document manual paths; scaffold gap remains |
| EP load failure isolation in production loaders | F004 | Partial — document current fail-fast; hardening future |
| Configurable security defense conflict policy | F005 | Partial — document override=True today |

---

## 11. Documentation remediation roadmap

Coherent implementation blocks derived from evidence (not microtasks):

### DOCS-2 — Platform architecture + navigation

- Add Platform Plugin developer route to `DOCUMENTATION_MAP.md` and `guides/README.md`.
- Sync stale maintainer plan Context note (F018).
- Add EXTENSION_AUTHOR_GUIDE **§0.1 extension surface decision tree** (F005).
- Cross-link §20.3 matrix rows to target domain guides (existing or new).

### DOCS-3 — Integrations / Tools / Skills

- **Tools:** maintenance only (reference surface).
- **Integrations:** add INTEGRATIONS.md third-party author routing; expand secrets/env_prefix in §2.
- **Skills:** add `SkillResolver` runtime path, external package quickstart mirroring §16 Tools pattern.

### DOCS-4 — Context / Memory / RAG

- **Context:** new `CONTEXT_PLUGIN_AUTHOR_GUIDE.md` — protocol, minimal plugin, EP, `ContextProfile`, bootstrap, tests (MD-01).
- **Memory:** new `MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md` — factory protocols, host wiring, fix bootstrap semantics (MD-02, F009).
- **RAG:** optional `examples/platform_plugins/intergrax_reference_rag_plugin/` wheel (MD-07); document external-EP-only for local.

### DOCS-5 — Security / Policy / Tool invocation

- Expand security defense author material — override semantics, pyproject, minimal plugin (MD-06, F010).
- New policy rule handler guide with minimal `PolicyRuleHandler` (MD-03).
- New tool invocation pattern author guide — EP, `ToolInvocationMode`, minimal pattern (MD-04).

### DOCS-6 — Vendor Knowledge + cross-domain examples

- Promote or repackage `acme_reference` as installable example under `examples/platform_plugins/` (MD-08).
- Multi-capability package example building on `intergrax_catalog_fixture` pattern.

### DOCS-7 — Final developer-doc validation

- Verify all 12 surfaces: docs-only implementation walkthrough.
- Link check CI for author guide paths.
- Re-run PLUGIN-DOC matrix; target ≥10/12 PARTIAL→COMPLETE for external-EP paths.

---

## 12. Validation performed

| Check | Result |
|-------|--------|
| Referenced canonical files exist | **Pass** (sampled paths verified) |
| Code examples match public APIs | **Pass** (Tools, RAG, Integration examples sampled) |
| Relative links in EXTENSION_AUTHOR_GUIDE §9–12 | **Fixed** (3 corrections) |
| Claims align with PLATFORM_PLUGINS + AUDIT-1 | **Pass** |
| Production code changed | **No** (link fixes only in author guide) |

---

## 13. Evidence index

| Artifact | Role |
|----------|------|
| [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) | Level 1 canon |
| [`EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md) | Level 2 canon |
| [`RAG_EXTENSION_GUIDE.md`](../../technical/guides/RAG_EXTENSION_GUIDE.md) | RAG domain canon |
| [`VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md`](../../technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md) | VK domain canon |
| [`PLATFORM_PLUGIN_PRODUCTION_AUDIT.md`](PLATFORM_PLUGIN_PRODUCTION_AUDIT.md) | AUDIT-1 findings |
| [`PLATFORM_PLUGIN_9_CLOSEOUT.md`](PLATFORM_PLUGIN_9_CLOSEOUT.md) | Program closeout |
| `examples/platform_plugins/intergrax_reference_tool_plugin/` | Executable external reference |
| `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` | Dual-mode proof |

---

*End of PLATFORM-PLUGIN-DOCS-1 audit artifact.*

---

## 14. DOCS-3 remediation status (PLATFORM-PLUGIN-DOCS-3)

**Date:** 2026-08-12 · **Branch:** `development` · **Scope:** Integrations, Tools, Skills developer journeys only

### Matrix outcome (D1–D16)

| Surface | DOCS-1 | DOCS-3 | Notes |
|---------|--------|--------|-------|
| **Integrations** | PARTIAL | **COMPLETE** | EXTENSION_AUTHOR_GUIDE §2 expanded; INTEGRATIONS.md third-party path; `env_prefix`/secrets/lifecycle/troubleshooting documented |
| **Tools** | COMPLETE | **COMPLETE** | Reference flows preserved; §17 + TOOLS.md lifecycle/failure/troubleshooting added |
| **Skills** | PARTIAL | **COMPLETE** | EXTENSION_AUTHOR_GUIDE §4 + §16.6–§16.7; SKILLS.md third-party path; `SkillResolver` runtime documented; `custom_pack` remains in-repo only (not falsely labeled installable) |

### Shared blocks aligned

- Discovery: `installed ≠ discovered ≠ enabled ≠ production-qualified`
- Trust: trusted in-process Python
- Qualification: host-owned semantic approval, not attestation
- Secrets: not in Platform Plugin metadata or EP values
- Local vs external: same domain contract, different delivery where supported

### Runtime capability gaps

**None** for Integrations, Tools, or Skills third-party author paths covered in DOCS-3. Generic Platform Plugin lifecycle unload API remains intentionally absent (documented as host/category ownership).

### Changed documentation (DOCS-3 allowlist)

1. `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md`
2. `docs/project/architecture/INTEGRATIONS.md`
3. `docs/project/architecture/TOOLS.md`
4. `docs/project/architecture/SKILLS.md`
5. `docs/project/maintainers/plans/PLATFORM_PLUGIN_DOCUMENTATION_AUDIT.md` (this section)

---

## 15. DOCS-4 remediation status (PLATFORM-PLUGIN-DOCS-4)

**Date:** 2026-08-12 · **Branch:** `development` · **Scope:** Context, Memory stores, RAG chunker/retriever/reranker developer journeys

### Matrix outcome (D1–D16)

| Surface | DOCS-1 | DOCS-4 | Notes |
|---------|--------|--------|-------|
| **Context** | MISSING | **COMPLETE** | New `CONTEXT_PLUGIN_AUTHOR_GUIDE.md` — `ContextPlugin`, EP, `ContextProfile`, bootstrap, runtime, tests, troubleshooting |
| **Memory stores** | MISSING | **PARTIAL** | New `MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md` — bootstrap count-only corrected; `SessionTurnIndexStorePlugin` wired; user/session EP resolver gap documented |
| **RAG chunker** | PARTIAL | **COMPLETE** | `RAG_EXTENSION_GUIDE.md` §0 journey matrix, local path §0.2, `RagProfile` runtime §0.3 |
| **RAG retriever** | PARTIAL | **COMPLETE** | Same guide upgrade |
| **RAG reranker** | PARTIAL | **COMPLETE** | Same guide upgrade |

### Shared blocks aligned

- Discovery: `installed ≠ discovered ≠ enabled ≠ production-qualified`
- Trust: trusted in-process Python
- Qualification: host-owned semantic approval; RAG live-backend qualification separated (§0.4)
- Secrets: not in Platform Plugin metadata or EP values
- Lifecycle: no universal Platform Plugin unload API

### RUNTIME_CAPABILITY_GAPS

| Gap | Surfaces affected |
|-----|-------------------|
| No shipped Tier-3 resolver for `UserProfileStorePlugin.create_user_profile_store` from EP discovery | Memory — user profile store |
| No shipped Tier-3 resolver for `SessionStoragePlugin.create_session_storage` from EP discovery | Memory — session storage |
| Context scaffold CLI parity with Tools | Context — DX only; external-EP path complete |

### REFERENCE_EXAMPLE_GAPS (deferred to DOCS-6)

| Gap | Notes |
|-----|-------|
| Installable external Context plugin wheel under `examples/platform_plugins/` | Documentation + unit-test patterns suffice for DOCS-4 |
| Installable external Memory store wheel | Test fixtures labeled; no production sample package |
| `examples/platform_plugins/intergrax_reference_rag_plugin/` | RAG guide §15 skeleton + `test_rag_plugin_discovery.py` suffice |

### Changed documentation (DOCS-4 allowlist)

1. `docs/project/technical/guides/CONTEXT_PLUGIN_AUTHOR_GUIDE.md` (new)
2. `docs/project/technical/guides/MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md` (new)
3. `docs/project/technical/guides/RAG_EXTENSION_GUIDE.md`
4. `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md`
5. `docs/project/architecture/CONTEXT_ENGINEERING.md`
6. `docs/project/architecture/MEMORY.md`
7. `docs/project/architecture/RAG.md`
8. `docs/project/maintainers/plans/PLATFORM_PLUGIN_DOCUMENTATION_AUDIT.md` (this section)

---

## 16. DOCS-5 remediation status (PLATFORM-PLUGIN-DOCS-5)

**Date:** 2026-08-12 · **Branch:** `development` · **Scope:** Security defenses, Policy rule handlers, Tool invocation patterns

### Matrix outcome (D1–D16)

| Surface | DOCS-1 | DOCS-5 | Notes |
|---------|--------|--------|-------|
| **Security defenses** | PARTIAL | **COMPLETE** | New `SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md` — contract, EP, profile enablement, `override=True` semantics, failure/troubleshooting |
| **Policy rules** | MISSING | **PARTIAL** | New `POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md` — contract + packaging; runtime EP bootstrap and declarative enforcement gaps documented |
| **Tool invocation patterns** | MISSING | **COMPLETE** | New `TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md` — contract, EP, mode resolution, local instance override, F009 performance |

### Shared blocks aligned

- Discovery: `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`
- Trust: trusted in-process Python
- Qualification: host-owned semantic approval, not attestation
- Secrets: not in Platform Plugin metadata or EP values
- Local parity classified per surface (see guides §5)

### RUNTIME_CAPABILITY_GAPS

| Gap | Surfaces affected | Evidence |
|-----|-------------------|----------|
| `wire_policy_bundle` does not call `load_policy_rule_plugins` | Policy | `policy_wiring.py` creates `PolicyRuleRegistry()` only; loader exists but unused in production wiring |
| Declarative `policy_rules` in `domain_fragments` not evaluated at runtime | Policy | No production caller of `PolicyRuleRegistry.evaluate_rule` |
| Security/policy EP loaders fail-fast (no isolate) | Security, Policy | `defense_plugin_loader.py`, `plugin_loader.py` — AUDIT F004 |

### REFERENCE_EXAMPLE_GAPS (deferred to DOCS-6)

| Gap | Notes |
|-----|-------|
| Installable external Security defense wheel under `examples/platform_plugins/` | Test fixture `intergrax_security_defense_fixture` suffices for DOCS-5 |
| Installable external Policy rule handler wheel | Unit-test patterns in `test_plugin_discovery.py` |
| Installable external Tool invocation pattern wheel | Unit-test `_CustomPattern` in `test_tool_invocation_registry.py` |

### Failure isolation (current semantics)

| Loader | Bootstrap vs lazy | One broken plugin blocks group? | Isolation |
|--------|-------------------|----------------------------------|-----------|
| Security (`load_security_defense_plugins`) | Catalog bootstrap | **Yes** — first `PluginLoadError` / `TypeError` aborts | None |
| Policy (`load_policy_rule_plugins`) | Host-invoked only | **Yes** — same fail-fast | None |
| Tool invocation (`load_tool_invocation_pattern`) | Per lookup | **No** — fails only matching id lookup | Unrelated EP names not loaded |

### LOCAL_PARITY classification

| Surface | Local path | Classification |
|---------|------------|----------------|
| Security | `register_security_defense_plugin()` + profile ids | Advanced host composition |
| Policy | `PolicyRuleRegistry.register()` + explicit loader | Advanced host composition; external-EP-first for discovery |
| Tool invocation | `RuntimeConfig.tool_invocation_pattern` instance | Advanced host composition; EP for discoverable ids |

### ENTERPRISE_ROADMAP_CANDIDATES

| ID | Surface | Gap | Category | Evidence | Why enterprise | Priority suggestion |
|----|---------|-----|----------|----------|----------------|---------------------|
| PLUGIN-ENT-CAND-001 | Memory | No Tier-3 EP resolver for `UserProfileStorePlugin.create_user_profile_store` | EXTENSIBILITY | DOCS-4 carry-forward | Multi-tenant apps need host-owned store wiring without custom glue | Medium |
| PLUGIN-ENT-CAND-002 | Memory | No Tier-3 EP resolver for `SessionStoragePlugin.create_session_storage` | EXTENSIBILITY | DOCS-4 carry-forward | Same as CAND-001 for session tier | Medium |
| PLUGIN-ENT-CAND-003 | Context | Scaffold CLI parity with Tools | DX | DOCS-4 carry-forward | Application-team onboarding; not security-critical | Low |
| PLUGIN-ENT-CAND-004 | Security | EP defense registration always `override=True` — shipped ids replaceable without policy | GOVERNANCE / SECURITY | `defense_plugin_loader.py:29`, AUDIT F005 | Enterprise needs configurable deny-by-default / audit on defense collision | High |
| PLUGIN-ENT-CAND-005 | Security, Policy | EP loaders fail-fast — one broken plugin blocks entire group bootstrap | RELIABILITY / OPERATOR_CONTROL | AUDIT F004; bespoke loaders | Noisy neighbor EP should not deny whole security/policy surface | Medium |
| PLUGIN-ENT-CAND-006 | Policy | `load_policy_rule_plugins` not wired from `wire_policy_bundle` | EXTENSIBILITY / DX | `policy_wiring.py` vs `plugin_loader.py` | Third-party handlers require undocumented host glue | High |
| PLUGIN-ENT-CAND-007 | Policy | Declarative YAML rules not enforced via `evaluate_rule` at runtime | GOVERNANCE | `domain_fragments["policy_rules"]` only | Policy-as-code without runtime effect is enterprise governance gap | High |
| PLUGIN-ENT-CAND-008 | Policy | No centrally governed handler allowlist / signed policy bundles | GOVERNANCE / OPERATOR_CONTROL | No allowlist API in registry | Regulated tenants need approved handler + bundle provenance | Medium |
| PLUGIN-ENT-CAND-009 | Tool invocation | O(N) EP scan per `load_tool_invocation_pattern` lookup | SCALABILITY | AUDIT F009; `tool_invocation_registry.py` | Hardening at scale; ordinary perf work unless inventory required | Low (hardening) |

**ENTERPRISE_ROADMAP_CANDIDATE classifications (YES/NO):**

| ID | YES/NO | Rationale |
|----|--------|-----------|
| CAND-004 | **YES** | Security collision policy is governance/security enterprise requirement |
| CAND-005 | **YES** | Operator control over partial catalog degradation |
| CAND-006 | **YES** | Blocks complete third-party policy handler path without custom host code |
| CAND-007 | **YES** | Policy without enforcement is enterprise governance gap |
| CAND-008 | **YES** | Regulated environments need allowlist/provenance |
| CAND-009 | **NO** | Ordinary hardening — cache/index; not enterprise differentiator alone |

### Changed documentation (DOCS-5 allowlist)

1. `docs/project/technical/guides/SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md` (new)
2. `docs/project/technical/guides/POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md` (new)
3. `docs/project/technical/guides/TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md` (new)
4. `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md`
5. `docs/project/architecture/PLATFORM_PLUGINS.md`
6. `docs/project/architecture/TOOLS.md`
7. `docs/project/maintainers/plans/PLATFORM_PLUGIN_DOCUMENTATION_AUDIT.md` (this section)
