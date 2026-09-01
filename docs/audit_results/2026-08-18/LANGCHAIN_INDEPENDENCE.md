# LANGCHAIN_INDEPENDENCE - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer/capability:** LANGCHAIN_INDEPENDENCE
- **Tier(s):** Cross-layer capability - Tier-0 `intergrax/compat/langchain/`, `intergrax/knowledge/`, `intergrax/rag/`, `intergrax/llm_adapters/`, `intergrax/integrations/`, packaging/CI gates
- **audited_sha:** `70c947c889f40222e5efb191241bdd8fa9035b17`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/capabilities/architecture/LANGCHAIN_INDEPENDENCE.md`
- **Plan doc(s):**
  - `docs/project/capabilities/plan/LANGCHAIN_INDEPENDENCE.md`
- **Scope in:**
  - LangChain document compatibility bridge (`from_langchain_document`, `from_legacy_rag_hit`)
  - trusted scope authority for compatibility conversion
  - `check_langchain_boundary.py` allowed-zone and static dynamic-import detection
  - `pyproject.toml` compatibility extras vs parsing extras
  - dependency inventory and conformance evidence pinning
  - historical LCI-0A..8A delivery as positive controls
- **Scope out:**
  - remediation implementation
  - source/test/CI/script/packaging changes
  - creating domain-layer `docs/project/architecture/LANGCHAIN_INDEPENDENCE.md` or `docs/project/maintainers/plans/LANGCHAIN_INDEPENDENCE.md`
  - second RAG/document/LLM abstraction
  - rewriting historical LCI closeout rows as if delivery never occurred
- **Prior audit reference(s):** [`RAG`](RAG.md); [`LLM_ADAPTERS`](LLM_ADAPTERS.md); [`INTEGRATIONS`](INTEGRATIONS.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `1b1151cca`

## Executive summary

**Verdict: FAIL.** Four accepted HIGH and two accepted MEDIUM findings show that compatibility conversion can mint canonical tenant/workspace/namespace scope from foreign LangChain or legacy provider metadata; the boundary guard blanket-trusts entire provider directories and misses common `importlib` alias forms; parsing extras can install LangChain without selecting named compatibility extras; and current conformance evidence is pinned to a historical inventory SHA while architecture presents counts as current. Positive controls: strategic LangChain-free core + optional compatibility remains sound; default dependencies contain no `langchain*`/`langgraph*`; clean-install gate blocks forbidden distributions and validates installed package origin; native `KnowledgeDocument` and LLM contracts remain canonical; `NativeOllamaAdapter` remains default; LangGraph remains optional legacy boundary; historical LCI program delivery facts remain valid. Remediation is **PLANNED**, not implemented. Findings harden compatibility trust and conformance enforcement only.

## Verdict

**FAIL** - 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-LANGCHAIN_INDEPENDENCE-01

- **Severity:** HIGH
- **Category:** SECURITY / SCOPE AUTHORITY
- **Status at publication:** ACCEPTED
- **Remediation block:** LCI-COMPATIBILITY-SCOPE-INTEGRITY
- **Claim falsified:** Compatibility conversion cannot establish canonical tenant/workspace/namespace authority from foreign LangChain document metadata alone.
- **Observation:** `from_langchain_document()` resolves canonical `tenant_id`, `namespace`, and `workspace_id` directly from LangChain `Document` metadata and builds `KnowledgeDocument.scope` from those values. This violates the documented system-owned scope boundary where user/foreign metadata may only match or confirm trusted scope.
- **Location:**
  - `intergrax/compat/langchain/documents.py` - `from_langchain_document()`
- **Reproduction:** Convert a LangChain `Document` whose metadata carries arbitrary `tenant_id` / `namespace` / `workspace_id`; observe native scope minted from untrusted metadata without external trusted scope injection.
- **Impact:** Foreign metadata can become authoritative routing identity; cross-link [`RAG`](../../project/architecture/RAG.md) **RAG-SCOPE-CONTRACT-INTEGRITY**.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LANGCHAIN_INDEPENDENCE-02

- **Severity:** HIGH
- **Category:** SECURITY / PROVIDER TRUST BOUNDARY
- **Status at publication:** ACCEPTED
- **Remediation block:** LCI-COMPATIBILITY-SCOPE-INTEGRITY
- **Claim falsified:** Legacy/provider hit conversion cannot mint system ownership from provider-returned metadata.
- **Observation:** `from_legacy_rag_hit()` accepts provider hit metadata/content/id, reconstructs a LangChain `Document` via `make_langchain_document()`, then calls `from_langchain_document()`. Therefore actual legacy/provider-returned metadata can become canonical `KnowledgeDocument` scope/provenance authority.
- **Location:**
  - `intergrax/rag/document_loaders/compat/legacy_runtime_document.py` - `from_legacy_rag_hit()`
  - `intergrax/compat/langchain/documents.py` - `from_langchain_document()`
- **Reproduction:** Pass a legacy hit with attacker-controlled scope fields in metadata; observe native document scope derived from provider metadata without trusted expected scope verification.
- **Impact:** Provider-returned metadata can mint tenant/workspace/namespace authority; coordinate with **RAG-SCOPE-CONTRACT-INTEGRITY** and **IDENTITY_TRUST** where applicable. Do not create a second document contract.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LANGCHAIN_INDEPENDENCE-03

- **Severity:** HIGH
- **Category:** ARCHITECTURE ENFORCEMENT / FAIL-OPEN ALLOWLIST
- **Status at publication:** ACCEPTED
- **Remediation block:** LCI-BOUNDARY-ENFORCEMENT-INTEGRITY
- **Claim falsified:** Boundary guard enforces conditional compatibility rules across every allowed provider family, not directory-wide trust.
- **Observation:** `check_langchain_boundary.py` blanket-allows `intergrax/compat/langchain/`, `intergrax/integrations/providers/`, `intergrax/llm_adapters/providers/`, and `intergrax/legacy/`. Architecture permits these paths only conditionally (optional dependency, native mapping, no ABI leakage, controlled/lazy import). The checker applies eager-provider enforcement only to `intergrax/integrations/providers/` via `provider_eager_imports()` and does not apply equivalent checks to `intergrax/llm_adapters/providers/`. A new eager LangChain import in an arbitrary LLM provider can pass allowed-zone classification.
- **Location:**
  - `scripts/maintenance/check_langchain_boundary.py` - `ALLOWED_ZONE_PREFIXES`, `provider_eager_imports()`
- **Reproduction:** Add a module-level `import langchain_core` under `intergrax/llm_adapters/providers/`; observe boundary guard classifies path as allowed without eager-import failure.
- **Impact:** New provider compatibility use can bypass conditional boundary intent; weakens permanent LCI-0B anti-regression posture.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LANGCHAIN_INDEPENDENCE-04

- **Severity:** HIGH
- **Category:** ARCHITECTURE ENFORCEMENT / STATIC-GATE BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** LCI-BOUNDARY-ENFORCEMENT-INTEGRITY
- **Claim falsified:** LCI-0B recognizes common statically resolvable `importlib` aliases and `import_module` aliases for dynamic LangChain imports.
- **Observation:** Boundary checker recognizes literal `importlib.import_module("langchain...")` only when the AST receiver is exactly the name `importlib`. It does not recognize equivalent static forms such as `import importlib as il` + `il.import_module("langchain_core")` or `from importlib import import_module` + `import_module("langchain_core")`.
- **Location:**
  - `scripts/maintenance/check_langchain_boundary.py` - `_extract_importlib_module()`
- **Reproduction:** Use aliased `importlib` or `import_module` binding for a LangChain module in a protected zone; observe guard miss while literal `importlib.import_module` would be detected.
- **Impact:** Protected-zone dynamic-import evasion undermines permanent architecture guard; requires adversarial regression fixtures.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LANGCHAIN_INDEPENDENCE-05

- **Severity:** MEDIUM
- **Category:** PACKAGING / CONTRACT DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** LCI-PACKAGING-EVIDENCE-INTEGRITY
- **Claim falsified:** LangChain compatibility is reachable only through explicitly named LangChain compatibility extras unless packaging contract explicitly declares transitive opt-in.
- **Observation:** Architecture presents five explicit LangChain/LangGraph opt-in compatibility extras (`llm-langchain-ollama`, `rag-langchain-loaders`, `rag-langchain-embeddings`, `rag-langchain-splitters`, `langgraph-legacy`). `pyproject.toml` also places `langchain-community` directly in `parsing-office` and `parsing-pdf`. Users may install LangChain through ordinary parsing extras without selecting a named compatibility extra. This does not break the LangChain-free default install but weakens explicit compatibility opt-in semantics.
- **Location:**
  - `pyproject.toml` - `[project.optional-dependencies]` `parsing-office`, `parsing-pdf`
- **Reproduction:** Install `parsing-office` or `parsing-pdf` extra only; observe `langchain-community` installed without any named `rag-langchain-*` extra.
- **Impact:** Compatibility opt-in semantics are ambiguous for parsing surfaces.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LANGCHAIN_INDEPENDENCE-06

- **Severity:** MEDIUM
- **Category:** EVIDENCE / DOCUMENTATION DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** LCI-PACKAGING-EVIDENCE-INTEGRITY
- **Claim falsified:** Current LangChain independence conformance evidence is mechanically pinned to the audited repository SHA.
- **Observation:** Architecture presents current inventory counts as current capability evidence. The dependency inventory satellite is explicitly pinned to 2026-08-09 commit `20bc04d249f4956c2616cfee222cb10fb067cc2f` while audited repo is `70c947c889f40222e5efb191241bdd8fa9035b17`. The detailed inventory still contains historical current-requirement wording (e.g. LangGraph rows saying required/default-install) while current `pyproject.toml` and architecture classify LangGraph as optional `langgraph-legacy`. `validate_langchain_inventory.py` is a migration-era validator with hard-coded historical replacements rather than a clean current-repo evidence generator.
- **Location:**
  - `docs/project/capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`
  - `docs/project/capabilities/architecture/LANGCHAIN_INDEPENDENCE.md` - as-built baseline counts
  - `scripts/docs/validate_langchain_inventory.py`
- **Reproduction:** Compare architecture hub inventory scale claims with inventory satellite pin date/SHA; compare LangGraph classification in inventory rows vs `pyproject.toml` optional extras.
- **Impact:** Operators and agents may treat stale migration inventory as current conformance proof.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Strategic architecture remains LangChain-free core + optional compatibility | NOT falsified |
| Default `[project].dependencies` contain no `langchain*`/`langgraph*` | NOT falsified |
| Clean-install gate exists and blocks forbidden installed distributions | NOT falsified |
| Clean-install gate validates installed package origin | NOT falsified |
| Native `KnowledgeDocument` remains canonical document ABI | NOT falsified |
| Native LLM contracts remain canonical | NOT falsified |
| `NativeOllamaAdapter` remains canonical/default Ollama path | NOT falsified |
| `LangChainOllamaAdapter` remains compatibility-only | NOT falsified |
| LangGraph remains optional legacy boundary (`KEEP_OPTIONAL`) | NOT falsified |
| No evidence justifies reintroducing LangChain into core | NOT falsified |
| No second RAG/document/LLM abstraction is needed | NOT falsified |
| Findings harden compatibility trust and conformance enforcement only | NOT falsified |
| Historical LCI-0..8 delivery facts remain valid delivery facts | NOT falsified |
| Protocol-v2 FAIL does not mean the historical migration was undone | NOT falsified |

## Historical LCI delivery vs Protocol-v2 residual findings

Historical **COMPLETE / APPROVED** LCI rows (`LCI-0A` … `LCI-8A`, Native Ollama regression gate, FINAL SYSTEM GATE) remain valid delivery facts - native contracts, boundary guard, optional extras, install gates, and migration closeout were delivered as claimed. The six accepted Protocol-v2 findings document **residual compatibility scope authority, boundary-enforcement gaps, packaging opt-in drift, and stale conformance evidence** at `audited_sha`. Remediation hardens the existing LangChain Independence capability; it does **not** reopen closed historical LCI rows, remove optional compatibility, or require a second document/LLM abstraction.

## Root-cause remediation grouping

### LCI-COMPATIBILITY-SCOPE-INTEGRITY - trusted scope injection and provider-hit identity validation

**Priority:** P0  
**Findings:** `AUDIT-20260818-LANGCHAIN_INDEPENDENCE-01`, `02`

Foreign/legacy LangChain/provider metadata never becomes authoritative tenant/workspace/namespace/provenance identity without a trusted external scope. Compatibility conversion receives trusted canonical scope separately; foreign metadata may only match/confirm trusted scope; mismatch fails closed. Cross-link **RAG-SCOPE-CONTRACT-INTEGRITY** and **IDENTITY_TRUST** where applicable. Do not create a second document contract.

### LCI-BOUNDARY-ENFORCEMENT-INTEGRITY - conditional provider exemption and robust static dynamic-import detection

**Priority:** P1  
**Findings:** `AUDIT-20260818-LANGCHAIN_INDEPENDENCE-03`, `04`

The permanent LangChain architecture guard enforces conditional compatibility rules across every provider family and resists common static `importlib` aliases. Allow specific reviewed compatibility boundaries/capabilities, not entire provider directories. Track common statically resolvable `importlib` / `import_module` aliases and add adversarial regression fixtures. No need for general Python execution or unrestricted dynamic analysis.

### LCI-PACKAGING-EVIDENCE-INTEGRITY - explicit compatibility packaging semantics and current conformance evidence

**Priority:** P1/P2  
**Findings:** `AUDIT-20260818-LANGCHAIN_INDEPENDENCE-05`, `06`

Compatibility opt-in semantics are truthful in packaging and current LangChain independence evidence is mechanically pinned to the current repository state. Either native parsing extras are LangChain-free and compatibility loader dependencies live only in named compatibility extras, or docs/package contract explicitly declares transitive opt-in. Preserve historical migration inventory as historical evidence; maintain a separate mechanically generated/current conformance evidence record pinned to a specific SHA. Do not rewrite historical evidence as if it never existed.

## Cross-links to existing remediation

| Existing block | Relationship |
|----------------|--------------|
| **RAG-SCOPE-CONTRACT-INTEGRITY** / **RAG** | Canonical scoped retrieval and document scope authority for LCI-01/02 |
| **IDENTITY_TRUST** | Trusted principal/scope injection where compatibility conversion binds execution identity |
| **LCI-0B** (historical) | Boundary guard baseline extended by LCI-03/04 - do not conflate historical delivery with residual enforcement gaps |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `70c947c889f40222e5efb191241bdd8fa9035b17`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests and CI gates are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical LCI closeout rows remain valid delivery facts - not rewritten.
- Feature ownership preserved under `docs/project/capabilities/` - no domain-layer LANGCHAIN_INDEPENDENCE plan created.

## Open questions / blocked items

- Finding 01/02: exact trusted-scope injection API shape at compat boundary - deferred to remediation design.
- Finding 05: whether parsing surfaces should become LangChain-free natively or declare explicit transitive opt-in - deferred to packaging remediation.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-LANGCHAIN_INDEPENDENCE-01` … `AUDIT-20260818-LANGCHAIN_INDEPENDENCE-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
