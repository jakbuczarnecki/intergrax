# COMM-5 Flagship Visual & Claim Contract

**Status:** frozen planning contract (COMM-5G1)  
**Branch:** `development`  
**Scope:** public claim boundaries and future visual specifications only - no graphics, no README integration in G1.

---

## 1. Purpose

This document freezes **public truth** and **visual scope** before COMM-5 asset generation (ChatGPT/user session) and documentation integration (later G-stages).

G1 answers:

1. What the Advanced Flagship proof actually proves.
2. What it does **not** prove.
3. Which claims are safe for public documentation.
4. Which claims would overstate the evidence.
5. Which visual stories communicate the strongest differentiators.
6. Where each future visual should appear.
7. How the existing LKW Quick Start / RAG-like visual relates to the governed decision proof.
8. Which documentation pages need later integration.

**Hard rule:** Cursor and this contract do **not** produce final graphics. Four future assets are specified textually only.

---

## 2. Proof source of truth

| Artifact | Role |
|----------|------|
| `proof_infrastructure/governed_hybrid_knowledge_proof/advanced_flagship_proof.py` | Canonical Advanced Flagship runner - eight scenarios, PASS/FAIL gate |
| `proof_infrastructure/governed_hybrid_knowledge_proof/flagship_policy.py` | Versioned policy rules (REV17/REV18), four live obligations |
| `proof_infrastructure/governed_hybrid_knowledge_proof/flagship_docker_scenario.py` | Four-provider Docker scenario, `LIVE_ONLY`, empty indexed retriever |
| `proof_infrastructure/governed_hybrid_knowledge_proof/flagship_docker_environment.py` | Compose lifecycle, vendor restart |
| `proof_infrastructure/governed_hybrid_knowledge_proof/flagship_models.py` | Scenario IDs, typed proof output |
| `proof_infrastructure/governed_hybrid_knowledge_proof/flagship_formatter.py` | Structural history comparison, requirement formatting |
| `proof_infrastructure/governed_hybrid_knowledge_proof/flagship_admin_ports.py` | Proof-only vendor admin (seed, failure injection) |
| `applications/local_workspace_application/docker/docker-compose.governed-hybrid-proof.yml` | Four Docker vendors + Mongo volume `governed_proof_vendor_data` |
| `applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` | Public proof narrative (includes F3-F section) |
| `applications/local_workspace_application/docs/HYBRID_ASK_ARCHITECTURE.md` | Architecture acceptance for F3-F (§6.4.7) |
| `tests/unit/proof_infrastructure/test_advanced_flagship_proof.py` | Unit acceptance of runner contracts |

**Related but distinct proof (not Advanced Flagship):**

| Artifact | Role |
|----------|------|
| `proof_infrastructure/governed_hybrid_knowledge_proof/runner.py` | COMM-5D four-scenario proof - indexed policy + single live Project Status (in-process vendor) |
| `proof_infrastructure/governed_hybrid_knowledge_proof/docker_persistence_proof.py` | F3-E-R1 - Docker Security Status vendor persistence |

**Reproducibility commands (verified against current repo):**

```bash
docker compose -f \
applications/local_workspace_application/docker/docker-compose.governed-hybrid-proof.yml \
up --build -d

uv run python -m \
proof_infrastructure.governed_hybrid_knowledge_proof.advanced_flagship_proof
```

**What the user should expect:** Docker brings up four controlled vendor services and MongoDB; the runner prints a summary table, per-scenario structural sections, distinct provider/connection/capability counts, and `ADVANCED FLAGSHIP: ALL SCENARIOS PASS` or `FAILURES DETECTED`.

**What PASS means:** All eight scenario assertions in `advanced_flagship_proof.py` passed - including four distinct live providers/connections/capabilities on REV17 success, policy-revision temporal flip, authority revoke before HTTP, distinct failure reason codes, vendor restart without reseed, and structural history comparison.

**What PASS does NOT mean:** Production readiness, real-user or commercial validation, real Jira/ServiceNow deployment, production LLM quality, business-correct answers for every domain, complete indexed + authorized live Hybrid Ask certification, or universal multi-vendor coverage beyond the four controlled proof vendors.

---

## 3. What is proven

| Capability / claim | Proof scenario | Evidence | Public wording | Confidence |
|--------------------|----------------|----------|----------------|------------|
| Policy-derived mandatory live evidence obligations | REV17 all satisfied | `flagship_policy.py` → `DeterministicEvidenceObligationDerivation` + 4 rules | Four deployment obligations are derived from versioned policy rules before any provider call | High |
| Four independent live provider integrations | REV17 all satisfied | 4 providers, 4 connections, 4 capabilities, 4 call IDs (`advanced_flagship_proof.py` L67–76) | Four authorized live sources can satisfy four mandatory obligations in one governed Ask run | High |
| Execution-time authority revalidation | Authority revoked | Governance HTTP count = 0, `AUTHORITY_UNAVAILABLE` (`advanced_flagship_proof.py` L155–176) | Live access authority is rechecked at execution; revoked binding can block provider access before HTTP | High |
| Temporal admissibility (max_age) | REV18 stale / REV18 fresh | Same 2h-old security evidence: SATISFIED under REV17 (24h), UNSATISFIED under REV18 (1h) | Policy temporal constraints can invalidate otherwise-present evidence | High |
| Policy revision changes admissibility without app branching | REV17 → REV18 stale | `MutableFlagshipPolicyRulesPort.set_revision`; different `derivation_snapshot_id` | A policy revision can change admissibility without rewriting application execution code | High |
| Distinct failure semantics | Authority / 503 / malformed / stale | `RequirementAdmissibilityReasonCodeV1` per scenario | Authority denial, provider outage, invalid response, and stale evidence are structurally distinct | High |
| Mandatory inadmissibility suppresses LLM | All failure scenarios | `scenario.llm.calls == 0` on UNSATISFIED runs | When mandatory evidence is inadmissible, LLM synthesis is not invoked | High |
| Structural proof survives run reload | REV17 + REV18 stale reload | `scenario.reload_run` preserves admissibility + reason codes | Admissibility outcomes and failure reasons persist in the Ask run record | High |
| External vendor truth in Docker-backed storage | Vendor restart | `environment.restart_project_vendor()`; no reseed; SATISFIED (`advanced_flagship_proof.py` L235–254) | Vendor domain state persists outside the Intergrax process and survives vendor process restart | High |
| Proof runner uses integration abstractions only | All scenarios | `_EmptyIndexedRetriever`; reads via `LiveCapabilityExecutorV1` + TenantConnection; admin via `FlagshipVendorAdminFacadeV1` | Proof coordination does not read vendor storage or vendor data APIs directly | High |
| Admissibility ≠ business approval | REV17 success answer text | Deterministic LLM JSON: structural admissibility wording | Admissibility governs synthesis permission, not whether deployment is approved | High |
| Governed decision is not ordinary RAG | Architecture + runner | Policy obligations + authority + temporal gate + LLM gating on `WorkspaceAskServiceV2` | Intergrax gates answers on structural evidence admissibility, not retrieval alone | High |

---

## 4. What is NOT proven

Explicit non-claims for public documentation:

| Not proven | Why |
|------------|-----|
| Full production / enterprise readiness | Proof is a controlled developer scenario; README and PROOFS already gate this |
| Real-user validation | No user study evidence |
| Commercial validation | No commercial deployment evidence |
| **Complete indexed + authorized live Hybrid Ask certification** | Advanced Flagship uses `QueryPolicyModeV2.LIVE_ONLY` with `_EmptyIndexedRetriever()` - no indexed retrieval in this proof |
| Real Jira / ServiceNow / enterprise vendor deployment | Controlled Docker vendors are proof systems with Mongo-backed fixtures |
| Production model quality | Deterministic proof LLM (`proof/flagship-deterministic`) |
| Business truth for every domain | Proof demonstrates governance mechanics, not domain correctness |
| All Hybrid Ask modes in one flagship run | `hybrid` mode with indexed + live participation together is not exercised in F3-F |
| Penetration / security certification | Adversarial tests are architectural invariants, not red-team certification |
| Universal vendor interoperability | Only four named controlled providers in compose |
| Intergrax reads Mongo as evidence | Mongo is vendor implementation detail; Intergrax reads vendor HTTP through integrations |

**Indexed + live boundary (authoritative wording for later integration):**

> Advanced Flagship proves a **multi-provider LIVE_ONLY governed decision** with policy-derived obligations, execution-time authority, temporal admissibility, exact failure semantics, and Docker-backed external vendor truth. It does **not** certify the complete **Hybrid Ask** path that combines indexed retrieval and authorized live evidence in a single admissibility gate.

This aligns with `README.md` (Hybrid Ask combining indexed and authorized live evidence is **not yet proven**) and `docs/project/proofs/PROOFS.md`.

---

## 5. Product Quick Start vs Advanced Flagship

| Dimension | **A. Product Quick Start** | **B. Advanced Governed Flagship** |
|-----------|---------------------------|-----------------------------------|
| Purpose | Easiest runnable product evaluation | Platform differentiation / governed decision proof |
| Path | One-command indexed Ask V1 | Docker compose + `advanced_flagship_proof` |
| Knowledge mode | Indexed sample document (`AURORA-17`) | `LIVE_ONLY` - four live vendor reads |
| Policy / obligations | Product upload + indexing | Versioned deployment policy rules (REV17/REV18) |
| Providers | Local RAG stack (Ollama/Qdrant path) | Four independent Docker vendor integrations |
| Authority / temporal | Not the focus | Runtime authority + max_age / valid_at |
| Failure semantics | Product troubleshooting scope | AUTHORITY_UNAVAILABLE, PROVIDER_FAILED, PROVIDER_RESPONSE_INVALID, EVIDENCE_TEMPORALLY_INVALID |
| LLM gating | Grounded answer on indexed hit | LLM only when overall admissibility SATISFIED |
| Persistence proof | Ask run + source citation | Structural admissibility + reason codes + vendor restart |
| Visual | *(removed)* `lkw-grounded-result-{light,dark}.svg` | `lkw-governed-evidence-gate-{light,dark}.png` in `applications/local_workspace_application/docs/assets/` |
| Public placement | README “Try LKW”, QUICKSTART | README “what is proven”, PROOFS, GOVERNED_HYBRID_KNOWLEDGE_PROOF |
| Ownership | **LKW reference product** | **LKW reference application** assets; **Intergrax platform** mechanisms via LKW `WorkspaceAskServiceV2` |

**Critical rule:** Do not conflate A and B. The Quick Start visual must remain labeled as onboarding / indexed Ask - not as the full Intergrax governed-evidence story.

---

## 6. Public claim hierarchy

### LEVEL 1 - 10-second claim

Intergrax decides whether enough **authorized, valid evidence** exists before an LLM may answer - and records why.

### LEVEL 2 - CTO / architect claim

Intergrax provides reusable platform contracts for **evidence obligations**, **execution-time authority**, **temporal admissibility**, and **structural proof**. A governed Ask run derives mandatory requirements from versioned policy, resolves them through authorized live integrations, blocks synthesis on any mandatory failure with explicit reason codes, and persists reviewable outcomes - demonstrated in the Advanced Flagship proof with four independent Docker-backed vendors.

### LEVEL 3 - technical proof claim

The Advanced Flagship proof (`advanced_flagship_proof.py`) exercises `WorkspaceAskServiceV2` in `LIVE_ONLY` mode with four policy-derived live obligations (Project Status, Security Status, Change Approval, Governance Approval), `WorkspaceLiveAccessRuntimeAuthority`, temporal constraints (`MaxAgeTemporalConstraintV1`, `ValidAtTemporalConstraintV1`), deterministic admissibility evaluation, LLM suppression on `EvidenceAdmissibilityStatusV1.UNSATISFIED`, persisted `RequirementAdmissibilityReasonCodeV1` values across reload, and vendor restart without reseed over Docker-backed external storage - without the proof runner accessing vendor Mongo or raw vendor APIs directly.

### Core thesis (frozen wording - evidence-backed)

**Long technical version:**

Intergrax does not merely give an LLM access to knowledge and tools. For governed Hybrid Ask paths, it derives mandatory evidence obligations from versioned policy rules, resolves live obligations through tenant connections only after runtime authority approval, evaluates temporal constraints on returned evidence, treats authority denial, provider failure, invalid provider contracts, and temporal invalidity as distinct admissibility outcomes, suppresses LLM synthesis when any mandatory obligation is unsatisfied, and persists structural proof - policy basis, derivation snapshot, requirement evaluations, and reason codes - so reviewers can reload a run and see why synthesis was allowed or refused. The Advanced Flagship proof demonstrates this end-to-end for four independent live providers with Docker-backed vendor truth.

**Short CTO version:**

Intergrax gates answers on admissible evidence - not model enthusiasm. Policy defines what proof is required; authority and temporal rules decide if live evidence counts; the LLM runs only when the platform allows it; every refusal is structurally recorded.

**One-sentence README version:**

Intergrax determines what evidence is required, whether authorized sources satisfy it at execution time, and preserves structural proof of why an answer was allowed or refused - before the LLM synthesizes.

*(Qualification for Advanced Flagship specifically: demonstrated on a four-provider LIVE_ONLY proof path; complete indexed + live Hybrid Ask remains a separate, not-yet-certified boundary.)*

---

## 7. Current claim audit

| CURRENT CLAIM | SOURCE | STATUS | RATIONALE |
|---------------|--------|--------|-----------|
| Hybrid Ask combining indexed and authorized live evidence is **not yet proven** | `README.md` L179 | **KEEP** | Accurate; F3-F is LIVE_ONLY |
| Product Quick Start = indexed Ask V1, not Hybrid certification | `README.md` L173–175 | **KEEP** | Correct separation from flagship |
| LKW visual = Quick Start, not finished UI | `README.md` L165–166 | **KEEP** | Accurate caption |
| Intergrax = governed foundation for specialized apps | `README.md` L6–8 | **KEEP** | Platform framing; not over-specific |
| Governed Hybrid Knowledge Proof resolves through **authorized indexed and live sources** | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` L3 | **QUALIFY** | True for COMM-5D harness broadly; Advanced Flagship subsection is LIVE_ONLY - lead with admissibility thesis, qualify indexed for F3-F |
| Mixed indexed + authorized live Hybrid Ask remains incomplete | `PROOFS.md`, `WHY_INTERGRAX.md` | **KEEP** | Authoritative boundary |
| No dedicated accepted public Governed Execution proof yet | `PUBLIC_DOCUMENTATION_MAP.md` L125 | **QUALIFY** | Advanced Flagship is bounded Hybrid Ask / evidence admissibility proof - later integrate as **Governed Evidence Decision Proof**, not full Governed Execution domain proof |
| Advanced flagship = four-provider Docker governed decision | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` §Advanced flagship | **STRENGTHEN** | Add to PROOFS.md and README “proven today” in later G-stage |
| Adversarial invariants verified on COMM-5D harness | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` §Adversarial | **KEEP** | Scoped to named harness; not flagship-only |
| Mongo never direct Intergrax evidence source | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` L323 | **KEEP** | Matches architecture |
| `hybrid` mode requires indexed + live participation (architecture) | `HYBRID_ASK_ARCHITECTURE.md` §3.3 | **KEEP** | Architecture target; distinguish from F3-F proof mode |
| LKW = Backend Product Alpha / MVP | `README.md`, `PROOFS.md` | **KEEP** | Accurate status |
| “Eliminates hallucinations” / “production-ready” | - | **REMOVE** if ever introduced | Forbidden unsupported wording (§17) |

---

## 8. Differentiation claims audit

| # | Claim | Status | Supporting scenario | Safe public wording | Dangerous overstated wording |
|---|-------|--------|---------------------|---------------------|------------------------------|
| 1 | Obligations derived from versioned policy rules | **PROVEN** | REV17 - four `RequireLiveEvidencePolicyRuleV1` in `flagship_policy.py` | Mandatory live obligations are derived from versioned policy rules | “Any policy language auto-compiles to obligations” |
| 2 | Policy revision changes admissibility without app code change | **PROVEN** | REV17 → REV18 stale - `set_policy_revision` only | Policy revision can change admissibility without application graph rewrite | “Zero configuration delta ever” |
| 3 | Same requirement ID survives revision; snapshot/basis change | **PROVEN** | `build_history_comparison` - same `:security` requirement_id, different snapshot | Same logical requirement can persist while policy basis and derivation snapshot change | “Identical run IDs across revisions” |
| 4 | Four independent providers satisfy four obligations | **PROVEN** | REV17 - len(providers)==4 assertion | Four independent integrations can satisfy four mandatory obligations | “Every Ask uses four providers” |
| 5 | Authority rechecked at execution time | **PROVEN** | `WorkspaceLiveAccessRuntimeAuthority` on every live call | Authority is revalidated at execution time | “Continuous real-time IAM sync” |
| 6 | Revoked authority stops access before HTTP | **PROVEN** | Authority revoked - governance HTTP = 0 | Revoked live binding can block provider access before HTTP occurs | “Impossible to call providers without approval” (global) |
| 7 | Provider failure ≠ authority failure | **PROVEN** | Authority revoked (HTTP 0) vs provider 503 (HTTP 1) | Structural reason codes distinguish authority and provider outcomes | “All failures are permission errors” |
| 8 | Malformed response ≠ provider outage | **PROVEN** | Malformed vs 503 - `PROVIDER_RESPONSE_INVALID` vs `PROVIDER_FAILED` | Invalid provider contract is distinct from provider outage | “HTTP 200 always means valid evidence” |
| 9 | Stale evidence ≠ unavailable evidence | **PROVEN** | REV18 stale - evidence present, `EVIDENCE_TEMPORALLY_INVALID` | Temporal invalidity is distinct from missing or failed evidence | “Fresh data always means approved deployment” |
| 10 | Mandatory inadmissibility suppresses LLM | **PROVEN** | All UNSATISFIED scenarios - `llm.calls == 0` | LLM synthesis is suppressed when mandatory admissibility fails | “LLM never runs” (unconditional) |
| 11 | Failure/admissibility reasons survive reload | **PROVEN** | REV17 + REV18 stale reload checks | Structural reasons survive Ask run reload | “Immutable blockchain audit” |
| 12 | External vendor truth outside Intergrax process | **PROVEN** | Docker vendors + Mongo volume | Vendor domain state lives in external Docker-backed storage | “Customer VPC deployment proven” |
| 13 | Vendor restart does not destroy persisted truth | **PROVEN** | Vendor restart scenario - no reseed | Vendor restart does not destroy persisted vendor records | “Survives all disasters” |
| 14 | Proof runner does not access vendor storage directly | **PROVEN** | Architecture + admin ports only for seed/control | Proof runner coordinates through abstractions, not vendor storage | - |
| 15 | Proof runner does not call vendor data APIs directly | **PROVEN** | Reads via `LiveCapabilityExecutorV1` / integrations | Vendor reads go through Intergrax integration abstractions | - |
| 16 | Intergrax does not use Mongo as evidence source | **PROVEN** | Compose + GOVERNED doc boundary | Intergrax consumes vendor HTTP integrations, not vendor Mongo | “Mongo-free architecture” |
| 17 | Flagship > ordinary RAG / tool calling | **PROVEN** (bounded) | Full admissibility gate + policy + authority + temporal + proof | Governed admissibility gate exceeds retrieve-then-answer RAG | “RAG is useless” / “LangGraph cannot do this” |
| 18 | Complete Hybrid Ask indexed + live | **NOT PROVEN** | F3-F uses `_EmptyIndexedRetriever`, `LIVE_ONLY` | Complete indexed + authorized live Hybrid Ask certification remains incomplete | “Full hybrid indexed + live is proven” |

---

## 9. Visual Story Pack

Four assets - light/dark pairs in `applications/local_workspace_application/docs/assets/`.

> **Post-G1 ownership refinement:** the Advanced Flagship visuals are LKW reference
> application assets, while Intergrax remains the underlying platform mechanism.
> Global Intergrax README assets remain in `docs/project/assets/public/readme/`.

---

### Visual 1 - Governed Decision Hero

**Working filenames:**

- `applications/local_workspace_application/docs/assets/lkw-governed-evidence-gate-light.png`
- `applications/local_workspace_application/docs/assets/lkw-governed-evidence-gate-dark.png`

| Field | Specification |
|-------|---------------|
| **Purpose** | Replace the impression that LKW == ordinary RAG; introduce platform evidence gate |
| **Audience** | CTO, architect, technical reviewer landing on README |
| **Primary message** | The model does not decide whether enough evidence exists. The platform does. |
| **Required objects** | Question → versioned policy → four obligation boxes → four qualified live sources → execution-time authority check → temporal validity check → admissibility gate → ALLOW / REFUSE branches → LLM only on ALLOW → structural proof record |
| **Forbidden implications** | All queries use four providers; indexed+live fully certified; every deployment uses this topology; production readiness |
| **Source claims** | §3 table; thesis §6; REV17 success + failure scenarios |
| **Target docs** | `README.md` LKW section (after ecosystem/platform), `WHY_INTERGRAX.md`, `PROOFS.md` |
| **Recommended README placement** | After “What is boundedly proven today”, before “Try LKW” / Quick Start - see §16 |

**Visual hierarchy:** Top - question; middle-left - policy + obligations; middle-right - four sources; center - admissibility gate (largest element); bottom - proof record fan-in.

**Required labels:** Policy REV; Obligation; Authority OK/DENIED; Temporal OK/STALE; Admissibility SATISFIED/UNSATISFIED; LLM 0/1; Structural proof.

**Optional labels:** Provider names (Project Status, Security Status, Change Approval, Governance Approval); connection refs.

**Labels to avoid:** “Production ready”; “Zero hallucination”; “Full hybrid certified”; “Jira/ServiceNow”.

---

### Visual 2 - Policy Revision Admissibility

**Working filenames:**

- `applications/local_workspace_application/docs/assets/lkw-policy-revision-admissibility-light.png`
- `applications/local_workspace_application/docs/assets/lkw-policy-revision-admissibility-dark.png`

| Field | Specification |
|-------|---------------|
| **Purpose** | Show policy-driven temporal admissibility flip without code change |
| **Audience** | Architect, security/governance reviewer |
| **Core message** | Same application execution topology + same vendor evidence → different admissibility when policy revision tightens max_age |
| **Required panels** | REV17: security max_age 24h, 2h-old evidence → SATISFIED → LLM 1 · REV18: security max_age 1h, same 2h-old evidence → TEMPORALLY INVALID → LLM 0 |
| **Precise annotation** | Note different `derivation_snapshot_id` / policy basis; same `:security` requirement_id - not “same run ID” |
| **Forbidden implications** | Application redeploy required; evidence changed between panels |
| **Source claims** | `advanced_flagship_proof.py` REV17/REV18 stale; `build_history_comparison` |
| **Target docs** | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md`, `HYBRID_ASK_ARCHITECTURE.md` §6.4.5–6.4.7, future README callout |

---

### Visual 3 - Failure Semantics

**Working filenames:**

- `applications/local_workspace_application/docs/assets/lkw-evidence-failure-semantics-light.png`
- `applications/local_workspace_application/docs/assets/lkw-evidence-failure-semantics-dark.png`

| Field | Specification |
|-------|---------------|
| **Purpose** | Communicate distinct refusal reasons |
| **Audience** | SRE, architect, compliance-oriented reviewer |
| **Core message** | No answer is not a single failure state - Intergrax retains why synthesis was suppressed |
| **Four columns** | (1) AUTHORITY UNAVAILABLE - HTTP 0 · (2) PROVIDER FAILED - HTTP 1 / 503 · (3) PROVIDER RESPONSE INVALID - HTTP 200 malformed · (4) EVIDENCE TEMPORALLY INVALID - HTTP success, time fail |
| **Shared footer** | Mandatory obligation UNSATISFIED; LLM 0; persisted reason code |
| **Source claims** | Scenarios: authority revoked, provider 503, malformed, REV18 stale |
| **Target docs** | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md`, `HYBRID_ASK_ARCHITECTURE.md` §6.4.6 |

---

### Visual 4 - External Truth / Four Vendors

**Working filenames:**

- `applications/local_workspace_application/docs/assets/lkw-external-evidence-authority-light.png`
- `applications/local_workspace_application/docs/assets/lkw-external-evidence-authority-dark.png`

| Field | Specification |
|-------|---------------|
| **Purpose** | Show external persistence boundary and four-vendor topology |
| **Audience** | Platform engineer, technical reviewer |
| **Core message** | Governed Intergrax execution reaches four vendor services through connections/integrations; vendor state persists externally |
| **Required objects** | Intergrax governed execution → 4 connections → 4 provider integrations → 4 vendor services (Project Status, Security Status, Change Approval, Governance Approval) → external persisted store per vendor |
| **Annotations** | “Intergrax never reads vendor storage directly.” · Proof runner → abstractions · NOT proof runner → Mongo/raw vendor data API |
| **Forbidden implications** | Mongo as product claim; real enterprise vendor logos as production claims |
| **Source claims** | `docker-compose.governed-hybrid-proof.yml`; vendor restart scenario |
| **Target docs** | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` Docker section, `HYBRID_ASK_ARCHITECTURE.md` provider table |

---

## 10. Visual style contract

Audited family: `applications/local_workspace_application/docs/assets/` (PNG, light/dark pairs). Global Intergrax README assets: `docs/project/assets/public/readme/`.

| Characteristic | Observable convention |
|----------------|----------------------|
| **Format** | PNG for Intergrax readme heroes; `-light` / `-dark` suffix pairing |
| **Aspect ratio** | Wide landscape (~16:9), full-width README `<picture>` usage |
| **Brand** | Intergrax wordmark top-left on hero assets |
| **Title hierarchy** | Large centered H1-style title; one-line subtitle in muted body color |
| **Containers** | Rounded rectangles with 2–3px colored borders; soft fill tints by semantic role |
| **Color roles** | Purple - agent/governance; blue - capabilities/knowledge; teal - evidence/audit; green - success/outcome; orange - human approval / warning |
| **Flow** | Numbered steps (1–6) for sequences; solid arrows primary, dashed for proposed/secondary |
| **Density** | High information density with bottom legend explaining arrow types |
| **Typography** | Sans-serif; bold colored step numbers; short labels (not paragraphs inside diagram) |
| **Spacing** | Generous outer margin; grouped columns for parallel concepts |
| **Icons** | Simple line icons in rounded squares - not photorealistic |
| **Naming** | `intergrax-{topic}-{light|dark}.png` - lowercase kebab-case |
| **LKW exception** | Product Quick Start uses SVG under `applications/local_workspace_application/docs/assets/` - do not mix SVG/PNG families in one visual story without intentional distinction |

Future ChatGPT-generated assets should match this family for README cohesion.

---

## 11. Documentation target map

| Document | Current role | Needed change (later G-stage) | Visual | Priority |
|----------|--------------|-------------------------------|--------|----------|
| `README.md` | First contact; LKW Quick Start + AURORA visual; hybrid not proven | Insert governed hero + “proven today” hierarchy; link Advanced Flagship proof; keep Quick Start separate | Visual 1 (+ optional 2 callout) | P0 |
| `applications/.../LKW_PRODUCT_TOUR.md` | Product story | Add pointer: platform governed proof ≠ product quick start | - | P2 |
| `applications/.../QUICKSTART.md` | Indexed executable path | Cross-link to governed proof doc; no flagship conflation | Keep existing flow only | P2 |
| `applications/.../LKW_PLATFORM_PROOF.md` | LKW bounded proofs | Add row distinguishing LKW product proofs vs Intergrax governed flagship | - | P1 |
| `applications/.../HYBRID_ASK_ARCHITECTURE.md` | Architecture acceptance | Embed Visuals 2–4 in §6.4.7 (replace or supplement mermaid) | 2, 3, 4 | P1 |
| `applications/.../GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` | Public proof narrative | Qualify opening thesis for F3-F LIVE_ONLY; embed visuals | 1–4 | P0 |
| `docs/project/proofs/PROOFS.md` | Public proof dashboard | Add **Governed Evidence Decision Proof** row; bounded LIVE_ONLY scope | Link to Visual 1 | P0 |
| `docs/project/overview/WHY_INTERGRAX.md` | Strategic thesis | One paragraph + link; no overclaim indexed+live | Visual 1 (optional) | P1 |
| `docs/project/community/PUBLIC_DOCUMENTATION_MAP.md` | Routing | Add governed flagship proof route under platform evidence | - | P2 |

**G1 action:** none of the above edited in G1 except this contract file.

---

## 12. README future placement plan

**Current README order (observed):**

1. Ecosystem hero  
2. Choose your path  
3. Platform map  
4. LKW section → **What is proven today** (Product Quick Start + Governed Evidence Decision Proof) → governed evidence gate visual → Try LKW

**Recommended future hierarchy:**

1. Ecosystem hero *(unchanged)*
2. Choose your path *(unchanged)*
3. Platform map *(unchanged)*
4. **LKW product positioning** *(short - unchanged intent)*
5. **What is proven today** *(Product Quick Start + Governed Evidence Decision Proof as separate bullets)*
6. **Governed decision hero (Visual 1)** - platform differentiation in Governed Evidence section
7. **Try LKW** commands *(unchanged)*
8. Deeper doc routes *(PROOFS, GOVERNED_HYBRID_KNOWLEDGE_PROOF)*

**Consistency:** README separates Quick Start from Governed Evidence Proof textually and visually. The legacy Quick Start SVG (`lkw-grounded-result`) is removed; Product Quick Start remains documented without a dedicated diagram.

---

## 13. Claim language - avoid marketing theater

**Forbidden unsupported wording:**

- guarantees correct answers  
- eliminates hallucinations  
- production-ready / enterprise-ready  
- fully autonomous governance  
- complete zero-trust AI  
- works with any vendor  
- proves real-world enterprise deployment  
- replaces LangGraph / LangGraph cannot do this  

**Safe positioning vs workflow frameworks:**

A workflow framework can orchestrate similar flows. Intergrax differentiation is **reusable platform contracts** for evidence requirements, execution-time authority, temporal admissibility, LLM gating, and structural proof - demonstrated in bounded proof runners, not universal superiority claims.

---

## 14. Intergrax vs LKW ownership

| Claim | Platform capability? | LKW executable proof? | Public placement |
|-------|---------------------|-------------------------|------------------|
| Evidence obligation derivation | Yes - `intergrax/runtime/evidence/` | Via LKW `WorkspaceAskServiceV2` in proof | Architecture + GOVERNED proof doc |
| Runtime live access authority | Yes - platform + LKW wiring | Advanced Flagship | HYBRID_ASK_ARCHITECTURE, proof doc |
| Temporal admissibility | Yes | Advanced Flagship | Proof doc, Visual 2 |
| Indexed Ask V1 product path | Yes (RAG stack) | **Product Quick Start** | README, QUICKSTART |
| AURORA-17 onboarding proof | Product feature | **LKW only** | README Quick Start visual |
| Four-provider LIVE_ONLY flagship | Platform mechanism | Proof runner (LKW app stack) | PROOFS, README “proven today” |
| Slack DM Ask | LKW product direction | Bounded LKW proof | LKW docs |
| Docker controlled vendors | Proof infrastructure | Not a product feature | Proof doc only |
| Complete Hybrid Ask hybrid mode | Platform architecture target | **Not proven** | PROOFS boundary |

**Rule:** Lead with **Intergrax platform** for admissibility/governance claims; cite **LKW** as the reference application and proof harness, not as if every platform domain is productized in LKW 1.0.

---

## 15. Advanced proof naming

| Label | Recommendation |
|-------|----------------|
| **Public-facing name** | **Governed Evidence Decision Proof** |
| **Technical / internal name** | Advanced Flagship Proof / Governed Hybrid Knowledge Proof (F3-F) |
| **Short label** | Governed Evidence Proof |

Rationale: describes what is differentiated (evidence admissibility gating + structural decision proof), avoids implying full hybrid indexed+live certification, and distinguishes from Product Quick Start.

**No code or file renames in G1.**

---

## 16. Evidence matrix

| Scenario | Policy | Vendor evidence | Authority | HTTP (security/governance) | Admissibility | LLM |
|----------|--------|-----------------|-----------|----------------------------|---------------|-----|
| REV17 valid | REV17 (max_age 24h) | All four domains valid; security 2h old | All bindings ACTIVE | 4 live calls | SATISFIED | 1 |
| REV18 stale | REV18 (max_age 1h) | Same as REV17 (security still 2h old) | ACTIVE | 4 live calls | UNSATISFIED (security TEMPORALLY INVALID) | 0 |
| REV18 fresh | REV18 | Security refreshed to 30m old | ACTIVE | 4 live calls | SATISFIED | 1 |
| Authority revoked | REV17 | Valid baseline | Governance binding DISABLED at execution | governance 0; others may partial | UNSATISFIED (AUTHORITY_UNAVAILABLE) | 0 |
| Provider 503 | REV17 | Valid baseline | ACTIVE | security 1 (503) | UNSATISFIED (PROVIDER_FAILED) | 0 |
| Malformed response | REV17 | Valid baseline | ACTIVE | security 1 (200 invalid) | UNSATISFIED (PROVIDER_RESPONSE_INVALID) | 0 |
| Vendor restart | REV17 | Persisted external records; no reseed | ACTIVE | 4 live calls | SATISFIED | 1 |

*HTTP counts reflect proof assertions on security/governance scenarios; REV17 success implies four distinct successful live calls.*

---

## 17. Source traceability

| Public claim | Source file / scenario |
|--------------|------------------------|
| Four policy-derived obligations | `flagship_policy.py` - `build_flagship_deployment_policy_rules` |
| REV17 all satisfied, LLM 1 | `advanced_flagship_proof.py` - `flagship-rev17-valid` |
| Policy revision changes max_age | `flagship_policy.py` - `_REV17_SECURITY_MAX_AGE_SECONDS` vs `_REV18_SECURITY_MAX_AGE_SECONDS` |
| Same requirement_id, different snapshot | `flagship_formatter.py` - `build_history_comparison` |
| REV18 stale - TEMPORALLY INVALID | `advanced_flagship_proof.py` - `flagship-rev18-stale` |
| Authority revoke → HTTP 0 | `advanced_flagship_proof.py` - `flagship-authority-revoked`; `FlagshipControllableOrchestratorV1` |
| Provider 503 - PROVIDER_FAILED | `advanced_flagship_proof.py` - `flagship-provider-503`; `SecurityStatusReadBehaviorV1.HTTP_503` |
| Malformed - PROVIDER_RESPONSE_INVALID | `advanced_flagship_proof.py` - `flagship-malformed` |
| LLM suppressed on UNSATISFIED | All failure scenarios - `scenario.llm.calls == 0` |
| Reload preserves reasons | `advanced_flagship_proof.py` - `reload_run` checks |
| LIVE_ONLY - no indexed retrieval | `flagship_docker_scenario.py` - `_EmptyIndexedRetriever`, `QueryPolicyModeV2.LIVE_ONLY` |
| Vendor restart persistence | `advanced_flagship_proof.py` - `flagship-vendor-restart`; `flagship_docker_environment.restart_project_vendor` |
| Compose + Mongo volume | `docker-compose.governed-hybrid-proof.yml` - `governed_proof_vendor_data` |
| Proof runner boundary | `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md` §Docker vendor persistence; scenario wiring |
| Indexed + live not certified (flagship) | `flagship_docker_scenario.py` - `indexed_sources=()`, empty retriever |
| COMM-5D indexed + live (separate proof) | `proof_infrastructure/governed_hybrid_knowledge_proof/runner.py` |
| Hybrid not proven (public) | `README.md`, `docs/project/proofs/PROOFS.md` |

---

## 18. Acceptance rule (G1)

**TASK COMPLETES WITH:**

- 0 generated PNG  
- 0 generated SVG  
- 0 new rendered diagrams  
- 0 README changes  
- 0 asset changes  
- 0 production code changes  

Text-only documentation contract in this file.

---

## 19. Document metadata

| Field | Value |
|-------|-------|
| Task | COMM-5G1 |
| Canonical path | `docs/project/proofs/COMM_5_FLAGSHIP_VISUAL_AND_CLAIM_CONTRACT.md` |
| Supersedes | Informal audit notes under `docs/audit_results/` for public integration planning |
| Next stages | G2+ asset generation (external) · G3+ doc/README integration per §11 |
