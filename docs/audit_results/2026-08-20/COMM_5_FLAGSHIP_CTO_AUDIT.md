# COMM-5 Flagship Skeptical CTO Audit

**Campaign:** `2026-08-20` (flagship proof acceptance; not a Protocol v2 layer audit)  
**Auditor stance:** skeptical CTO / principal architect / AI platform architect / senior technical buyer  
**Mode:** read-only except this campaign; no implementation, no README rewrite, no marketing polish  
**Audited code SHA:** `654a7c0e3fe823a43a2620645848248023e1c64e`  
**Development at audit publication:** later docs-only commits may sit on `development`; flagship proof/production Ask V2 paths were unchanged through `65aaf33a6a6dba9b336162ec547cd677f4edad91`  
**Evidence executed this session:** `uv run python -m proof_infrastructure.governed_hybrid_knowledge_proof` → `4 / 4 PROOFS PASSED`; `uv run pytest tests/unit/proof_infrastructure/test_governed_hybrid_knowledge_proof.py tests/unit/proof_infrastructure/test_governed_hybrid_knowledge_adversarial.py` → `26 passed`

Assumption held throughout: RAG, tool calling, policy engines, provenance, approvals, access control, and observability already exist elsewhere. Complexity is not credit.

---

## Executive verdict

**Verdict: B — DIFFERENTIATED COMPOSITION**

The COMM-5 proof demonstrates a **real, enforced composition** on the production Workspace Ask V2 path: typed evidence obligations, indexed retrieval plus live HTTP, configuration reload before provider execution, admissibility evaluation **before** any synthesizer call, and durable **structural** run records.

It does **not** demonstrate unique primitives, unique technology, hallucination prevention, enterprise certification, or historically replayable facts.

A competent team can assemble RAG + tool calling + an `if missing: don't call the model` guard in days. What is harder to copy quickly is the **cross-layer contract**: plan ≠ obligations, caller cannot silently drop mandatory obligations, live evidence is identified by call identity (not “some JSON arrived”), authority is re-read at execution rather than trusted from plan time, and a missing required live result is a typed `INSUFFICIENT_EVIDENCE` run rather than a prompt suggestion.

That is a platform-shaped capability. It is not a moat, and it is not yet the LKW first-screen product story.

**Overall score: 72 / 100**

---

## What the proof actually demonstrates

Thesis under audit:

> Intergrax does not merely give an LLM access to knowledge and tools. It determines what evidence is required for an answer to be admissible, resolves that evidence through currently authorized sources, revalidates authority at execution time, and preserves structural proof of why the result was valid.

| Clause | Result | What is actually proven |
|--------|--------|-------------------------|
| A — evidence requirements | **PROVEN with qualification** | Typed compose of product + provider obligations; caller may only add; plan proposals are not obligations. Live-required semantics in this proof come from a **proof-local** provider strategy, not from default LKW serving. |
| B — indexed + live | **PROVEN as composition** | Real ingest → `local.workspace.search` → real HTTP Project Status; one run uses both. Embeddings are hash-based proof doubles. Stronger invariant is obligation identity, not “RAG + API”. |
| C — execution-time authority | **PROVEN (strongest clause)** | Canonical disable after indexed retrieval; `WorkspaceLiveAccessRuntimeAuthority.is_usable` reloads configuration before `handler.execute`; HTTP reads = 0. Optional executor port is a wiring risk; production serving injects the authority when live catalog is present. |
| D — admissibility before synthesis | **PROVEN when obligations are on the plan** | `evaluate_execution_admissibility` runs before `HybridAskAnswerAssemblerV2`. LLM cannot override that gate. Generic HYBRID without live obligations can still reach the LLM, then fail citation checks — not the flagship path. |
| E — history / structural proof | **PROVEN as structural identity, not factual replay** | `get_run` retains run id, revision, obligations, indexed provenance, live hash/binding/call, admissibility, answer/status. EPHEMERAL forbids durable live body and receipts. Scenario 04 does not reconstruct that `SEC-417` was OPEN. |

Prompt-only reproduction of the **flagship path** fails at the contracts below, not at model instructions.

---

## What is commodity

These are table stakes. Do not sell them as Intergrax-specific:

1. **Vector / indexed RAG** — ingest, chunk, search, cite. Hash embeddings in the proof are weaker than production semantic RAG.
2. **Live HTTP / tool calling** — an authorized GET of project status is a normal integration.
3. **Policy configuration** — HYBRID mode, allowed connections/capabilities, budgets, retention enum.
4. **Workspace / tenant lookup** — `workspace_not_found` is basic ACL, not a knowledge-governance thesis.
5. **Deterministic fixture synthesis** — YES/NO in 01/02 is a local rule engine (`DeploymentReadinessDeterministicLLM`), not a general LLM behavior proof.

---

## Where differentiation really exists

Three invariants are stronger than “good integration”:

1. **Evidence obligations are a first-class, fail-closed contract**, distinct from the execution plan. `EvidencePlanV1` carries both `ordered_live_call_proposals` and `required_evidence_obligations`. Product HYBRID only auto-requires indexed grounding (`product:hybrid:indexed`). Per-call live evidence must be attached by provider/product planning (`ProviderEvidencePlanV1`) or additive caller obligations. `compose_evidence_obligations` is append-only; duplicate `requirement_id` raises. `WorkspaceAskCommandV2` rejects `INDEXED_ONLY` with `provider_request`.

2. **Plan-time authorization is not execution-time permission.** `LiveCapabilityExecutorV1` calls `runtime_authority.is_usable(...)` **before** handler resolution and HTTP. `WorkspaceLiveAccessRuntimeAuthority` reloads workspace configuration, binding ACTIVE status, connection administrative status, attachment, and capability catalog. COMM-5 scenario 03 + attack B show canonical `LiveAccessLifecycleService.disable` after indexed retrieval → Project Status `read_request_count = 0`.

3. **Admissibility is evaluated on acquired evidence identity, then synthesis is skipped.** Live requirements match `call_id`, not “any live blob”. Indexed evidence cannot satisfy a live obligation. Malformed/404/503 HTTP that yields no valid live items leaves the obligation `NO_MATCHING_EVIDENCE`. Authority denial (HTTP 0) and provider failure (HTTP 1) are distinguishable; both finalize a typed run with `answer = None` when the live obligation is required.

What is **not** differentiation: the ORION story, mermaid diagrams, four PASS banners, or the size of the harness.

**Composition vs novelty:** the composition is the product. Treat it as a **governed hybrid ask runtime**, not as a new retrieval algorithm.

---

## Four-scenario audit

### 01 REALITY — OPEN blocker → NO

| | |
|--|--|
| **What it proves** | With SATISFIED indexed + live evidence, a synthesizer that applies the indexed policy to current live status can answer NO. HTTP reads = 1. Live body is not stored as durable evidence (`PersistedLiveEvidenceProvenanceV2`). |
| **What it does not prove** | That a hosted LLM would apply the policy correctly; that semantic RAG found the policy (hash embeddings); that this is the Slack/HTTP product path; that answers are “true”. |
| **CTO interest** | Medium. Shows hybrid inputs actually reach one decision. The interesting part is the gate around it, not YES/NO. |
| **Strength** | **3 / 5** |

### 02 FRESHNESS — only external state changes → YES

| | |
|--|--|
| **What it proves** | Same question, same configuration revision, same indexed policy; control API closes `SEC-417`; next Ask reads live HTTP again and answers YES. Indexed policy did not change. |
| **What it does not prove** | Cache-busting in production CDNs; clock/bitemporal query; that “same plan contract” is more than the same question string (`same_plan_policy_contract` in the runner compares `run_2.question`). |
| **CTO interest** | Medium-high for “live state can flip the answer without re-indexing policy”. |
| **Strength** | **3 / 5** |

### 03 AUTHORITY — revoke after planning → HTTP 0 / LLM 0

| | |
|--|--|
| **What it proves** | Binding ACTIVE at plan/index time is insufficient. Canonical disable increments `configuration_revision`. Runtime authority denies. Provider is not invoked. Admissibility UNSATISFIED. `INSUFFICIENT_EVIDENCE`. Indexed evidence remains and does not substitute. |
| **What it does not prove** | Distributed race after `is_usable` returns true; multi-process config propagation; that every production constructor injects authority (executor port is optional; serving does inject when live catalog exists). Timing of revoke is a proof wrapper (`_RevokeAfterIndexedRetriever`), not a production scheduler. |
| **CTO interest** | **High.** This is the clause a buyer cannot get from “the prompt says to check permissions”. |
| **Strength** | **5 / 5** |

### 04 HISTORY — retrieve prior run after live world changed

| | |
|--|--|
| **What it proves** | Ask #1 remains COMPLETED / NO with obligations, indexed provenance, live `content_hash` / `call_id` / `live_access_binding_id`, admissibility SATISFIED, `configuration_revision`. Later CLOSED blocker does not mutate the stored run (also attack M). |
| **What it does not prove** | Reconstruction of the OPEN blocker payload; signed integrity; crash-recovery durability (in-memory `InMemoryDocumentStore`); “why NO was valid” in a human-explainable factual sense. EPHEMERAL live bodies are gone by contract. |
| **CTO interest** | Medium if positioned as **audit identity**; low if positioned as **historical replay**. |
| **Strength** | **3 / 5** as structural proof; **1 / 5** as factual reconstruction |

---

## Adversarial audit

Do not treat `test_adversarial_attack_matrix_all_pass` as evidence. It constructs `AdversarialAttackResultV1(..., passed=True)` objects and asserts they are all `passed`. The real suite is the `test_attack_*` functions (26 tests passed this session, including flagship tests).

| Attack | Class | Judgment |
|--------|-------|----------|
| **A** required live missing (connection disabled; indexed present) | **STRONG** | Indexed cannot close a live obligation; HTTP 0; LLM 0; typed UNSATISFIED. Core thesis. |
| **B** mid-flight revoke | **STRONG** | Same invariant as scenario 03; still the highest-value attack. |
| **C** wrong binding | **MEDIUM** | Plan validation `live_binding_not_found`. Normal ACL, correctly fail-closed before HTTP. |
| **D** wrong tenant | **WEAK** | `_WorkspaceAuthority` stub → `workspace_not_found`. Not multi-tenant isolation under real infra. |
| **E** wrong workspace | **WEAK / REDUNDANT** | Same as D. |
| **F** malformed / invalid-schema payload | **STRONG** | HTTP 1, no live provenance, LLM 0, persisted `INSUFFICIENT_EVIDENCE`. Proves “called” ≠ “satisfied”. |
| **G** 404 / 503 | **STRONG** | Same class as F; worth having both transport and schema failure. |
| **H** caller downgrade | **STRONG** (contract) / **MEDIUM** (coverage) | Pydantic `provider_request_requires_live_mode` + compose duplicate reject. Does not prove a HYBRID caller can omit `provider_request` and skip live obligations (they can; see qualifications). |
| **I** stale plan | **WEAK / REDUNDANT** | Same harness flag as B (`revoke_after_indexed=True`). Does not feed a serialized stale plan into a second process. |
| **J** connection disabled | **MEDIUM** | Overlaps A; useful as connection-status vs binding-disable, but not a new architectural boundary. |
| **K** capability mismatch | **MEDIUM** | Plan `live_capability_not_allowed`. Standard allowlist. |
| **L** EPHEMERAL leak | **MEDIUM** | Important honesty test; defense layer labeled `SYNTHESIS_BLOCKED` is wrong (LLM = 1). Proves persistence projection, not synthesis blocking. |
| **M** historical immutability | **MEDIUM** | Same-process `get_run` after control API mutation. Not durable store / attacker-with-write-access. |
| **N** wrong call evidence | **MEDIUM** | Unit-level `evaluate_evidence_admissibility` `LIVE_CALL_MISMATCH`. Does not show the orchestrator attempting to swap call ids. |
| **O** duplicate/replay | **WEAK** | Unique `requirement_id` at compose. Not cryptographic replay protection, not duplicate HTTP replay. |

**Net:** the suite is directionally serious (A/B/F/G/H). It is padded with tenant stubs, a tautological matrix, and a clone of the revoke test. A buyer should be shown A, B, F/G, H — not a 15-row PASS table.

---

## Trust / reliability assessment

| Property | Score (1–5) | Basis |
|----------|-------------|--------|
| Deterministic enforcement | **4** | Typed models, plan validation, admissibility function. Weakened by optional `runtime_authority` on the executor and proof-local provider strategy for live obligations. |
| Fail-closed synthesis | **4** | Flagship path: UNSATISFIED → assembler never constructed. Generic HYBRID without live obligations can still invoke the LLM, then fail citations. |
| Current authority | **5** | Reload-before-HTTP is implemented and proven for disable-after-index. |
| Stale data risk | **3** | Live path is fresh HTTP. Indexed policy can be stale relative to un-reindexed files. No staleness SLA. Hash embeddings are not a quality control on retrieval. |
| Provenance | **3** | SHA-256 content hashes and ids. Not signed. Live payload discarded under EPHEMERAL. |
| Auditability | **3** | Enough to say “this run claimed SATISFIED with hash H”. Not enough to re-try the factual question from history. In-memory store in the proof. |
| Provider failure semantics | **4** | HTTP 0 vs HTTP 1 distinguished; malformed/5xx finalize `INSUFFICIENT_EVIDENCE` rather than crashing the Ask. Receipts only when retention is `RECEIPT_ONLY`. |

---

## Business value

The proof is a **deployment-readiness toy** (ORION, score ≥ 90, no OPEN security blocker). It is a legitimate **analogy** for construction claims, compliance gates, go-live decisions, and internal policy questions. It is **not** domain certification, legal advice, or a regulated-product control.

**Why a company cares that synthesis is structurally blocked rather than prompted to be careful:**

- A prompt is advice to a stochastic component. The model can ignore it, and the system still emits an answer.
- The flagship gate means: **no synthesizer call, `answer is None`, status `insufficient_evidence`, LLM count 0**, and (when revoked) **zero provider HTTP**. That is a different operational and review posture: “we refused to answer” versus “we asked the model to be cautious”.
- Measurable non-invocation (HTTP 0 / LLM 0) is something security and compliance reviewers can test. Prompt text is not.

Buyer-relevant **if** the customer already has indexed policy plus a live system of record and needs the answer to be **inadmissible** when either is missing or unauthorized. Buyer-irrelevant **if** they only wanted Slack + RAG.

Do not claim this proof reduces hallucination rate, wins RFPs on “enterprise-grade”, or replaces a GRC platform.

---

## Hard-to-copy analysis

**What another competent team would need:**

| Band | Work |
|------|------|
| **A. Days** | RAG over a folder; HTTP tool; prompt “use policy + status”; store citations. |
| **B. Moderate integration** | Workspace policy object; allowlists; persist run records; `if not both sources: return None`. |
| **C. Hard because of cross-layer contracts** | Separate plan vs obligations; append-only compose; execution-time config reload that physically skips the handler; call-id-matched admissibility; EPHEMERAL projection that cannot store live bodies; provider failure as a valid typed run; serving wiring that refuses Ask V2 live catalog without runtime authority. |
| **D. Not proven hard to copy** | Multi-tenant isolation, cryptographic integrity, durable production stores, admin-path security, scale, concurrent disable vs in-flight HTTP, default productization of provider obligations in LKW HTTP/Slack. |

Large codebase ≠ defensibility. Band C is real engineering, not a secret algorithm.

---

## Remaining objections

### Blocker for the flagship claim

None that falsify the **narrow** claim: *Ask V2 can require indexed+live evidence, re-check live authority at execution, skip synthesis when required evidence is missing, and retain structural run identity under EPHEMERAL retention — as exercised by this harness.*

The following **would** be blockers if the claim were broadened to “LKW product already does this for every hybrid question” or “historically reconstructable”:

- Production `WorkspaceAskServiceV2` serving (`workspace_routes.py`) injects runtime authority but **does not** attach a Project Status `provider_strategy`. Live obligations in COMM-5 are proof-harness (`_OrionDeploymentProviderStrategy`).
- EPHEMERAL history cannot replay the live facts that justified NO.

### Not a blocker, but a limitation

- In-memory document/vector stores in the proof; not Trusted Ask restart durability.
- Hash embeddings; retrieval quality is not proven.
- Deterministic synthesizer; hosted LLM policy application is not proven.
- `_RevokeAfterIndexedRetriever` is proof-only timing injection (disable itself is canonical).
- `LiveCapabilityExecutorV1.runtime_authority` is optional; mis-wiring would skip the check (serving currently guards live catalog + missing authority).
- TOCTOU between `is_usable == True` and `handler.execute` HTTP is untested.
- Adversarial D/E/I/O/matrix are weak.

### Future hardening

- Concurrency / distributed config invalidation
- Durable production stores and signed run records
- Provider compromise and provenance authenticity (hash is not a signature)
- Policy administration authentication/authorization (proof seeds via repository puts)
- Scale / performance
- Multi-tenant isolation on real infra
- Cryptographic integrity
- Historical replay limits made explicit in every customer-facing sentence
- Default provider strategies so LKW serving attaches live obligations without the COMM-5 harness

---

## Claim safety

### SAFE CLAIMS

- Intergrax Workspace Ask V2 can treat evidence requirements as a typed plan contract, not only as prompt text.
- When a live evidence obligation is on the validated plan, missing or invalid live evidence blocks answer synthesis (`INSUFFICIENT_EVIDENCE`, LLM calls = 0 in the flagship/adversarial path).
- Indexed evidence cannot satisfy a required live `call_id` obligation.
- Live access that was valid at plan time can be denied at execution after a canonical binding disable; the Project Status HTTP read need not occur.
- Provider HTTP that returns malformed/404/503 need not count as satisfying a live obligation.
- Under EPHEMERAL retention, durable live evidence is provenance (ids, hash, binding, call), not the raw body.
- A prior Ask run can be re-read with the same answer/status/admissibility after the live world changes, in this in-memory proof store.
- The four-scenario CLI is locally reproducible without cloud credentials.

### QUALIFIED CLAIMS

- **“Runtime-governed”** — true for the wired Ask V2 live executor with authority injected; not a property of every Intergrax agent loop.
- **“Fail-closed”** — true for the obligation gate and many plan validations; not true that every HYBRID ask skips the LLM.
- **“Evidence-aware”** — true as contracts + admissibility; not true as semantic understanding of policy.
- **“Authorized live knowledge”** — true for binding + connection + capability checks on the live call path.
- **“LLM cannot bypass required evidence”** — true **once obligations are on the plan and admissibility runs first**. False as a blanket statement about all Ask modes and about a jailbroken model that never gets called.
- **“Historically explain why NO was valid”** — allowed only as **structural** explanation (obligations, hashes, revision, status). Not as factual replay of blocker state.

### UNSAFE CLAIMS (README must not make)

- Prevents hallucinations / always correct
- Enterprise-grade / production-ready / certified / compliant
- Unique technology / unmatched / only platform that…
- Fully auditable / historically reconstructable / bitemporal replay of live payloads
- Default LKW Slack or HTTP Ask already is this flagship hybrid proof
- Adversarial suite is a penetration test or security certification
- Semantic RAG quality, hosted LLM fidelity, durable multi-tenant production isolation

README currently states Hybrid Ask combining indexed and authorized live evidence is **not yet proven**. That sentence is **stale relative to this proof** and **still directionally honest relative to the LKW first-screen product** (indexed Ask V1 / AURORA-17). Fix by **replacing the story**, not by deleting the qualification about product packaging.

---

## README first-screen recommendation

**Headline:**  
Governed hybrid knowledge: answers are admissible only when required indexed and live evidence is present under current authority.

**Supporting sentence:**  
Intergrax plans what evidence is mandatory, fetches it through authorized indexed and live sources, re-checks live permission at execution time, and records structural proof — or returns no answer.

**Key visual concept:**  
Policy (indexed) + live system of record + runtime authority → **admissibility gate** → answer **or** stop. Not “chat over documents”.

**Proof matrix (30-second):**

| | HTTP | LLM | Result |
|--|-----:|----:|--------|
| Reality (OPEN blocker) | 1 | 1 | NO |
| Freshness (blocker closed) | 1 | 1 | YES |
| Authority revoked after plan | **0** | **0** | cannot determine |
| History | — | — | prior NO still recorded |

**Command:**  
`uv run python -m proof_infrastructure.governed_hybrid_knowledge_proof`

Do not lead with AURORA-17, Slack, or a linear “approved knowledge → grounded answer” RAG strip.

---

## Documentation gaps

### Current README

**Yes — the first screen underrepresents, and partly contradicts, current capability.**

- Hero workflow is indexed Ask V1 (AURORA-17, Slack-as-surface, “grounded answer”).
- Explicit line: hybrid indexed + authorized live is **not yet proven** — false for COMM-5D/E as a **technical proof**; still true that it is not the **product quickstart**.
- Platform map still frames Knowledge as RAG-first.

**Relegate:** AURORA-17 / indexed-only quickstart as “simplest local path”, not the thesis.  
**Supersede with:** governed hybrid admissibility (this proof) as the flagship technical story for architects/CTOs. Keep R&D / non-certification honesty.

### Proof doc `GOVERNED_HYBRID_KNOWLEDGE_PROOF.md`

| Dimension | Score | Notes |
|-----------|-------|--------|
| Technical accuracy | **4** | Matches harness/Ask V2 for the four scenarios. Over-speaks 04 as “why NO was valid” without saying the OPEN payload is gone. Does not disclose proof-local provider strategy vs serving. |
| Clarity | **4** | Thesis, matrix, CLI, limitations on EPHEMERAL are readable. |
| Visual quality | **3** | Adequate mermaid; not a buyer-grade first screen. |
| Persuasion | **3** | Honest but still lists 15 adversarial PASS rows including weak tests. |
| Proof reproducibility | **5** | CLI ran 4/4 this session in ~14s; pytest 26 passed. |
| Limitations honesty | **4** | EPHEMERAL called out. Missing: fake embeddings, deterministic LLM, in-memory store, proof-only revoke wrapper, serving without provider strategy. |

**COMM-5G changes (exact):**

1. Lead with the admissibility gate and scenario 03 (HTTP 0), not with RAG flavor.
2. State that live **obligations** in this proof are attached by a provider evidence plan in the harness; generic serving Ask V2 today revalidates authority but does not automatically attach Project Status live obligations.
3. Rewrite 04 as structural identity; forbid “full reconstruction”.
4. Disclose hash embeddings + deterministic synthesizer + in-memory stores.
5. Split adversarial table: STRONG vs supporting; drop or footnote the tautological matrix test.
6. One visual: gate, not a second RAG flowchart.
7. Keep the existing CLI and test commands; they work.

---

## Acceptance score

| Dimension | Score |
|-----------|------:|
| Technical credibility | 82 |
| Architectural differentiation | 68 |
| Business relevance | 70 |
| Proof realism | 64 |
| Adversarial strength | 61 |
| Auditability | 72 |
| Documentation clarity | 73 |
| Visual communication | 68 |
| Reproducibility | 88 |
| Claim safety | 78 |
| **Overall** | **72** |

Scores are not inflated for sophistication. Realism and adversarial strength are the drag: controlled HTTP, hash embeddings, deterministic LLM, in-memory persistence, and a padded attack list.

---

## Go / No-Go

| Decision | Result |
|----------|--------|
| GO for COMM-5G documentation showcase? | **YES** |
| GO for README flagship replacement? | **YES** |
| GO for external promotion as a differentiator? | **YES WITH QUALIFICATION** |
| GO for claiming unique technology? | **NO** |

Qualification for external promotion: talk about **enforced composition** (obligations, execution-time authority, fail-closed admissibility, structural provenance). Do not talk about uniqueness, certification, hallucination elimination, or historical replay. Do not imply the Slack/AURORA-17 quickstart is this proof.

---

## Final recommendation

Ship COMM-5G as an **architect/CTO showcase of Ask V2 governed hybrid knowledge**, with claim hygiene from this audit. Replace the README first screen so Intergrax is not introduced as simple RAG. Keep the proof CLI as the 30-second evidence.

Do **not** position this as unique technology. Position it as: **most stacks give the model tools; this path decides whether an answer is allowed to exist.**

Follow-ups that increase conviction (not required for COMM-5G copy): production provider strategy on the serving path, durable store proof, hosted-LLM synthesis test, and a non-redundant adversarial slice (A, B, F/G, H only) for buyer demos.

**Production changes in this task: NONE.**
