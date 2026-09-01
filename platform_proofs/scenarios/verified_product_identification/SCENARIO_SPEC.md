---
scenario_slug: verified_product_identification
lifecycle: IMPLEMENTATION_INITIALIZED
implementation_status: INITIALIZED
intergrax_fit: COMPLETED
gap_decision: RESOLVED
observability_contract: COMPLETED
application_vs_proof_ownership: COMPLETED
---

# Scenario Specification

**Scenario:** Verified Product Identification at Catalog Scale  
**Slug:** `verified_product_identification`  
**Proof class:** SCENARIO  
**Status:** IMPLEMENTATION INITIALIZED — gated skeleton generated; domain implementation in progress; no executable proof, evidence, or report yet.

[← Back to public Scenario page](README.md)

---

## A. SCENARIO

### Real problem

Large distributors, manufacturers, parts catalogs, and ecommerce operators maintain catalogs with **millions of product offers**. Offers arrive from heterogeneous retailers with inconsistent structure: GTINs, MPNs, retailer SKUs, key-value attributes, HTML spec tables, partial descriptions, and noisy metadata.

Users — technicians, buyers, support agents, procurement staff — describe products imperfectly:

- natural-language descriptions without identifiers;
- partial or mistyped model/part numbers;
- compatibility requirements ("same voltage as…", "fits 2019 F-150 crew cab");
- mixed constraints (brand + capacity + interface + form factor);
- fragments of identifiers (GTIN missing check digit, truncated MPN).

The catalog contains **near-identical variants** that differ by parameters that matter operationally: voltage, interface, ECC vs non-ECC, capacity tier, regional SKU, generation suffix.

The business task is **verified product identification**: determine whether a specific product identity is supported by catalog evidence — or honestly refuse when evidence is insufficient, contradictory, or ambiguous.

This is **not** semantic search over 3.77M records. It is **not** RAG over a product database. It is identity establishment under ambiguity at catalog scale.

### Who has the problem

- **Parts and procurement teams** sourcing replacement components from large heterogeneous catalogs.
- **Technical support and field service** identifying correct variants from customer descriptions.
- **Ecommerce catalog operators** matching user intent to the correct offer/SKU among millions of near-duplicates.
- **AI platform owners** who must prove identification systems do not equate retrieval ranking with verified identity.

### Why it matters

Selecting the wrong variant has direct operational consequences. Unlike a low-stakes search suggestion, a wrong identity decision can propagate into physical fulfillment, installation, and service workflows. Systems that sound confident while returning the top semantic match create false operational certainty.

### Failure consequences

- wrong replacement component shipped (incompatible voltage, interface, size, or capacity);
- incorrect RAM/storage/network part in a configured system;
- wrong automotive or industrial component installed;
- return/RMA and reverse logistics cost;
- technician or line downtime while correct part is re-identified;
- incorrect bulk procurement;
- wasted time across support, warehouse, and field teams;
- erosion of trust in AI-assisted catalog tools that "always find something."

### Why it is difficult

The canonical catalog is real and adversarial by nature:

**Scale and heterogeneity**

- 3,770,377 selected offers from WDC V2 non-normalized corpus;
- 2,753,163 unique `cluster_id` values; 399,868 multi-offer clusters containing 1,417,082 offers;
- 24,641,565 raw attribute entries across 337,163 unique attribute names;
- 1,129,415 offers with some GTIN; 714,082 with multiple identifier types.

**Missingness and noise**

- 2,242,958 offers without brand; 1,547,289 without category; 1,363,772 without description;
- multilingual content, encoding damage, retailer-local identifiers;
- `specTableContent` present on all selected records but often contains store boilerplate, shipping tables, or size charts rather than true product specs;
- `keyValuePairs` and `specTableContent` sometimes conflict or differ in scope.

**Identity ambiguity**

- near-identical variants differing by critical parameters;
- semantic descriptions that match multiple products;
- identifier collisions and retailer-local SKU semantics;
- incorrect or overly broad categories.

**Evaluation difficulty**

- `cluster_id` is useful reference signal from WDC, not infallible ground truth;
- full-catalog search at 3.77M scale is required for credible proof — tiny fixtures alone are insufficient.

### Naive / simple failure mode

**Vector search + LLM picker:**

```text
embed 3.77M offers
→ retrieve top-10 by cosine similarity
→ rerank (optional)
→ LLM: "pick the best match"
→ return top candidate as product identity
→ PASS (wrong)
```

This fails because:

- semantic similarity ≠ material identity;
- reranker score ≠ verification;
- the model may confidently pick a near-identical incompatible variant;
- missing distinguishing constraints are not surfaced;
- contradictory hard specs can be ignored in fluent prose;
- local-SKU collisions look plausible;
- abstention is not a first-class outcome.

**Exact identifier lookup only:**

```text
extract identifier from query
→ exact match in catalog
→ return match
→ PASS (wrong when identifier partial, wrong, or absent)
```

This fails because realistic requests omit identifiers, contain typos, use natural language, or require distinguishing variants within the same product family.

**Harness theater:**

```text
proof benchmark injects correct candidate into top-1
→ verifier always passes
→ PASS (wrong)
```

### WOW factor

WOW is **not** embedding 3.77M products, hybrid search plumbing, or agent count.

WOW is the visible separation:

```text
USER REQUEST — incomplete / imprecise product description
↓
CATALOG — 3.77M noisy offers; several convincing candidates
↓
MULTI-CHANNEL RETRIEVAL — recall-oriented candidate generation
↓
RERANKING — better ordering, still not identity proof
↓
EVIDENCE VERIFICATION — material constraints checked against source fields
↓
VERIFIED — only when evidence supports identity
   or
BOUNDED REFUSAL — AMBIGUOUS / INSUFFICIENT_INFORMATION / NO_MATCH
```

A skeptical reviewer should see: **the top-ranked candidate is not automatically the verified product**, and the system can refuse rather than fabricate certainty.

### Skeptic Challenge

> "Why isn't this just: embed 3.77M products, retrieve top-10, rerank, ask GPT to choose one?"

**Design response:**

| Skeptic claim | Design counter |
| --- | --- |
| "Vector search finds the product" | Vector recall finds *candidates*; similarity score is not identity evidence. Near variants can score higher than the correct product on one channel. |
| "Reranking fixes it" | Reranking improves ordering; it does not verify material constraints. A reranker can promote a semantically polished incompatible variant. |
| "The LLM will pick correctly" | Models confidently select plausible but wrong variants when hard specs conflict or distinguishing facts are missing. |
| "Just use exact identifier lookup" | Many real requests lack reliable identifiers; partial codes and retailer SKUs collide; variants share family names. |
| "3.77M scale is the hard part" | Scale is necessary but not sufficient. The hard part is **verified identity under ambiguity** at scale. |
| "Cluster ID gives you ground truth" | WDC clusters are reference signal for benchmark construction, not infallible oracle labels for every identity decision. |

The proof must demonstrate that **retrieval quality** and **verification safety** are separate measured obligations.

> "Why not just exact identifier lookup?"

Because the scenario includes cases where users provide natural language only, partial identifiers, typos, model names mixed with compatibility constraints, and close variants that share identifiers or family names. Identifier channels are first-class — but not sufficient alone.

### Adversarial conditions

Future benchmark case classes (design — fixtures not yet created):

| # | Case class | What it tests |
| --- | --- | --- |
| 1 | Exact identifier match | GTIN/MPN/SKU channel recalls correct offer; verification confirms |
| 2 | Partial identifier | Truncated or incomplete code still routes to candidate set |
| 3 | Typo / malformed product code | Lexical/normalized retrieval tolerates edit distance without false verification |
| 4 | Natural-language description only | Semantic recall without identifier; verification uses attributes |
| 5 | Close product variant | Near-identical titles; hard spec distinguishes (ECC vs non-ECC, SATA vs NVMe) |
| 6 | Correct product not rank-1 on one channel | Fusion/recall must surface true candidate before verification |
| 7 | Semantic near-match with conflicting hard spec | High similarity; verification rejects |
| 8 | Missing critical distinguishing constraint | `INSUFFICIENT_INFORMATION` — user or catalog lacks distinguishing fact |
| 9 | Two indistinguishable candidates | `AMBIGUOUS` abstention |
| 10 | Incorrect/missing category | Retrieval must not rely solely on category metadata |
| 11 | Missing brand | Identification from specs/description/identifiers |
| 12 | Noisy `specTableContent` / boilerplate | Evidence extraction ignores boilerplate; uses source fields |
| 13 | Multilingual query vs catalog content | Recall across language mismatch where feasible |
| 14 | Identifier conflict | Typed identifiers disagree; verification does not silently override |
| 15 | No valid product exists | `NO_MATCH` — not retrieval failure |

### Scenario Quality Gate

This scenario is a **candidate** for human Scenario Quality Gate because:

- real enterprise pain exists (wrong part identity at scale);
- failure has meaningful operational cost;
- the problem was not invented to demo a single platform feature;
- naive vector search + LLM picker is a credible false solution;
- ambiguity, missing evidence, near-duplicates, and identifier noise are intrinsic;
- outcomes are evaluable with explicit PASS/FAIL semantics;
- the story is understandable without Intergrax internals;
- WOW comes from verified identity vs retrieval ranking, not infrastructure theater;
- a skeptical engineer can challenge the design before any code exists;
- a real 3.77M-offer dataset foundation is validated with measured statistics.

**Gate decision:** ACCEPTED — independent human Scenario Quality Gate completed (2026-09-01).

### Application Survival Test

> If proof infrastructure, evaluator, evidence packaging, and report generation are removed, does a useful autonomous application component remain that still solves the underlying problem?

Required answer: **YES**.

The production application must still:

- accept user product-identification requests;
- query the full catalog through the retrieval stack;
- verify identity against catalog evidence;
- return `VERIFIED`, `AMBIGUOUS`, `INSUFFICIENT_INFORMATION`, or `NO_MATCH`;
- surface clarification needs when distinguishing facts are missing.

If removing proof infrastructure removes verification or catalog search, redesign is required.

### Application Observability Test

> If the proof evaluator, evidence packaging, and HTML report are removed, does the application/runtime still produce enough structured execution information to reconstruct its material decisions, actions, observations, challenges, recoveries, diagnostics, and terminal result?

Required answer: **YES**.

Proof must consume production-path structured observations. Proof must not reconstruct fictional model reasoning post hoc.

### Observability / Explainability / Diagnostics Contract

**Material decisions:** (application-owned, must be observable):

1. Parsed query constraints (identifiers, brand, model, hard/soft constraints, negatives, missing facts).
2. Retrieval channel invocation and per-channel candidate counts.
3. Fusion strategy outcome (which candidates entered the merged pool).
4. Reranker ordering of finalists.
5. Evidence extraction per finalist (which source fields support/contradict constraints).
6. Verification verdict per finalist and terminal outcome selection.
7. Abstention/clarification reason when not `VERIFIED`.

**Observability coverage** — production-path trace must expose at minimum:

| Stage | Observable fields |
| --- | --- |
| Query understanding | extracted identifiers (typed), brand, model/series, product class, hard constraints, soft preferences, negative constraints, missing distinguishing facts |
| Identifier extraction | raw token, normalized form, identifier type, confidence/source span |
| Retrieval | channels invoked (`exact`, `lexical`, `structured`, `vector`), query per channel, candidate count per channel, top candidate IDs per channel |
| Fusion | merged candidate pool size, channel contribution per candidate |
| Reranking | pre/post rank positions for finalists, reranker model/score (bounded) |
| Top-K selection | finalist offer IDs entering verification |
| Evidence extraction | source field provenance (`record_json` path), supporting constraints, contradicting constraints, missing constraints |
| Verification | per-finalist status, aggregate terminal outcome, abstention reason |
| Timings | per-stage latency (bounded aggregates acceptable) |
| Failures | retrieval errors, empty channels, verification errors with structured diagnostics |

**Explainability:** Each verification decision must cite **catalog evidence references** (offer ID + source field path + extracted value). Confidence scores alone cannot override explicit contradiction.

**Evidence linkage:** Evidence items reference immutable source `record_json` fields — not derived-only search text without provenance.

**Action correlation:** Tool/search invocations link to retrieval channel traces.

**Diagnostics:** Structured failure states for empty retrieval, verification timeout, identifier parse failure, database unavailable.

**Redaction:** PII in raw descriptions if present in source records; full 3.77M embeddings or bulk record dumps are **not** required in traces.

**Operator visibility:** Terminal outcome, finalist offer IDs, constraint support/contradiction summary, abstention reason.

**Proof consumption:** Proof report projects from canonical structured trace artifacts. Proof evaluator compares terminal outcome and evidence citations against hidden benchmark expectations — it does not re-run verification.

**Machine-readable artifact:** Expected projection includes query constraints, channel metrics, finalist evidence graph, terminal outcome — design-stage; exact schema deferred to implementation.

**Application Observability Test result:** **YES** (by design — required before implementation acceptance).

### Conditional authoring prompts

**Hidden truth / evaluator leakage:** Benchmark cases carry hidden expected outcome and expected identity (offer ID and/or cluster reference). Hidden expectations are available to the **proof evaluator only** — never to the application runtime, query prompts, retrieval indexes exposed to the model, or verification prompts. Proof must **never** inject the correct product into the candidate set.

**Evidence boundary:** Legitimately observable evidence is whatever the application retrieves from the canonical catalog through declared channels — `record_json` source fields, derived search indexes with provenance back to source, and parsed query constraints from the user request. Evaluator ground truth is not runtime evidence.

**Alternative hypotheses / failure alternatives:** For each case, plausible wrong candidates must remain in the search space — semantically similar variants, identifier collisions, category neighbors. The system must distinguish verified identity from plausible alternatives using material constraints.

**Independence:** Verification evaluates candidates against extracted constraints and source evidence. It does not receive hidden benchmark labels, expected offer IDs, or proof-evaluator verdicts during execution.

**Temporal semantics:** **Not material** for this scenario. Catalog snapshots are quasi-static for a proof run. Price freshness and offer availability are out of scope unless explicitly added in a future revision.

**Side effects / recovery / HITL / governance:** No mutating side effects in scope. Clarification questions to the user are an application behavior (`INSUFFICIENT_INFORMATION`), not proof-harness behavior. Governance relevant only insofar as the application must not fabricate catalog evidence.

---

## B. SOLUTION

### APPLICATION vs PROOF HARNESS

| APPLICATION / PRODUCTION PATH OWNS | PROOF OWNS |
| --- | --- |
| Query understanding (identifiers, constraints, missing facts) | Benchmark case definitions |
| Multi-channel candidate generation (exact, lexical, structured, vector) | Hidden expected identity / expected outcome |
| Hybrid fusion and reranking | Adversarial case selection and stratification |
| Evidence extraction from source `record_json` | Evaluator (outcome + evidence checks) |
| Identity / constraint verification | Metric aggregation (recall, MRR, FP rate, abstention correctness) |
| Terminal outcome decision (`VERIFIED`, `AMBIGUOUS`, `INSUFFICIENT_INFORMATION`, `NO_MATCH`) | PASS/FAIL verdict |
| Clarification requirement surfacing | Evidence projection for report |
| runtime execution trace / diagnostics (bounded) | Reproduction metadata |
| Source provenance back to catalog fields | Baseline comparison harness configuration |
| Database/search infrastructure for full catalog | Gold-set curation pipeline (future) |
| Derived search indexes (with provenance) | Leakage guards and fixture isolation tests |

**PROOF DOES NOT OWN:** fabricated rationale; reconstructed model intent not present in runtime artifacts; post-hoc explanation generated by another LLM; inserting correct candidates into retrieval results; running the verifier on behalf of the application.

### Desired behavior

The identification system behaves like a disciplined catalog engineer:

1. **Parse** the user request into typed identifiers, brand/model hints, hard constraints, soft preferences, negative constraints, and missing distinguishing facts.
2. **Generate candidates** through independent retrieval channels optimized for recall.
3. **Fuse and rerank** to improve ordering — without treating rank as identity.
4. **Extract evidence** from immutable source catalog fields for each finalist.
5. **Verify** material identity constraints — supporting, contradicting, and missing evidence explicit.
6. Return **`VERIFIED`** only when constraints are supported and no disqualifying contradiction remains.
7. Return **`AMBIGUOUS`** when multiple candidates remain materially indistinguishable.
8. Return **`INSUFFICIENT_INFORMATION`** when required distinguishing facts are missing.
9. Return **`NO_MATCH`** when candidates are sufficiently rejected and no verified product remains.
10. Emit **bounded structured trace** sufficient for proof projection without proof-only logging.

### Step-by-step story

#### VERIFIED path (intended success story)

```text
USER — "Samsung 990 PRO 2TB M.2 2280 NVMe PCIe 4.0"
↓
QUERY UNDERSTANDING — brand=Samsung; model family=990 PRO; capacity=2TB (hard);
  form=M.2 2280 (hard); interface=NVMe PCIe 4.0 (hard/likely); no GTIN provided
↓
CANDIDATE GENERATION
  exact: no GTIN in query → channel skipped or low yield
  lexical: "990 PRO" + "2TB" → recall set includes Samsung SSD variants
  structured: capacity=2TB, form factor candidates
  vector: semantic recall from natural-language query
↓
FUSION — merged pool includes correct 2TB 990 PRO and near variants (1TB, SATA, etc.)
↓
RERANKER — promotes likely Samsung 990 PRO variants; rank-1 may still be wrong variant
↓
EVIDENCE EXTRACTION — for top finalists, read source KVP/spec fields with provenance
↓
VERIFICATION
  finalist A (990 PRO 2TB NVMe): capacity ✓ interface ✓ form ✓ brand ✓ → supports
  finalist B (990 PRO 1TB NVMe): capacity ✗ → contradicts
  finalist C (870 EVO 2TB SATA): interface ✗ → contradicts
↓
VERIFIED — finalist A; evidence trail cites source fields
↓
RESOLVED envelope
```

#### AMBIGUOUS path

```text
USER — partial description matches two offers with identical material specs in catalog
↓
retrieval surfaces both finalists with equivalent supporting evidence
↓
verification cannot find material distinguishing constraint
↓
AMBIGUOUS — abstain; do not force top-1
↓
UNRESOLVED envelope
```

#### INSUFFICIENT_INFORMATION path

```text
USER — "Lenze motor for line 4" (voltage/coupling not specified)
↓
retrieval surfaces multiple Lenze motors with different voltage ratings
↓
verification: missing critical distinguishing constraint in query and not inferable from catalog alone
↓
INSUFFICIENT_INFORMATION — surface what fact is needed (e.g., voltage, mounting, part number)
↓
UNRESOLVED envelope
```

#### NO_MATCH path

```text
USER — "ABC-9999 widget 48V" (product does not exist in catalog)
↓
retrieval may surface near matches (ABC-9998, 24V variant)
↓
verification: identifier and voltage constraints contradict all finalists
↓
NO_MATCH — conclusive rejection; no fabricated product
↓
RESOLVED envelope
```

### Ideal architecture

Design from first principles — not constrained by current Intergrax implementation.

```mermaid
flowchart TD
    UQ[USER QUERY] --> QU

    subgraph QU[QUERY UNDERSTANDING]
        direction TB
        QU1[identifier extraction]
        QU2[brand / model]
        QU3[product class]
        QU4[hard constraints]
        QU5[soft preferences]
        QU6[negative constraints]
        QU7[missing facts]
    end

    QU --> MCR

    subgraph MCR[MULTI-CHANNEL RETRIEVAL]
        direction TB
        subgraph CH1[1. EXACT IDENTIFIER]
            E1[GTIN]
            E2[MPN]
            E3[SKU]
            E4[productID]
        end
        subgraph CH2[2. LEXICAL / BM25]
            L1[model numbers]
            L2[part numbers]
            L3[technical tokens]
            L4[exact phrases]
        end
        subgraph CH3[3. STRUCTURED ATTRIBUTE SEARCH]
            S1[voltage]
            S2[dimensions]
            S3[capacity]
            S4[interface / size]
            S5[compatibility constraints]
        end
        subgraph CH4[4. VECTOR / SEMANTIC]
            V1[natural-language similarity]
            V2[synonyms]
            V3[descriptive recall]
        end
    end

    CH1 --> HF[HYBRID FUSION]
    CH2 --> HF
    CH3 --> HF
    CH4 --> HF
    HF --> RR[RERANKER]
    RR --> TK[TOP-K FINALISTS]
    TK --> EE

    subgraph EE[EVIDENCE EXTRACTION]
        direction TB
        EE1[source field provenance]
        EE2[supporting evidence]
        EE3[contradicting evidence]
        EE4[missing evidence]
    end

    EE --> IV[IDENTITY VERIFICATION]
    IV --> OV[VERIFIED]
    IV --> OA[AMBIGUOUS]
    IV --> OI[INSUFFICIENT_INFORMATION]
    IV --> ON[NO_MATCH]
    OI --> CL[clarification question]
    CL --> UF[user supplies missing fact]
    UF --> QU
```

**Layer roles:**

- **Exact retrieval** — for strong typed identifiers (GTIN, MPN, SKU, productID).
- **BM25 / lexical** — for model numbers, part numbers, and exact technical tokens.
- **Structured retrieval** — for hard constraints such as voltage, capacity, dimensions, and interface.
- **Vector retrieval** — for natural-language and semantic recall.
- **Hybrid fusion** — merges independent candidate-generation signals without treating any single channel as identity proof.
- **Reranker** — improves ordering; it does **not** verify identity.
- **Evidence extraction** — binds each finalist to source-truth catalog fields with provenance.
- **Verification** — decides whether identity is actually supported by evidence.

**TOP-RANKED CANDIDATE IS NOT A VERIFIED PRODUCT.**

**Scenario architecture invariant (provider neutrality):** Scenario implementation **MUST NOT** be PostgreSQL-centric. Business/application code **MUST** depend on platform/application search and storage **contracts**, not PostgreSQL-specific APIs. Query understanding, product verification, outcome semantics, and the proof evaluator **MUST NOT** require rewriting when operators choose a different qualified storage/retrieval configuration. Provider-specific capabilities belong behind integration/storage/retrieval boundaries — not in scenario application logic.

**Canonical reference configuration:** For reproduction convenience, a later reference deployment **MAY** choose PostgreSQL + pgvector (relational/catalog store + vector backend in one deployable unit). That is an operator/provider **choice**, not a scenario architecture dependency. The scenario architecture remains **provider-neutral** and must also allow configurations such as:

- PostgreSQL + pgvector
- MySQL + Qdrant
- another relational/catalog store + another qualified vector backend

Illustrative trade-offs (provider choice — not mandatory scenario architecture):

| Configuration | Benefit | Cost / risk |
| --- | --- | --- |
| PostgreSQL + pgvector | Single ops surface; structured + lexical + vector co-location | Index tuning at 3.77M scale; attribute heterogeneity |
| MySQL (or other relational/catalog) + Qdrant | Mix relational catalog with specialized ANN/hybrid vector backend | Two systems; provenance/join complexity |
| Separate lexical engine (e.g., Elasticsearch/OpenSearch) + vector backend | Strong lexical | Additional infrastructure — only if measured lexical gap |

### Data preparation model

Two representations coexist:

**Immutable source truth (verification authority)**

- raw `record_json` per offer
- original identifiers, title, description, category, brand
- original `keyValuePairs`, `specTableContent`, `cluster_id`
- never silently overwritten by derived cleaning

**Derived search representation (recall authority)**

- normalized identifier forms (zero-stripped GTIN, case-folded MPN)
- lexical index text (title, description, model tokens)
- structured/search fields for high-value attributes (capacity, voltage, interface — normalized subset, not all 337k raw names as columns)
- embedding vectors for semantic recall
- optional cluster_id for benchmark stratification

**Critical rule:** derived data may improve search, but verification must cite **source fields** with provenance. If derived normalization conflicts with source, source wins for verification unless explicit transformation rules are logged.

Preprocessing is deterministic, reproducible, and versioned. Not implemented in this design task.

### Candidate generation

Independent channels with distinct failure modes:

**A. Exact identifier retrieval** — GTIN / MPN / SKU / productID as typed lookups, not embedding text. Retailer-local SKU/productID are weak global identity signals — typed, not automatically authoritative across retailers.

**B. Lexical retrieval** — model numbers, part numbers, fragmented names, technical tokens, exact phrases.

**C. Structured attribute retrieval** — hard requirements on normalized fields (voltage, dimensions, capacity, interface, size). Not all 337k raw attribute names become relational columns; normalization targets high-value constraint families.

**D. Dense vector retrieval** — semantic/natural-language recall. Vector score is **never** sufficient evidence of identity.

### Fusion

Combine independent channel results into a recall-oriented candidate pool. Exact algorithm not frozen — must preserve channel attribution per candidate for observability. Goal: if the correct offer is findable by any channel, it enters the pool before reranking.

### Reranking

Cross-encoder or LLM reranker improves finalist ordering. Reranker score is **not** verification. A high reranker score with contradicting hard specs must still fail verification.

### Verification

Answers: **"Do we have enough evidence to assert this candidate is the product?"**

For each finalist, compare material constraints from query understanding against evidence extracted from source catalog fields.

| Evidence class | Treatment |
| --- | --- |
| Supporting | Constraint satisfied by source field with provenance |
| Contradicting | Source field conflicts with hard constraint — disqualifies finalist |
| Missing | Required constraint cannot be confirmed or denied from available evidence |

**Examples of contradiction (must NOT verify):**

- requested 32GB ECC RDIMM → candidate 32GB non-ECC UDIMM
- requested M.2 NVMe → candidate M.2 SATA
- requested part ABC-123 → candidate ABC-123A with different GTIN
- requested 48V → candidate 24V in source KVP

Confidence alone cannot override contradiction.

### Outcomes

| Outcome | Semantics | Envelope |
| --- | --- | --- |
| **VERIFIED** | Material identity constraints supported; no disqualifying contradiction | RESOLVED |
| **NO_MATCH** | Sufficient evidence to reject finalists; no verified candidate remains. **Not** retrieval failure. | RESOLVED |
| **AMBIGUOUS** | ≥2 finalists materially indistinguishable with available evidence | UNRESOLVED |
| **INSUFFICIENT_INFORMATION** | Required distinguishing facts missing from request or catalog | UNRESOLVED |

### Guarantees

Candidate system-level guarantees (design stage — not yet demonstrated):

- Retrieval and verification are **separate stages** with separate metrics.
- Exact identifiers are a **first-class typed channel**.
- Vector and reranker scores **cannot** single-handedly produce `VERIFIED`.
- Verification cites **source provenance** (`record_json` field paths).
- Contradictions **disqualify** finalists regardless of similarity score.
- Abstention outcomes are **first-class** (`AMBIGUOUS`, `INSUFFICIENT_INFORMATION`).
- `NO_MATCH` is a conclusive identity judgment, not "nothing found in top-k."
- Full canonical evaluation runs against the **3.77M-offer search corpus**.
- Proof never injects correct candidates or runs verification on behalf of the application.

### Claim

Candidate bounded falsifiable claim (design — **not** a proven public claim):

> **Given the declared canonical multi-million-offer catalog and adversarial product-identification cases, the system can generate candidate products through independent retrieval channels and accept an identification only when material identity constraints are supported by traceable catalog evidence; when decisive evidence is contradictory or insufficient, it returns a bounded non-verified outcome instead of presenting the highest-ranked candidate as fact.**

Avoid: "always," "all products," "zero hallucinations," "perfect compatibility," "universal product matching."

### PASS

Candidate PASS semantics — thresholds **TO BE FROZEN DURING PROOF DESIGN / BENCHMARK CALIBRATION**:

**Retrieval quality**

- Correct product/offer survives candidate generation for required case classes (recall@K).
- Independent retrieval channels are observable in trace.
- Correct product not ranked #1 on one channel still enters pool when another channel recalls it.

**Verification safety**

- Hard negative with high semantic similarity is **rejected** by verification.
- Identifier contradictions are not silently overridden.
- Verifier cites source evidence used (field provenance).
- `AMBIGUOUS` case abstains — no forced single product.
- `INSUFFICIENT_INFORMATION` surfaces missing distinguishing fact.
- `NO_MATCH` does not fabricate a product.

**Proof integrity**

- Canonical evaluation operates against full 3.77M-offer search corpus.
- Target architecture measured against simpler baseline(s) — improvement is numeric, not narrated.
- Hidden benchmark labels do not leak into runtime.
- Proof harness does not inject correct candidate or run application verification.

### FAIL

Explicit FAIL if any of the following occurs:

- PASS based only on model prose without structured evidence.
- Top-1 retrieval automatically accepted as `VERIFIED`.
- Vector or reranker score used as identity proof.
- Evaluation silently uses tiny fixture corpus instead of full catalog without explicit bounded scope declaration.
- Hidden ground truth leaks into application runtime context.
- Benchmark manually injects correct candidate into retrieval results.
- Missing candidate replaced by fabricated product.
- Contradicting hard spec ignored because of high similarity.
- `AMBIGUOUS` case forced to single product.
- `NO_MATCH` returned when a verified candidate exists, or `VERIFIED` when contradiction exists.
- Proof harness performs business verification instead of application.
- Baseline comparison omitted or narrated without measurement.

### Baseline / ablation plan

Future benchmark must prove architectural complexity earns its keep:

| Variant | Description |
| --- | --- |
| **BASELINE A** | Lexical/exact search only |
| **BASELINE B** | Dense vector search only |
| **BASELINE C** | Hybrid retrieval without reranker or verification (top-1/top-k acceptance) |
| **TARGET** | Hybrid candidate generation + reranking + verification |

Metrics (thresholds TO BE FROZEN during benchmark calibration):

- Recall@K / candidate recall per case class
- Ranking quality (MRR, NDCG@K where appropriate)
- Top-1 identity accuracy (verification-adjusted, not raw retrieval)
- False-positive identification rate
- Abstention correctness (`AMBIGUOUS`, `INSUFFICIENT_INFORMATION`)
- Hard-negative rejection rate
- Latency per stage (p50/p95)
- Resource/cost characteristics (optional, bounded reporting)

### Adversarial attacks

See § Adversarial conditions (15 case classes). Proof stratifies cases across identifier reliability, variant proximity, missingness, noise, and multilingual content where feasible.

### Excluded claims

- Universal compatibility checking across arbitrary systems or configurations.
- Perfect attribute normalization across all 337k raw attribute names.
- Infallible `cluster_id` as oracle ground truth.
- Zero false positives at production scale without stated thresholds.
- Proof that any single retrieval channel alone solves the scenario.
- Claim that PostgreSQL/pgvector is the only viable infrastructure.
- Real-time price accuracy or offer availability guarantees.
- Multilingual perfection across all catalog languages.

### Limitations

- Single bounded domain: **product identification from catalog evidence** at measured WDC scale.
- Dataset distribution / public artifact hosting **not finalized** (`DATASET_DISTRIBUTION = OPEN`).
- `cluster_id` useful for benchmark construction — requires quality filters and human/deterministic validation for hard gold sets.
- Implementation initialized (gated skeleton only); no executable proof or verified run yet.
- Benchmark thresholds not yet calibrated on gold set.
- Attribute normalization scope deliberately bounded — not all raw KVP names modeled relationally.

### Dataset provenance / reproducibility

| Item | Value |
| --- | --- |
| Source | Web Data Commons Large Scale Product Corpus V2 — `offers_corpus_all_v2_non_norm` |
| Original source size | 26,507,210 offers |
| Selection rule | `keyValuePairs != null` OR `specTableContent != null` |
| Canonical count | 3,770,377 product offers |
| Canonical artifact | `selected_offers.parquet` (~1.71 GiB, ZSTD, `record_json` column) |
| Builder | `dataset/build_wdc_dataset.py` — deterministic, streaming, lossless at JSON-value semantic level |
| Profiler | `dataset/profile_selected_dataset.py` — measured statistics on full canonical set |

**DATASET_DISTRIBUTION = OPEN / MUST RESOLVE BEFORE PUBLIC REPRODUCTION.**

Candidate mechanisms: dedicated proof-assets repository + release artifacts; GitHub Release asset; external immutable artifact storage with checksum verification in proof runner. Not chosen in this task.

Preferred future reviewer experience:

```text
clone repo
→ start/run scenario
→ canonical dataset automatically resolved/downloaded
→ checksum verified
→ database initialized
→ proof ready
```

---

## C. INTERGRAX FIT

**Status: COMPLETED**

Audit date: 2026-09-01 · repository HEAD `768c1d6f5bc55194d6e0bb7d2d8642a3caeceb68` · branch `development`.

Participating domains (discovered during fit): **RAG** (retrieval, vectorstore, reranker, ingest, routing), **Integrations** (PgVector provider), **Execution / Observability** (trace, functional diagnostics), **Decision System** (generic lifecycle only — not product identity), **Platform Proofs** (evidence / evaluator framework).

**TEST-ONLY SUBSTITUTE on canonical scenario path:** **NO** (implementation initialized — gated skeleton only; audit confirms no prohibited harness shortcuts are required for fit classification).

### Fit matrix

| Scenario need | Ideal role | Intergrax mechanism | Current owner | Evidence | Status | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| **A. Query understanding** — identifier extraction (GTIN/MPN/SKU/productID), brand/model, product class, hard/soft/negative constraints, missing facts | Parse user request into typed product constraints before retrieval | `QueryRouter` (`intergrax/rag/routing/query_router.py`) — RAG **cost/complexity tier** routing (`fast`/`standard`/`deep`); optional LLM tier classifier | Tier-0 RAG | `QueryRouter.route()` word-count / `?` heuristics; `classify_route_tier_with_llm` — no product field extraction | **MISSING** (product semantics) | **BUILD** |
| **A. Query understanding** — product constraint parsing | Domain-specific NL → typed constraints | None shipped | — | Grep: no GTIN/MPN/brand/capacity parser in `intergrax/` | **MISSING** | **BUILD** (application) |
| **B. Exact identifier retrieval** — typed GTIN/MPN/SKU/productID lookup at catalog scale | Independent recall channel; not embedding similarity | `MetadataFilter` equality on vector `payload` JSONB (`payload @> %s::jsonb` in `PgVectorRagStore._where_clause`) | Tier-0 RAG / PgVector provider | `test_pgvector_live_qualification.py` — `MetadataFilter(conditions={"group": 1})`; per-record `INSERT` not identifier index | **PARTIAL** — equality filter only; not a dedicated exact-identity channel | **BUILD** (application catalog indexes + lookup SQL) |
| **B. Exact identifier retrieval** — platform generic exact lookup channel | Reusable exact-match retriever | No `exact` / `identifier` retriever in `RetrieverRegistry`; only `vector_similarity`, `hybrid`, `graph_rag`, etc. | Tier-0 RAG retrievers | `retriever_bootstrap.py`, `retriever_registry.py` | **MISSING** | **BUILD** (application); platform reuse limited to `MetadataFilter` if app normalizes identifiers into payload |
| **C. Lexical / BM25 retrieval** — model/part numbers, technical tokens | Independent lexical recall channel | `LexicalIndex` BM25 (`intergrax/rag/vectorstore/sparse/lexical_index.py`); `LexicalHybridSupport.query_hybrid` RRF fusion (`lexical_hybrid.py`); `HybridRetriever` fallback token-overlap lexical (`hybrid_retriever.py`) | Tier-0 RAG | Unit/integration via InMemory/Qdrant stores; **not** PgVector | **SHIPPED** (InMemory/Qdrant/Weaviate hybrid path) | **REUSE** (non-PgVector backends) / **EXTEND** (PgVector — see M) |
| **C. Lexical / BM25** — PostgreSQL full-text at 3.77M scale | Durable lexical index co-located with vectors (design hypothesis) | PgVector provider: **no** `query_hybrid`, **no** FTS/BM25 | Integrations / PgVector | `pgvector/rag_store.py` — dense `query()` only; no `LexicalHybridSupport` mixin | **MISSING** on PgVector | **EXTEND** (platform PgVector) or **BUILD** (application PostgreSQL FTS tables) |
| **D. Structured attribute retrieval** — voltage, capacity, interface, dimensions, compatibility | Hard-constraint recall on normalized attributes | `MetadataFilter`: exact `field == value` + optional `IN` membership (`native_vectorstore.py`); PgVector rejects membership (`require_membership_support`) | Tier-0 RAG contracts | `MetadataFilter.matches_payload`; pgvector `_where_clause` JSON containment only | **PARTIAL** — equality/IN only; no ranges, no heterogeneous 337k-name normalization | **BUILD** (application normalization + indexes); **REUSE** equality filters where normalized |
| **E. Dense vector retrieval** | Semantic / NL recall channel | `PgVectorRagStore.query()` — cosine distance `embedding <=> %s::vector`, score `1 - distance` | Integrations / PgVector | `pgvector/rag_store.py`; `test_pgvector_live_qualification.py` (live gate) | **SHIPPED** / **QUALIFIED WITH LIMITATIONS** (50-doc soak, not 3.77M) | **REUSE** + **CONFIGURE** (DSN, dimension) |
| **E. Dense vector retrieval** — ANN index at millions scale | Approximate nearest-neighbor performance | Schema creates scope/source B-tree indexes only; **no HNSW / IVFFlat** on `embedding` | PgVector provider | `_ensure_schema()` — `idx_*_scope`, `idx_*_source`; no `CREATE INDEX ... USING hnsw/ivfflat` | **MISSING** | **EXTEND** (platform PgVector schema) |
| **F. Hybrid fusion** — merge exact + lexical + structured + vector with channel attribution | Recall-oriented pool; per-candidate channel provenance | `reciprocal_rank_fusion` + `LexicalHybridSupport.query_hybrid` (dense+lexical RRF); `fuse_graph_channels` score-sum (vector+keyword+graph); `HybridRetriever` single retriever orchestration; `RetrievalTrace.channel_contributions` | Tier-0 RAG | `lexical_hybrid.py`, `graph_channel_fusion.py`, `graph_rag_retriever.py`, `retrieval_service.py` `_apply_retriever_execution_trace` | **PARTIAL** — 2–3 channel fusion patterns exist; **no** generic 4-channel product fusion orchestrator | **BUILD** (application multi-channel orchestration); **REUSE** RRF/score-sum primitives |
| **G. Reranker** — cross-encoder / API rerank; score ≠ verification | Improve finalist ordering post-fusion | `BaseReranker`, `RerankerCandidate`, `RerankerManager`, providers: `cross_encoder`, `cohere`, `jina`, `semantic`, `embedding_cosine`, `rrf_reranker`; `RetrievalService.retrieve_single_pass` rerank stage | Tier-0 RAG rerankers | `reranker_types.py` (full `KnowledgeDocument` on candidate); `retrieval_service.py` L150–184; provider tests | **SHIPPED** | **REUSE** + **CONFIGURE** (`RagProfile.enable_rerank`, `reranker_id`) |
| **H. Top-K / candidate ABI** | Stable finalist handoff to verification | `RetrievalHit`, `RetrievalChunk`, `RerankerCandidate` — `channel`, `rank`, `vector_id`, `KnowledgeDocument` provenance/metadata | Tier-0 RAG | `base_retriever.py`, `retrieval_result.py`, `reranker_types.py` | **SHIPPED** | **REUSE** |
| **I. Evidence extraction** — source provenance, supporting/contradicting/missing per constraint | Bind finalists to immutable `record_json` fields | `KnowledgeDocument` provenance (`source_id`, field paths via metadata); RAG returns chunks — **no** constraint-level evidence graph | Tier-0 knowledge contracts | `KnowledgeDocument` schema; no product evidence extractor in platform | **MISSING** (product semantics) | **BUILD** (application) |
| **J. Identity verification** — material constraint check vs catalog evidence | Separate stage from retrieval ranking | Decision System — generic proposal→verification→resolution lifecycle (`DECISION_SYSTEM.md`); `VerificationLoop` in `runtime/adaptive/` is **critic/tool** verification, not product identity | Execution / Decision (generic) | `decision_strategy.py` protocol; no product constraint verifier implementation | **PARTIAL** — lifecycle contracts reusable; **no** semantic product verifier | **BUILD** (application verifier); **REUSE** Decision lifecycle **optionally** for terminal authority envelope |
| **K. Outcome semantics** — `VERIFIED`, `AMBIGUOUS`, `INSUFFICIENT_INFORMATION`, `NO_MATCH`, clarification loop | Bounded terminal business outcomes | Decision System uses `ACCEPTED`/`UNRESOLVED`/etc. — **not** 1:1 product identification outcomes | Application (per § B ownership) | Scenario spec § B; platform has no `VERIFIED` product outcome type | **MISSING** (scenario-specific) | **BUILD** (application); **NOT REQUIRED** in platform core |
| **L. Dataset ingest / indexing** — 3.77M offers, derived search representations | Deterministic preprocessing + index publication | `IngestPipeline` (loader→splitter→embed→vectorstore); `IndexingManager`; `add_records` upsert `ON CONFLICT DO UPDATE` | Tier-0 RAG ingest | `ingest_pipeline.py`; pgvector per-record insert loop; `async_job_recommended` flag for large sources | **PARTIAL** — document-chunk ingest path exists; **not** catalog-scale bulk product ingest | **BUILD** (application catalog pipeline); **EXTEND** (PgVector bulk ingest) |
| **M. PostgreSQL / PgVector** — operational store (design hypothesis) | Single deployable unit for relational + search | `PgVectorRagStore`, `create_pgvector_vector_store`, `VectorstoreManager` | Integrations | See **PgVector findings** below | **QUALIFIED WITH LIMITATIONS** | **REUSE** (dense) + **EXTEND** (scale features) |
| **N. Observability / diagnostics** — per-stage trace per observability contract § A | Production-path structured observability | `RetrievalTrace` (route, retriever, rerank, latencies, `hybrid_used`, `channel_contributions`); `rag_span`; C1 functional diagnostic spec (retrieval op + candidates) | Runtime diagnostics / RAG | `retrieval_result.py`; `c1_rag_functional_diagnostic_specification.py` — **generic RAG**, not product-identification stages | **PARTIAL** — RAG trace shipped; product stages (query constraints, per-channel counts, evidence graph) **not** pre-defined | **BUILD** (application trace schema); **REUSE** trace/diagnostic infrastructure |
| **O. Proof evidence / evaluator** — PASS/FAIL vs hidden benchmark; no runtime leakage | Proof consumes production artifacts | `PlatformProofEvidence` v3 (`scripts/proof/intergrax_platform_proof_evidence*.py`); evaluator pattern (`ai_incident_investigation/proof/evaluator.py`) | Platform proofs | `PLATFORM_PROOF_PROTOCOL.md`; existing scenario proof packages | **SHIPPED** (framework); scenario wiring **not** built | **REUSE** (framework) + **BUILD** (scenario evaluator) |

### PgVector findings (code-evidenced)

| Question | Finding | Evidence |
| --- | --- | --- |
| Dense retrieval? | **Yes** — cosine via pgvector `<=>` operator | `PgVectorRagStore.query()` L143–162 |
| Metadata filters? | **Yes** — JSONB containment `payload @> %s::jsonb`; scope keys enforced | `_where_clause()` L477–500; live qual test L463–478 |
| HNSW / IVFFlat / ANN index? | **No** — sequential scan ordering by distance; only B-tree on `(tenant_id, namespace, workspace_id)` and `source_id` | `_ensure_schema()` L406–418 |
| Behavior at millions of vectors? | **Not qualified** — live qualification soak = 50 documents, p95 threshold 5s; brute-force distance sort | `test_pgvector_live_qualification.py` L617–637 |
| Bulk ingest path? | **No** — row-by-row `INSERT` in Python loop | `add_records()` L92–119 |
| Batching? | **No** dedicated batch API | same |
| Replacement semantics? | **Yes** — `ON CONFLICT (tenant_id, namespace, workspace_id, logical_id) DO UPDATE` | L102–107; live qual replacement tests L529–572 |
| PostgreSQL FTS / BM25? | **No** on PgVector provider | No FTS SQL; no `query_hybrid` on `PgVectorRagStore` |
| `query_hybrid` / RRF on PgVector? | **No** — `VectorstoreManager.query_hybrid` requires `NativeHybridSearchProvider`; PgVector does not implement it | `hybrid_search.py`; `vectorstore_manager.py` L312–316 |

### Hybrid findings

| Topic | Finding |
| --- | --- |
| Where fusion lives | `LexicalHybridSupport.query_hybrid` (RRF dense+BM25); `fuse_graph_channels` (score-sum vector+keyword+graph); `HybridRetriever` (native hybrid or dense+token-overlap blend) |
| Algorithms | RRF (`k=60` default), weighted alpha blend, graph score-sum |
| Independent channels | GraphRAG demonstrates multi-channel with `channel_contributions`; **no** platform orchestrator for exact+lexical+structured+vector as **four independent inputs** — scenario must compose above `RetrievalService` or custom retriever |
| PgVector | Hybrid/BM25 **not** available; InMemory/Qdrant (sparse-enabled)/Weaviate **are** |
| Scenario without duplicating platform | Application can call multiple retrievers / SQL lookups and fuse using platform RRF helper or own orchestration; should **not** reimplement vector-store contracts |

### Reranker findings

| Topic | Finding |
| --- | --- |
| Contract | `BaseReranker.rerank(query, candidates, limit)` → `RerankerResult` |
| Implementations | `cross_encoder`, `cohere`, `jina`, `semantic`, `embedding_cosine`, `rrf_reranker`, `ensemble_reranker` |
| Wiring | `RetrievalService` → `RerankerManager.rerank` after retriever prefetch; gated by `RagProfile.enable_rerank` |
| Metadata survival | `RerankerCandidate` carries full `KnowledgeDocument` — product metadata in document metadata survives reranking; reranker does not add evidence fields |

### Query understanding findings

`QueryRouter` ≠ product query understanding. It selects RAG retrieval **tier** for cost/latency, not typed product constraints. Product parsing (identifiers, brand, hard specs, missing facts) is **absent** from platform — correctly classified as **application BUILD**, not platform defect.

### Structured search findings

`MetadataFilter` supports exact equality and `IN` membership. PgVector **rejects** membership filters. No numeric ranges, no multi-field boolean queries, no attribute normalization. JSON containment is not a substitute for relational product-attribute search at WDC heterogeneity scale.

### Verification findings

Decision System provides **generic** decision lifecycle and verification composition — not product identity verification. No shipped verifier consumes `{candidate, typed_constraints, catalog_evidence}` for product specs. Application must own constraint comparison logic; may optionally map terminal outcomes onto Decision System resolution envelopes for audit consistency.

### Observability findings

`RetrievalTrace` covers RAG route/retriever/rerank/latency and graph `channel_contributions`. Scenario observability contract (§ A) requires additional application-owned fields: parsed constraints, per-channel candidate counts, evidence support/contradiction per finalist, verification verdict. Platform provides hooks; scenario must emit structured diagnostics on production path.

### Proof integration findings

`PlatformProofEvidence` v3 and proof runner patterns exist (`PLATFORM_PROOF_PROTOCOL.md`, `ai_incident_investigation`). Scenario evaluator, benchmark fixtures, and evidence projection are **not** built — framework **REUSE**, scenario **BUILD**.

### Fit summary counts

| Decision | Count (matrix rows) |
| --- | ---: |
| **REUSE** | 5 |
| **CONFIGURE** | 2 |
| **EXTEND** | 5 |
| **BUILD** | 12 |
| **NOT REQUIRED** | 1 |

**Confirmed hypotheses (A–F from audit brief):**

- **A. PgVector dense-only** — **CONFIRMED**
- **B. Native hybrid/BM25 not on PgVector** — **CONFIRMED** (present on InMemory/Qdrant/Weaviate)
- **C. No HNSW/IVFFlat in PgVector schema** — **CONFIRMED**
- **D. Metadata filtering = JSON equality containment** — **CONFIRMED** (membership unsupported on pgvector)
- **E. Product query understanding absent** — **CONFIRMED**
- **F. Product identity verification = application responsibility** — **CONFIRMED**

---

## D. GAP DECISION

**Status: RESOLVED**

Frontmatter `gap_decision: RESOLVED`.

### 1. REUSE AS-IS

| Gap / need | Owner | Why | Blocks implementation? |
| --- | --- | --- | --- |
| Dense vector retrieval (qualified vector backend) | Platform (provider contracts) + application (configure) | e.g. `PgVectorRagStore`, Qdrant, InMemory — via platform vectorstore contracts; not a PostgreSQL-only requirement | No — configure provider per deployment |
| Reranker pipeline | Platform RAG | `RetrievalService` + `RerankerManager` shipped with metadata-preserving ABI | No |
| Candidate / chunk ABI | Platform RAG | `RetrievalHit` / `RerankerCandidate` / `RetrievalChunk` | No |
| Hybrid retrieval (lexical + dense) | Platform RAG | BM25+RRF via `LexicalHybridSupport` / `HybridRetriever` on backends that support it | No — provider choice; application uses contracts, not a fixed backend |
| Proof evidence framework | Platform proofs | `PlatformProofEvidence` v3, proof protocol | No — evaluator is separate build |
| RAG trace primitives | Platform RAG + runtime | `RetrievalTrace`, `rag_span` | No — extend with app fields |

### 2. SCENARIO CONFIGURATION

| Gap / need | Owner | Minimum change | Blocks? |
| --- | --- | --- | --- |
| Vector / catalog provider wiring | Application / ops | Provider-specific env and scope (e.g. `INTERGRAX_PGVECTOR_DSN` for canonical PostgreSQL+pgvector reference only) | Yes for vector channel — via chosen provider config |
| Reranker selection | Application | `RagProfile(enable_rerank=True, reranker_id=...)` | No |
| Retriever profile | Application | Choose `hybrid` vs `vector_similarity` per qualified backend capabilities | Partial — lexical channel may need separate path on some providers |

### 3. PLATFORM EXTENSION

| Gap | Why it exists | Owner | Minimum change | Blocks? | Public arch change? |
| --- | --- | --- | --- | --- | --- |
| PgVector ANN index (HNSW/IVFFlat) | Schema lacks vector index; 3.77M brute-force not credible | Platform / Integrations | Add optional index DDL + migration policy in `PgVectorRagStore._ensure_schema` | **Yes** for pgvector-at-scale proof | Yes — provider schema |
| PgVector bulk/batch ingest | Row-by-row insert inadequate for 3.77M | Platform / Integrations | `COPY`/batch insert API or ingest optimization | **Yes** for indexing SLA | Minor — provider API |
| Provider hybrid / FTS gaps | Some providers are dense-only or lack co-located lexical search | Platform / Integrations **or** application catalog indexes | Extend provider or compose separate lexical + vector backends per deployment | **Yes** only for chosen canonical reference configuration | Yes if platform-owned |
| Generic multi-channel fusion orchestrator | Today fusion is 2-channel (hybrid) or graph-specific | Platform RAG (optional) | Reusable fusion service accepting N channel result lists + attribution | No — application can fuse locally using `reciprocal_rank_fusion` | Only if promoted to generic platform capability |
| `MetadataFilter` range / rich queries | Equality-only insufficient for capacity/voltage ranges | Platform RAG contracts (optional) | Extend filter contract + provider SQL | No — application can use SQL outside vector ABI | Yes if added to portable contract |

### 4. SCENARIO / APPLICATION BUILD

| Gap | Why not platform | Owner | Minimum change | Blocks? |
| --- | --- | --- | --- | --- |
| Product query understanding | Domain-specific NL + identifier semantics | Application | Parser/LLM structured extraction → typed constraint model | **Yes** |
| Exact identifier catalog indexes | GTIN/MPN/SKU tables are catalog schema, not generic RAG | Application | Normalized identifier tables + lookup queries | **Yes** |
| Structured attribute normalization | 337k WDC attribute names — product domain | Application | Bounded normalization for high-value constraint families | **Yes** |
| Multi-channel candidate orchestration | Four channels with product-specific queries | Application | Invoke exact/lexical/structured/vector paths; fuse with attribution | **Yes** |
| Evidence extraction per finalist | Provenance to `record_json` paths | Application | Field-level evidence graph from source catalog | **Yes** |
| Identity verification engine | Material constraint logic is product-domain | Application | Support/contradict/missing per constraint; disqualify on contradiction | **Yes** |
| Terminal outcomes + clarification loop | `VERIFIED`/`AMBIGUOUS`/etc. are scenario semantics | Application | Outcome state machine; `INSUFFICIENT_INFORMATION` → user prompt | **Yes** |
| Catalog ingest / derived representations | WDC `record_json` preprocessing | Application | Deterministic builder: lexical text, normalized attrs, embeddings | **Yes** |
| Product observability contract | Stage fields beyond generic `RetrievalTrace` | Application | Structured trace projection per § A observability table | **Yes** (for proof acceptance) |
| Scenario proof evaluator + benchmarks | Hidden labels + PASS/FAIL | Application / proof package | Fixtures, evaluator, evidence projection | **Yes** (for proof) |

### 5. DEFER / NOT REQUIRED

| Item | Rationale |
| --- | --- |
| Map product outcomes into Decision System core types | Optional; generic Decision lifecycle can wrap application verdicts — not required for MVP |
| GraphRAG channel | Product catalog identification does not require knowledge graph traversal in ideal architecture |
| Perfect normalization of all 337k attribute names | Explicitly out of scope per § B limitations |
| Million-scale provider qualification before implementation | Live qual proves correctness at small scale; million-scale is benchmark calibration, not platform gate blocker |

### Implementation roadmap (proposed order)

Provider-neutral application stages — storage/retrieval backends are configured per deployment, not hard-coded:

**Implementation note (VPI-IMPLEMENTATION-1):** Provider-neutral catalog/search contracts and immutable scenario-owned domain models established under `application/domain`, `application/contracts`, `application/ports`, and `application/catalog`. No real storage provider implemented yet.

1. **Catalog runtime / storage** — immutable `record_json`, identifier tables, configuration via catalog/search **contracts**; wire qualified vector and lexical backends behind platform integration boundaries.
2. **Preprocessing / search representation** — deterministic ingest from `selected_offers.parquet`; normalized identifiers, lexical text, structured attribute subset, embeddings.
3. **Candidate channels** — application query understanding; exact identifier lookup; lexical retrieval; structured filters; dense vector via configured vector backend.
4. **Hybrid fusion + reranker** — fuse channel results with attribution (`reciprocal_rank_fusion` or score-sum); wire `RetrievalService` reranker stage on finalists.
5. **Verification** — evidence extraction from source fields; constraint support/contradiction/missing; terminal outcome selection.
6. **Clarification / outcomes** — `INSUFFICIENT_INFORMATION` loop; `AMBIGUOUS` abstention; `NO_MATCH` vs retrieval-empty distinction.
7. **Observability** — application trace schema covering § A contract; hook into `RetrievalTrace` + custom diagnostics.
8. **Benchmark / proof** — gold cases, hidden evaluator, full-corpus runs, baseline variants A–C from § B.
9. **Public reproduction** — resolve `DATASET_DISTRIBUTION`; checksum auto-resolve; documented reproduction path.

### Platform gap priority (canonical reference configuration only)

If the canonical reference deployment chooses PostgreSQL + pgvector:

1. PgVector ANN index (HNSW) — blocks credible 3.77M vector search performance on that provider.
2. Bulk ingest — blocks practical indexing time on that provider.
3. PgVector lexical/FTS hybrid — blocks co-located lexical+vector on that single-database reference path.

**Alternative reference configuration:** MySQL + Qdrant (or similar) may satisfy hybrid BM25+dense via existing `LexicalHybridSupport` without rewriting scenario application logic — operator choice, not scenario architecture.

## E. PROOF BUILD

NOT STARTED — implementation initialized (gated skeleton); domain implementation and proof wiring in progress; dataset distribution resolution pending (scenario acceptance, Intergrax Fit, and Gap Decision are complete).

Before implementation confirm: production-capable application exists; canonical path has no prohibited fake/test shortcuts; controlled providers use normal application contracts; real model boundary configured if AI behavior is material; full 3.77M catalog loaded in search infrastructure.
