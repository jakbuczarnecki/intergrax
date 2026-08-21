<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Platform Proof Report Standard v1

**Task:** PP-REPORT-1  
**Status:** Canonical (contract design only — no renderer implementation)  
**Audience:** Proof authors, report renderer implementers, external reviewers  
**Scope:** All executable Platform Proofs registered in `scripts/proof/intergrax_proof_manifest.py`

---

## 1. Purpose

Every Platform Proof execution **must** produce a rich, human-readable, self-contained **Proof Report** artifact — regardless of outcome:

| Outcome | Report required |
|---------|-----------------|
| **PASS** | Yes |
| **FAIL** | Yes |
| **BLOCKED** | Yes |
| **CRASH** | Yes (when reporter has sufficient runtime metadata) |

The report is an **evidence presentation artifact**. It is **not** a separate source of truth.

A skeptical external reviewer must be able to open the report offline and understand:

- what was being proved and why it matters
- the exact falsifiable claim and success/falsification criteria
- what was **not** being claimed
- which real systems, vendors, and components participated
- data, environment, and scenario context
- step-by-step operational evidence (not private chain-of-thought)
- evaluator reasoning and verdict
- failures, limitations, and what remains unproven
- safe reproduction instructions

**Related canon:**

| Document | Role |
|----------|------|
| [`PROOFS.md`](PROOFS.md) | Public proof dashboard — links **accepted/published** evidence only |
| [`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`](../maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md) | Public claim qualification and promotion |
| [`PLATFORM_PROOF_PROTOCOL.md`](../../platform_proofs/PLATFORM_PROOF_PROTOCOL.md) | Platform proof methodology |
| [`PLATFORM_PROOF_AUTHORING_GUIDE.md`](../../platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md) | Author workflow |

---

## 2. Architectural separation (frozen)

Keep these concepts distinct:

```text
SuiteReceipt          — repository/suite execution record (orchestrator)
ProofReceipt / evidence — machine-readable evidence for one proof execution
ProofReport           — human-readable presentation derived from evidence
```

| Layer | Canonical format | Source of truth? |
|-------|------------------|------------------|
| **SuiteReceipt** | JSON (`intergrax.proof_suite_receipt.v1`) | Suite orchestration truth |
| **Proof evidence** | JSON / typed proof evidence contracts | **Yes** — per-proof factual truth |
| **ProofReport** | Self-contained HTML | **No** — presentation only |

**Dependency chain (frozen):**

```text
RUNTIME
  ↓
TYPED EVIDENCE MODEL
  ↓
ProofReceipt / evidence JSON
  ↓
ProofReport renderer
  ↓
report.html
  ↓ optional
report.pdf
```

Rules:

1. The renderer **must not** inspect arbitrary runtime objects to reconstruct truth.
2. All factual claims in the report **must** be derivable from structured evidence produced by the proof/runtime.
3. The report **must not invent facts**.
4. The report **must not** independently promote public status, qualification, or accepted-evidence lifecycle.

Reuse existing **`SuiteReceipt`** from `scripts/proof/intergrax_proof_contracts.py`. Do **not** merge suite receipts with domain `ProofReceipt` or with report presentation models.

---

## 3. Canonical formats (frozen)

| Role | Canonical format |
|------|-------------------|
| Machine-readable evidence | **JSON** / typed proof evidence contracts |
| Human-readable report | **Self-contained HTML** (`report.html`) |
| Optional projection | **PDF** generated from the same evidence/report model |

**Markdown is not** the canonical end-user report format.

The HTML report:

- opens directly from disk via `file://`
- requires **no server**, **no CDN**, **no external CSS/JS/fonts/images**
- works **offline**
- embeds CSS inline or in-document
- embeds diagrams as inline SVG (or equivalent static markup)
- may embed **small local-only JavaScript** for UX (expand/collapse, filtering, in-page navigation) that **never fetches external resources**
- remains readable with JavaScript disabled where practical

---

## 4. Report execution status

Report **execution status** is distinct from suite status, coverage lifecycle, public evidence status, and qualification.

### 4.1 Report status vocabulary

Every report **must** display one primary execution badge:

| Status | Meaning |
|--------|---------|
| **PASS** | Claim demonstrated under named proof conditions |
| **FAIL** | Claim not demonstrated; execution completed with evaluator/runtime verdict |
| **BLOCKED** | Environment or configuration prevented meaningful execution |
| **CRASH** | Unexpected termination before normal completion; partial evidence may exist |

### 4.2 Mapping from runner `ProofStatus`

Reuse `ProofStatus` from `scripts/proof/intergrax_proof_contracts.py` where possible:

| `ProofStatus` | Report status |
|---------------|---------------|
| `PASS` | **PASS** |
| `FAIL` | **FAIL** |
| `BLOCKED_ENVIRONMENT` | **BLOCKED** |
| `BLOCKED_CONFIGURATION` | **BLOCKED** |
| `SKIPPED_PLATFORM` | **BLOCKED** (with explicit skip reason) |
| `SKIPPED_PROFILE` | **BLOCKED** (with explicit skip reason) |

**CRASH** is assigned when the proof subprocess or in-process runner terminates abnormally (uncaught exception, signal, timeout kill, partial write) **before** emitting a normal terminal evidence record. CRASH reports are **mandatory** whenever sufficient metadata exists (see §16).

### 4.3 Status fields that must not collapse

Display separately where applicable:

| Field | Example | Must not conflate with |
|-------|---------|------------------------|
| **Report execution status** | PASS | Public BOUNDED PROOF badge |
| **Coverage lifecycle** (`PLATFORM_PROOF_MAP`) | EXECUTABLE | PASS/FAIL |
| **Public evidence status** | not yet accepted | PASS |
| **Qualification** | not qualified | PASS |

---

## 5. Standard versioning

| Identifier | Value (v1) |
|------------|------------|
| **Report standard** | Platform Proof Report Standard **v1** |
| **Evidence schema version** (target, PP-REPORT-2) | `intergrax.platform_proof_evidence.v1` |
| **Report model schema version** (target, PP-REPORT-2) | `intergrax.platform_proof_report.v1` |
| **Renderer version** (when implemented) | semver string recorded in provenance |

Future renderer or schema changes **must not** silently reinterpret old evidence. Version fields in evidence and report provenance are mandatory.

---

## 6. Mandatory report sections

Every Proof Report uses the **common skeleton** below. Section IDs are stable anchors for renderer templates and domain extensions.

| § | Section ID | Required |
|---|------------|----------|
| 1 | `report-identity` | Always |
| 2 | `executive-summary` | Always |
| 3 | `claim-under-test` | Always |
| 4 | `excluded-claims` | Always |
| 5 | `architecture-under-proof` | Always |
| 6 | `participants` | Always |
| 7 | `data-environment` | When applicable |
| 8 | `scenario-overview` | When scenarios exist |
| 9 | `execution-timeline` | Always |
| 10 | `evidence-graph` | When non-trivial evidence dependencies exist |
| 11 | `final-output` | When a final model/system output exists |
| 12 | `evaluator-verdict` | Always |
| 13 | `failure-analysis` | When status ≠ PASS |
| 14 | `limitations` | Always |
| 15 | `conclusion` | Always |
| 16 | `reproduction` | Always |
| 17 | `provenance` | Always |

Domain extensions (§18) **insert after** the common skeleton section they extend, or as subsections within §9/§10 where noted.

---

### §1 Report identity (`report-identity`)

**Required fields:**

| Field | Source |
|-------|--------|
| `proof_id` | Manifest / evidence |
| Proof title | Manifest / evidence |
| Domain | Manifest / evidence |
| Execution timestamp | Evidence (`started_at` / equivalent) |
| **Execution status** | Derived verdict — badge: PASS / FAIL / BLOCKED / CRASH |
| Source revision / commit SHA | Evidence |
| Git dirty flag | Evidence when available |
| Proof version | Proof-owned semantic version if declared |
| Evidence schema version | Evidence contract |
| Report schema version | Report model |
| Execution profile | `quick` / `full` / `live` |
| Environment / platform | Evidence |
| Report generation timestamp | Renderer |

Status badge must be visually dominant and consistent across all reports.

---

### §2 Executive summary (`executive-summary`)

Short prose (target: 120–250 words) covering:

- what was tested
- result (PASS/FAIL/BLOCKED/CRASH)
- most important finding
- strongest evidence pointer (section + evidence ID)
- primary limitation

Must be understandable without reading technical sections.

---

### §3 Claim under test (`claim-under-test`)

| Subfield | Required |
|----------|----------|
| Exact falsifiable platform claim | Yes |
| User / business relevance | Yes |
| Success criterion | Yes |
| Falsification criterion | Yes |

Wording must align with proof documentation under `platform_proofs/` and must not exceed manifest scope.

---

### §4 What this proof does not prove (`excluded-claims`)

**Mandatory.** Explicit out-of-scope list preventing claim inflation.

Minimum categories to address when applicable:

- production readiness
- commercial validation
- real-user validation
- universal provider / vendor compatibility
- all workloads or deployment modes
- product-specific workflows (for platform proofs)

Use concrete bullets — not generic marketing disclaimers alone.

---

### §5 Architecture under proof (`architecture-under-proof`)

Show the **real execution path** with participant class on each node:

| Class | Meaning |
|-------|---------|
| **REAL boundary** | Live external or platform boundary exercised |
| **CONTROLLED fixture** | Deterministic input/fixture (not substituting mechanism under proof) |
| **PROOF-owned component** | Code under `platform_proofs/` |
| **PLATFORM component** | Intergrax runtime / adapter / tool infrastructure |
| **EXTERNAL vendor** | Third-party provider or infrastructure |

Include an **inline diagram** (SVG recommended): architecture flow for the proof.

**Reference example — TOOLS-ITERATIVE-SQL-INVESTIGATION:**

```text
OpenAI model (EXTERNAL vendor, REAL)
  → LLM adapter (PLATFORM)
  → ToolPlanningService (PLATFORM)
  → bounded tool loop (PLATFORM)
  → RuntimeToolInvoker (PLATFORM)
  → platform_proof.sql.query tool (PROOF-owned, REAL SQL boundary)
  → PostgreSQL (EXTERNAL, REAL, CONTROLLED fixture dataset)
  → InvestigationProof / ToolCallTrace evidence (PLATFORM)
  → proof evaluator (PROOF-owned)
```

---

### §6 Participants / components (`participants`)

Structured table:

| Column | Description |
|--------|-------------|
| Component | Logical name |
| Implementation / vendor | e.g. OpenAI, PostgreSQL 16, Intergrax Nexus runtime |
| Version / model | Named version or model ID |
| Role | In execution path |
| Real / mock / fixture | Status |
| Relevant configuration | Non-secret config only |

---

### §7 Data / environment (`data-environment`)

When applicable:

| Topic | Content |
|-------|---------|
| Dataset identity | Name, scenario version |
| Row count / scale | Bounded counts |
| Seed / generation | Deterministic seed if used |
| Fingerprint / checksum | e.g. SHA-256 |
| Infrastructure identity | Docker image, DSN host/port pattern (no secrets) |
| Database role | read-only vs read-write |
| Ground-truth validation | What proof verified independently |

**Explicit distinction (required when both apply):**

| Bucket | Meaning |
|--------|---------|
| **Ground truth known to proof** | Facts verified by setup/evaluator independent of model |
| **Information available to model** | Only what runtime exposed via tools/messages |

---

### §8 Scenario overview (`scenario-overview`)

Per scenario:

| Field | Description |
|-------|-------------|
| Scenario ID | Stable ID |
| Question / problem | Investigative prompt |
| Expected behavior | PASS path |
| Negative / falsification condition | What would fail the claim |
| Result | pass / fail / not reached |
| Evidence count | Tool calls, proof steps, etc. |
| Relevant metrics | e.g. successful_tool_calls, investigation_proof_steps |

---

### §9 Execution timeline / trace (`execution-timeline`)

**Major required section.** Step-by-step operational evidence.

**Generic step record:**

| Field | Description |
|-------|-------------|
| STEP | Monotonic step index |
| PURPOSE | Operational intent (not private reasoning) |
| EVIDENCE_BASIS | Prior evidence IDs this step depends on |
| ACTION | What happened |
| INPUT | Bounded safe input summary |
| OBSERVATION | Bounded observable result |
| EVIDENCE_CREATED | New evidence IDs |
| STATUS | ok / fail / skipped |

**Agent / tool proofs — preferred column flow:**

```text
PURPOSE → EVIDENCE_BASIS → TOOL ACTION → TOOL ARGUMENTS → TOOL RESULT → NEW EVIDENCE
```

**Frozen rule:**

> **Operational decision trace ≠ chain-of-thought.**

Include:

- tool names, bounded arguments, bounded results
- evidence basis chain (`InvestigationProof` / `ToolCallTrace` semantics)
- retry/limit behavior when relevant
- termination reason (`stop_reason`)

Do **not** expose hidden reasoning, scratchpad, or provider-internal thought fields.

---

### §10 Evidence graph (`evidence-graph`)

When evidence dependencies are non-linear, render a **DAG** (inline SVG):

```text
Question → ToolCall-1 → Evidence-1 → ToolCall-2 → Evidence-2 → Final Answer
```

No external graph service. Graph nodes reference evidence IDs used elsewhere.

---

### §11 Final output (`final-output`)

Show the **final public model/system output** used for evaluation.

For LLM proofs: final answer text only — not hidden reasoning.

---

### §12 Evaluator verdict (`evaluator-verdict`)

List **explicit checks** with pass/fail and evidence pointers.

Avoid opaque standalone **PASS**.

Example pattern (TOOLS reference):

| Check | Result | Evidence |
|-------|--------|----------|
| Minimum successful tool-call count | ✓ | timeline §9 |
| Valid evidence-dependent follow-up (ENG-6 chain) | ✓ | InvestigationProof steps |
| Expected anomaly discovered (scenario A) | ✓ | SQL + final answer |
| Unsupported causal claim rejected | ✓ | evaluator patterns |
| Normal termination | ✓ | stop_reason |

---

### §13 Failure analysis (`failure-analysis`)

**Mandatory when status ≠ PASS.**

| Subfield | Content |
|----------|---------|
| Execution boundary reached | Last completed milestone |
| Completed milestones | Checklist |
| Failed milestone | First failure point |
| Exception / provider error | Redacted safe message |
| Failure classification | See table below |
| Not exercised | What stopped before running |

**Failure classifications** (align with proof/runtime conventions):

| Class | Use when |
|-------|----------|
| `PLATFORM_DEFECT` | Platform mechanism violated invariant |
| `PROOF_DEFECT` | Proof harness/evaluator error |
| `MODEL_BEHAVIOR_FAILURE` | Model output/behavior failed criterion |
| `PROVIDER_CONFIGURATION` | Provider misconfiguration |
| `PROVIDER_UNAVAILABLE` | Provider unreachable / auth |
| `ENVIRONMENT` | Infra/env missing or unhealthy |
| `EXPECTED_FALSIFICATION` | Negative scenario correctly failed claim |
| `BLOCKED_CONFIGURATION` | Config gate blocked run |
| `BLOCKED_ENVIRONMENT` | Env gate blocked run |
| `TIMEOUT` | Execution time limit |
| `CRASH` | Abnormal termination |
| `UNKNOWN` | Unclassified — use sparingly |

**Progress checklist example (CRASH / BLOCKED):**

```text
✓ dataset prepared
✓ database verified
✓ adapter constructed
✗ provider request rejected
○ model response
○ tool execution
○ evaluator
```

---

### §14 Limitations (`limitations`)

Execution-specific limitations — not boilerplate alone.

Examples: single model/provider, single OS run, bounded row cap (`MAX_VISIBLE_ROWS`), named Docker profile.

---

### §15 Conclusion (`conclusion`)

Human-readable synthesis:

- what evidence supports
- what was falsified or rejected
- what remains open
- whether qualification implications exist (**without** changing public status)

Report **must not** independently promote public or qualified status.

---

### §16 Reproduction (`reproduction`)

Safe reproduction only:

| Item | Include |
|------|---------|
| Source revision | Full SHA |
| Command | Canonical manifest command |
| Profile | `quick` / `full` / `live` |
| Non-secret env vars | Names only or placeholder patterns |
| External dependencies | Docker, provider account, etc. |
| Dataset fingerprint | When relevant |

**Never** include secrets (see §19).

---

### §17 Provenance (`provenance`)

| Field | Required |
|-------|----------|
| `proof_id` | Yes |
| Source SHA | Yes |
| Evidence schema version | Yes |
| Report schema version | Yes |
| Renderer version | When implemented |
| Evidence artifact path / checksum | When practical |
| Report artifact checksum | When practical |
| Generation timestamp | Yes |
| Suite run ID | When suite-orchestrated |

---

## 7. Domain extension mechanism

All reports use the **common skeleton** (§6). Domains **may** add sections via registered extension IDs:

```text
domain_extension_id: "<domain>.<section>"
parent_section: "<common-section-id>"   # optional anchor
title: "<Human title>"
required: true | false
```

Extensions **must not** replace common sections. They add typed subsections fed only from evidence fields declared safe for reporting.

### 7.1 Registered extension profiles (v1 — design only)

| Domain | Extension ID | Adds |
|--------|--------------|------|
| **TOOLS** | `tools.tool-call-trace` | Tool calls, arguments, observations, evidence basis, invocation result, retry/limit behavior |
| **RAG** | `rag.retrieval-trace` | Documents, chunks, ranking, citations, grounding |
| **MEMORY** | `memory.state-trace` | Writes, reads, recall, temporal state, mutation history |
| **GOVERNED_EXECUTION** | `governed.action-trace` | Requested action, policy decision, authority, approval, enforcement |
| **OBSERVABILITY** | `observability.correlation-trace` | Event timeline, task/run/attempt IDs, correlated traces, failure propagation |
| **CONTEXT_ENGINEERING** | `context_engineering.assembly-trace` | Sources, selection, token budget, compaction, final context |

Renderer implementations for domain extensions are **out of scope** for PP-REPORT-1.

---

## 8. Visual design standard

Common Intergrax report visual language (renderer requirements):

| Requirement | Specification |
|-------------|---------------|
| Layout | Single-column primary flow; max readable width ~960px; printable `@media print` rules |
| Hierarchy | H1 report title → H2 sections → H3 subsections; consistent numbering matching §6 |
| Status badges | PASS green, FAIL red, BLOCKED amber, CRASH dark red — with accessible contrast (WCAG AA target) |
| Key facts | Card components for identity, status, claim summary |
| Tables | Full-width responsive tables; zebra rows optional |
| Timeline | Vertical timeline with step markers and status icons |
| Diagrams | Inline SVG; monochrome + accent color |
| Code / SQL | Syntax-highlighted via embedded CSS classes (no external highlighter) |
| Evidence IDs | Monospace badge style (`evidence-id`, `tool-call-id`) |
| Expandable detail | `<details>`/summary or JS toggle for raw evidence |
| Tone | Technical evidence — **not** marketing hero sections |

Evidence readability **>** decoration.

---

## 9. Self-contained HTML requirements (checklist)

- [ ] Opens via `file://`
- [ ] No server required
- [ ] No CDN assets
- [ ] No external JavaScript
- [ ] No external CSS
- [ ] No remote fonts (system font stack only)
- [ ] No remote images
- [ ] All CSS embedded
- [ ] Diagrams inline (SVG)
- [ ] Optional embedded JS: local UX only, no network
- [ ] Readable with JS disabled for core content

---

## 10. Diagram policy

Generate diagrams only where they aid review:

| Diagram | When |
|---------|------|
| Architecture flow | §5 |
| Execution timeline | §9 (optional visual duplicate of table) |
| Evidence graph | §10 |
| Scenario flow | §8 with multiple branches |
| Failure boundary | §13 |
| Data-flow summary | §7 |

Prefer **generated inline SVG**. Do **not** require Mermaid or CDN at view time.

---

## 11. Reporting on failure and crash

**Critical rule:** A proof **CRASH** or **BLOCKED** run **must still produce a report** whenever the reporting pipeline has enough runtime metadata.

Design requirements (implementation: PP-REPORT-5):

1. Evidence emission should be **incremental** where feasible (setup, env checks, partial traces).
2. Reporter hooks run on `finally` paths — not only on PASS.
3. Missing sections render as **not reached** with explicit checklist state — not omitted silently.
4. A failure report is **valuable evidence** for operators and external reviewers.

---

## 12. Redaction and security (frozen)

### 12.1 Never include

- API keys, tokens, passwords
- Authorization headers
- Secret-bearing connection strings
- Full environment dumps
- Private chain-of-thought / hidden reasoning
- Unredacted PII unless explicitly part of bounded public fixture design

### 12.2 Redaction expectations

| Surface | Rule |
|---------|------|
| Tool inputs/outputs | Bounded size; declare `report_safe` in evidence schema |
| URLs | Strip query secrets; truncate long paths |
| Headers | Never emit auth headers |
| Environment/config | Names and non-secret values only |
| Provider errors | Provider-safe message; no embedded credentials |

### 12.3 Fail-closed

Evidence fields without explicit **`report_safe: true`** (or domain equivalent) **must not** appear in human report raw sections. Unknown secret-bearing fields: **redact or omit** — never guess.

Typed evidence contracts (PP-REPORT-2) **must** declare report safety per field or per section.

---

## 13. Raw evidence presentation

Pattern: **SUMMARY + BOUNDED RAW EVIDENCE**

| Layer | Content |
|-------|---------|
| Summary | Tables, counts, verdict lines |
| Raw (expandable) | Exact SQL, safe tool args, bounded tool results, safe errors, final answer, evaluator checks |

Rules:

- Do not truncate so aggressively that independent inspection is impossible.
- Cap row/character limits per field in evidence schema (e.g. align with `output_preview`, `MAX_VISIBLE_ROWS`).
- No uncontrolled dumps.

---

## 14. Artifact naming and location

Generated artifacts **must not** be committed to source-controlled docs by default.

### 14.1 Canonical layout (target — PP-REPORT-6)

Align with existing suite receipt directory `.artifacts/proof/` (`intergrax_proof_runner.py`):

```text
.artifacts/proof/<suite-run-id>/
  suite-receipt.json
  proofs/
    <PROOF_ID>/
      evidence.json
      report.html
      report.pdf          # optional — PP-REPORT-7
```

| Artifact | Naming |
|----------|--------|
| Suite directory | `<suite-run-id>` UUID from `SuiteReceipt.suite_run_id` |
| Per-proof directory | Uppercase manifest `proof_id` (e.g. `TOOLS-ITERATIVE-SQL-INVESTIGATION`) |
| Evidence | `evidence.json` |
| Report | `report.html` |
| PDF | `report.pdf` |

**Transitional note:** Current runner writes flat `suite-receipt` files as `{timestamp}-{profile}-{short-sha}.json`. PP-REPORT-6 migrates to hierarchical layout without changing proof semantics.

### 14.2 Versioning in filenames

Prefer directory + stable inner filenames over encoding versions in every filename. Schema version lives **inside** JSON and report provenance.

---

## 15. Publication and lifecycle model

Three report classes:

| Class | Definition | Default visibility |
|-------|------------|-------------------|
| **Generated report** | Every proof run | Local `.artifacts/` only |
| **Accepted report** | Reviewed / qualified evidence per maintainer workflow | Internal promotion candidate |
| **Published report** | Intentionally exposed via docs / public dashboard | Linked from [`PROOFS.md`](PROOFS.md) when accepted |

Rules:

- Do **not** auto-publish every local run.
- [`PROOFS.md`](PROOFS.md) links **accepted/published** reports only — not all generated artifacts.
- Generated HTML does **not** change public claim status.

Promotion workflow remains governed by [`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`](../maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md).

---

## 16. Reference example — TOOLS-ITERATIVE-SQL-INVESTIGATION

**Proof ID:** `TOOLS-ITERATIVE-SQL-INVESTIGATION`  
**Manifest:** `scripts/proof/intergrax_proof_manifest.py`  
**Entrypoint:** `platform_proofs/tools/iterative_sql_investigation/run_proof.py`

### 16.1 Claim (illustrative outline)

> The bounded iterative tool runtime can use **real SQL observations** from **PostgreSQL** via a **real LLM provider** to drive subsequent **evidence-dependent** tool calls, preserve an explicit **InvestigationProof** chain, and reach a **bounded conclusion** while rejecting unsupported causal claims.

### 16.2 Participants (illustrative)

| Component | Vendor | Role | Status |
|-----------|--------|------|--------|
| LLM | OpenAI (env-configured) | Planning / answers | REAL |
| LLM adapter | Intergrax | Provider boundary | PLATFORM |
| Tool loop | Intergrax Nexus | Bounded iteration | PLATFORM |
| SQL tool | `platform_proof.sql.query` | Read-only SELECT | PROOF-owned REAL |
| PostgreSQL | Docker | Fixture dataset | REAL + CONTROLLED fixture |
| Evaluator | Proof-local | Scenario invariants | PROOF-owned |

### 16.3 Scenario sketch

| ID | Intent | Falsification |
|----|--------|---------------|
| A | Find anomalous segment; reject volume-only explanation | Volume-only root cause accepted |
| B | Detect association; verify segmented evidence; **no** direct causation claim | Direct causation asserted |
| C | No staffing data — must report missing evidence | Staffing cause invented |

Evidence sources today: `ToolsSqlInvestigationProofResult`, `ScenarioRunResult`, `ToolCallTrace`, `InvestigationProof`, `ScenarioExecutionSnapshot` — future **`evidence.json`** consolidates these for reporting (PP-REPORT-2).

### 16.4 Execution timeline excerpt (PASS — scenario A)

| STEP | PURPOSE | ACTION | STATUS |
|------|---------|--------|--------|
| 1 | Prepare deterministic dataset | Materialize + verify fingerprint | ok |
| 2 | Verify DB | Row counts / anomaly hub stats | ok |
| 3 | Construct adapter | Resolve `INTERGRAX_LLM_*` | ok |
| 4 | Run bounded tool loop | SQL queries via real provider | ok |
| 5 | Record InvestigationProof | ENG-6 basis chain | ok |
| 6 | Evaluate scenario A | Pattern + invariant checks | ok |

### 16.5 Evaluator verdict excerpt (PASS)

- ✓ `successful_tool_calls` ≥ minimum  
- ✓ `investigation_proof_passes_eng6_chain`  
- ✓ North anomaly identified; volume-only explanation rejected  
- ✓ `stop_reason` ∈ successful termination set  

### 16.6 Failure example outline (FAIL — scenario B)

| STEP | STATUS |
|------|--------|
| Dataset + DB verify | ok |
| Tool calls return segmented evidence | ok |
| Final answer asserts direct causation | **fail** |
| Evaluator: `claims_direct_causation=true` | **FAIL** (`MODEL_BEHAVIOR_FAILURE`) |

### 16.7 BLOCKED example outline

| Condition | Report status |
|-----------|---------------|
| Missing `INTERGRAX_LLM_PROVIDER` | **BLOCKED** (`BLOCKED_CONFIGURATION`) |
| Docker unavailable | **BLOCKED** (`BLOCKED_ENVIRONMENT`) |
| Provider auth failure | **BLOCKED** or **CRASH** depending on termination path |

Evidence struct: `ToolsSqlInvestigationProofResult.blocked(...)` — report still generated with §13 checklist.

---

## 17. Implementation roadmap

| Task | Deliverable |
|------|-------------|
| **PP-REPORT-1** | Report Standard v1 (this document) |
| **PP-REPORT-2** | Typed evidence / report contract (`evidence.json` schema) |
| **PP-REPORT-3** | Generic self-contained HTML renderer |
| **PP-REPORT-4** | TOOLS proof integration |
| **PP-REPORT-5** | Crash / failure report generation |
| **PP-REPORT-6** | Generic proof runner integration + artifact layout |
| **PP-REPORT-7** | Optional PDF projection |
| **PP-5** | Accepted / public evidence publication |

**PP-REPORT-1 non-goals:** no HTML renderer, no PDF, no runtime semantic changes, no qualification changes, no frontend, no auto-publish.

---

## 18. Non-goals (explicit)

Do **not** (in PP-REPORT-1 and unless a later task says otherwise):

- implement HTML renderer or PDF generation
- change Platform Proof runtime semantics
- change proof qualification or public status rules
- build web frontend / hosting
- add React, Vue, or external SaaS reporting
- alter TOOLS proof semantics
- expose chain-of-thought
- auto-publish generated reports

---

## 19. Document history

| Version | Task | Notes |
|---------|------|-------|
| v1 | PP-REPORT-1 | Initial canonical standard — contract design only |
