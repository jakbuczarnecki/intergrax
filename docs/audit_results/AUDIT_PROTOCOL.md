# Intergrax Platform Audit Protocol v2

**Status:** Canonical  
**Version:** 2.0  
**Audience:** Human operators and model executors (any harness, any IDE)  
**Scope:** Full Intergrax platform — Tier-0 `intergrax/`, Tier-1 `intergrax/runtime/`, Tier-2 `agents/`, Tier-3 `applications/`

---

## Core mindset

> **Your job is not to confirm that Intergrax is well designed. Your job is to try to prove that it is not.**

**Falsification principles (brutal and neutral):**

- Assume every material claim is **UNVERIFIED** until independently examined.
- Actively search for counterexamples and failure modes.
- There is no reward for PASS and no reward for FAIL.
- Never manufacture, inflate, or preserve a finding merely to satisfy the adversarial posture.
- If a hypothesis cannot be supported with evidence, downgrade it to an open question or discard it.
- A clean **PASS** backed by a serious falsification attempt is a valid audit outcome.

A prior PASS, a green CI run, or polished architecture prose does not reduce skepticism.

This protocol is **model-executable**: an agent following it must produce the same artifacts, evidence standards, and verdict discipline regardless of tooling. It is **not** tied to Cursor, any specific orchestrator, `progress.json`, or deleted harness machinery.: an agent following it must produce the same artifacts, evidence standards, and verdict discipline regardless of tooling. It is **not** tied to Cursor, any specific orchestrator, `progress.json`, or deleted harness machinery.

---

## A. Audit purpose

An Intergrax platform audit is:

1. **Adversarial** — actively search for ways the system fails, lies, drifts, or only works by accident.
2. **Independent** — do not inherit optimism from authors, prior chats, or roadmap status (`Done`, `PASS`, milestone claims).
3. **Unincentivized toward PASS** — no quota, no desire to close quickly, no credit for a clean narrative. A FAIL with strong evidence is a successful audit.
4. **Defect- and assumption-focused** — primary output is falsified claims, hidden coupling, and false assumptions about behavior under production stress.
5. **Evidence-bound** — opinions without inspectable proof are hypotheses, not findings.

Audits do **not** implement fixes, refactor for style, or "helpfully" patch issues discovered mid-run (see section L).

---

## B. Source-of-truth hierarchy

When sources conflict, resolve in this order (highest wins):

| Rank | Source | Role |
|------|--------|------|
| **1** | **Implementation at exact SHA** | What the system actually does at audit time. Pin `git rev-parse HEAD` (or named tag/commit) in every campaign artifact. |
| **2** | **Tests and CI** | **Evidence, not proof.** Tests show what was exercised; they do not prove completeness, production safety, or architectural compliance. Missing or weak tests are themselves findings. |
| **3** | **Architecture docs** (`docs/project/architecture/`, capability architecture hubs) | **Target state** — design intent and invariants. Deviations are findings unless explicitly documented as accepted gaps. |
| **4** | **Plan docs** (`docs/project/maintainers/plans/`, capability plan hubs) | **Work claims** — what maintainers assert is built, deferred, or verified. Treat `Done` as a claim to falsify, not a fact. |
| **5** | **Prior audit artifacts** (`docs/audit_results/`) | **Historical only** — context for regression and fixed defects. Never authoritative over current implementation. |

**Rule:** If architecture says X and code does Y, record a finding unless a dated, explicit exception exists in architecture or plan with operator acceptance.

---


## B1. Architecture as auditable claim

Architecture (`docs/project/architecture/`, capability architecture hubs) is a **target-state source of truth for implementation**, but architecture itself is **also an auditable claim**.

The audit must not stop at "code matches architecture." Separately ask:

- Is the target architecture production-grade?
- Are the abstractions correct?
- Are responsibilities in the correct layer?
- Are dependency directions appropriate?
- Are stated guarantees realistically enforceable?
- Is the design safe under failure, retry, restart, concurrency, and multi-host operation?
- Does it support reusable platform mechanisms rather than current-product special cases?
- Would a second independent application reuse the mechanism cleanly?
- Does the design create unnecessary parallel universal mechanisms?
- Are important production invariants absent from the architecture?
- Is a documented abstraction only paper architecture?
- Are industry-standard mature production patterns materially stronger for this problem?

**Finding kinds (architecture vs implementation):**

| Kind | Meaning |
|------|---------|
| **IMPLEMENTATION DEFECT** | Code contradicts a valid architecture target or invariant. |
| **ARCHITECTURE DEFECT** | Target design is wrong, incomplete, or not production-grade. |
| **IMPLEMENTATION/ARCHITECTURE DRIFT** | Code and architecture disagree; ownership of the gap must be explicit. |

Do not treat canonical architecture prose as automatically correct. External comparison is evidence-driven when materially useful; do not perform internet research merely for decoration.


## C. Campaign model

An audit **campaign** is a **longitudinal audit sequence** across one or more layers. Each **layer audit** is an **immutable snapshot** pinned to an exact repository SHA.

After each accepted layer the canonical workflow is:

```
audit → persist result → sync architecture → sync implementation plan → commit → next layer
```

Therefore `development` may advance between layer audits. Do **not** model a multi-layer campaign as audited against a single pinned SHA.

### Campaign identity

- **Campaign date:** `YYYY-MM-DD` (UTC calendar date when the campaign **starts**).
- **Campaign root:** `docs/audit_results/YYYY-MM-DD/` (see same-day naming below).
- **Global registry:** `docs/audit_results/README.md` — protocol entry point, global campaign registry, latest-campaign discovery, legacy pointer.

### Campaign directory layout

```text
docs/audit_results/
  README.md
  AUDIT_PROTOCOL.md
  AUDIT_REMEDIATION_PROTOCOL.md
  YYYY-MM-DD/
    README.md              # REQUIRED — campaign master register AND rollup
    <LAYER_CODE>.md        # immutable per-layer audit snapshot
    ...
  legacy/
    ...
```

**Do not** require a separate `CAMPAIGN_SUMMARY.md`. The campaign `README.md` is the master register and campaign rollup.

### Same-day campaigns

Use **one** convention only:

- `YYYY-MM-DD`
- `YYYY-MM-DD_run-2`
- `YYYY-MM-DD_run-3`

Do **not** use `YYYY-MM-DD-a` / `-b` suffixes.

### Required campaign metadata (in campaign `README.md`)

| Field | Description |
|-------|-------------|
| `campaign_id` | Dated directory name (e.g. `2026-08-18` or `2026-08-18_run-2`) |
| `started_at` | UTC timestamp when campaign started |
| `completed_at` | UTC timestamp when campaign closed (or null while in progress) |
| `status` | `IN_PROGRESS` \| `COMPLETE` \| `ABORTED` |
| `campaign_start_sha` | Repository SHA at campaign start |
| `campaign_end_sha` | Final repository state after campaign-owned documentation synchronization |
| `scope` | Layers/domains in scope |
| `overall_verdict` | Campaign rollup verdict (section J) |

Legacy material is identified by location under `legacy/`, not by a competing active campaign status.

### Required per-layer register metadata (in campaign `README.md` layer table)

| Field | Description |
|-------|-------------|
| `layer` | Layer code (e.g. `MEMORY`, `ORCHESTRATION`) |
| `status` | Layer audit status |
| `audited_sha` | **Exact** SHA this layer was audited against (immutable once published) |
| `verdict` | Layer verdict (section J) |
| `finding_count` | Count by severity |
| `architecture_sync` | Whether arch sync completed for accepted findings |
| `plan_sync` | Whether plan sync completed for accepted findings |
| `post_sync_sha` | Commit SHA after arch/plan sync (when synchronization created a later commit) |

Every per-layer result file MUST record the same `audited_sha`. Evidence in that result refers to **that** SHA only.

**Do not** retroactively change `audited_sha` when architecture/plan sync advances `development`.

At campaign completion, `campaign_end_sha` records the final repository state after campaign-owned documentation synchronization.

### Concurrent changes between layer audits

If runtime/product code changes concurrently between layer audits:

- Audit the next layer against the **then-current** exact SHA; record it explicitly.
- If concurrent changes invalidate already-audited findings, flag the affected layer for **revalidation** rather than pretending the whole campaign used one immutable source tree.

### Lifecycle

1. Create dated folder + campaign `README.md` with `IN_PROGRESS` and `campaign_start_sha`.
2. Execute layers per operator roadmap (section P); persist each layer at its `audited_sha`.
3. After each accepted layer: arch/plan sync → commit → record `post_sync_sha` in campaign register.
4. Set campaign `COMPLETE` when scoped layers finish and rollup is written in campaign `README.md`, or `ABORTED` with reason.

**No dependency** on `progress.json`, orchestrator scripts, or IDE-specific state files.


## C1. Context and read discipline

- Audit **one layer** as the normal atomic unit.
- Begin from its architecture/plan ownership pair and known code entrypoints.
- Use path-filtered textual search before broad exploration.
- Follow concrete call paths and evidence.
- Do **not** run repo-wide semantic searches by default.
- Do **not** load all architecture/plan docs for all layers.
- Do **not** run full-suite tests when targeted tests or gates are sufficient.
- Expand context only when a concrete hypothesis or cross-layer dependency requires it.
- When expanding materially, record **why** in the layer file.

Accuracy outranks token savings, but targeted evidence outranks bulk reading.


## D. Layer-by-layer discipline

Intergrax is audited **one layer (domain) at a time** unless the operator explicitly scopes a cross-layer slice.

Per layer:

1. **Declare scope** — layer code, tier, paths, architecture/plan doc pair, in-scope/out-of-scope boundaries.
2. **Pin SHA** — all evidence references this commit.
3. **Load hierarchy** — read implementation first; use arch/plan as falsification checklists, not as ground truth.
4. **Execute falsification pass** — systematically attack section E targets within scope.
5. **Record findings** — each with ID, severity, category, evidence (sections H–I).
6. **Assign layer verdict** (section J).
7. **Present to operator** — no silent persistence of findings (section P).
8. **After acceptance** — persist artifacts, then arch/plan sync if required (section M).

Do not begin layer N+1 until layer N is `COMPLETE` or explicitly deferred in the campaign README with operator sign-off.

**Tier discipline (import boundaries):**

- Tier-0 `intergrax/` MUST NOT import from `agents/` or `applications/`.
- Tier-2 `agents/` MUST NOT import from `applications/`.
- Violations are always at least **HIGH** unless proven unreachable dead code with evidence.

---

## E. Falsification targets

For each in-scope component, attempt to disprove stated invariants. At minimum, probe:

### Architecture and structure

- **Architecture bypasses** — code paths that skip documented gates, policies, or lifecycle stages.
- **Alternate paths** — feature flags, env toggles, legacy branches, "admin" or debug entrypoints that change behavior.
- **Dependency violations** — cross-tier imports, circular deps, runtime plugin loading that breaks boundaries.
- **Hidden global state** — module-level singletons, process-wide caches, class variables, thread-locals used as implicit channels.
- **Product leakage** — application-specific logic in runtime/agents tiers; domain logic in wrong tier.
- **Host-specific behavior** — paths, shells, OS assumptions, hardcoded developer machine layout.
- **False portability** — claims of cloud-agnostic or multi-host operation contradicted by local-only locks, paths, or APIs.
- **Legacy paths** — deprecated modules still reachable; dual implementations where only one is documented.
- **Doc drift** — architecture/plan describes behavior or APIs that code no longer implements.

### Reliability and correctness

- **Fail-open** — errors swallowed, defaults that grant access, silent fallbacks, "best effort" that hides failure.
- **Restart / resume** — state after crash, partial writes, checkpoint replay, stale checkpoints.
- **Retries** — unbounded retry, retry without idempotency, thundering herd, duplicate side effects.
- **Idempotency** — duplicate delivery, at-least-once semantics without dedup keys.
- **Concurrency** — races, TOCTOU, missing locks, async boundaries, shared mutable state.
- **Multi-host** — split brain, non-distributed locks, file-based coordination on shared FS assumptions.
- **Durability** — fsync, atomic rename, WAL gaps, "write succeeded" before durable.
- **Failure / recovery** — partial failure leaves system inconsistent; no reconciliation path.

### Security and identity

- **Identity** — conflation of user, tenant, session, job, agent instance; missing propagation across layers.
- **Authorization** — checks only at edge; missing downstream enforcement; trust of caller-supplied IDs.
- **Security by convention** — "internal only", "not exposed", "trusted network" without enforcement.
- **Secrets** — logging, env dumps, error messages, test fixtures, committed placeholders treated as real.

### Operations and scale

- **Observability gaps** — failures without correlation IDs, missing metrics on critical paths, undebuggable async.
- **Scaling limits** — O(n) scans, unbounded queues, single-process bottlenecks presented as scalable.
- **Unbounded state** — caches, logs, queues, DB tables, in-memory maps without TTL or cap.
- **Cost / budget** — unbounded LLM/tool calls, missing limits, retry storms amplifying cost.

### Contracts and typing

- **Weak contracts** — undocumented JSON shapes, optional fields that change semantics, version skew.
- **`dict[str, Any]`** (and equivalent) — untyped payloads crossing trust boundaries without validation.
- **Reflection / `getattr` / dynamic dispatch** — behavior depends on string names; bypasses static analysis and policy hooks.
- **Hidden dependencies** — import side effects, registry mutation at import time, implicit plugin discovery.

### Verification quality

- **"Done" without behavior** — plan marks complete but no implementation or no observable effect.
- **Tests missing invariants** — critical properties never asserted (authz, idempotency, failure modes).
- **Paper abstractions** — interfaces with single no-op implementation; layers that delegate without adding enforcement.
- **Missing negative tests** — only happy path; no tests for denial, timeout, malformed input, crash mid-flight.
- **Cross-layer mismatches** — tier A assumes tier B provides guarantee B does not implement.

Record **negative results** (target examined, not falsified) briefly in the layer file — they bound confidence but are not findings.

---

## F. Production perspective

Evaluate as if deployed tomorrow under adversarial conditions. Ask:

| Stress | Questions |
|--------|-----------|
| **Failures** | What breaks first? Is failure contained? Is it visible? |
| **Restart** | What is lost, duplicated, or corrupted after process/host restart? |
| **Duplicate delivery** | Messages, webhooks, tool calls, file events — what happens twice? |
| **Retries** | Who retries, with what backoff, and what if the first attempt partially succeeded? |
| **Parallel execution** | Same resource mutated concurrently — last write wins? corruption? |
| **Partial outage** | DB up, queue down; API up, auth down — degrade safe or fail catastrophically? |
| **Multi-tenant** | Can tenant A read, trigger, or exhaust tenant B's resources? |
| **Scale** | 10x jobs, 100x files, long-running campaigns — what exhausts memory, FDs, API quotas? |
| **Malicious input** | Oversized payloads, path traversal, injection, prompt injection at tool boundaries. |
| **Policy denial** | When policy says no, is denial enforced at every layer or only logged? |
| **Timeouts** | Hung calls, partial results, cancel propagation. |
| **Drift** | Config, schema, feature flags, dependency versions — runtime vs documented. |

Lack of evidence for production-safe behavior under these stresses is a finding when the architecture claims that property.

---

## G. Finding classification

### Severity

| Level | Definition |
|-------|------------|
| **CRITICAL** | Exploitable security issue, data loss/corruption, cross-tenant breach, or production outage likely under normal operation. |
| **HIGH** | Serious correctness/reliability flaw, boundary violation, or fail-open with material impact; workaround difficult. |
| **MEDIUM** | Real defect or risk with limited blast radius, missing enforcement on non-critical path, or significant test/doc gap. |
| **LOW** | Minor inconsistency, maintainability hazard, weak typing on internal-only path, cosmetic doc drift. |

### Categories (use one primary)

- **IMPLEMENTATION DEFECT** — implementation contradicts specified behavior or invariants.
- **ARCHITECTURE DEFECT** — target design is wrong, incomplete, or not production-grade.
- **IMPLEMENTATION/ARCHITECTURE DRIFT** — code and architecture disagree; gap ownership must be explicit.
- **BOUNDARY VIOLATION** — tier/import/product-surface rule broken.
- **SECURITY** — authn/authz/secrets/trust-boundary issue.
- **RELIABILITY** — crash, duplication, lost work, non-idempotent side effects.
- **OPERABILITY** — observability, deployability, config, runbooks inadequate for claimed ops model.
- **SCALABILITY / COST** — unbounded resource use or economic risk.
- **TEST GAP** — missing or misleading tests; CI gives false confidence.
- **DOC DRIFT** — documentation or plan claims falsified by code.
- **PROCESS / CLAIM** — `Done` or `PASS` unsupported by evidence.

---

## G1. Finding lifecycle status

| Status | Meaning |
|--------|---------|
| `PROPOSED` | Produced by audit; not yet operator-accepted. |
| `ACCEPTED` | Operator accepted; queued for remediation. |
| `IMPLEMENTING` | Remediation in progress. |
| `IMPLEMENTED` | Fix applied; awaiting independent verification. |
| `VERIFIED` | Independent verification passed. |
| `CLOSED` | Campaign remediation rollup complete for this finding. |
| `DISPUTED` | Operator disputes; finding and evidence preserved without acceptance. |
| `DEFERRED` | Postponed; requires rationale and revisit trigger. |
| `REJECTED` | Invalid or out of scope; requires rationale. |
| `WITHDRAWN` | Withdrawn; ID is not reused. |

Rules: audit produces `PROPOSED`; operator acceptance → `ACCEPTED`; remediation starts → `IMPLEMENTING`; implementer may reach `IMPLEMENTED`; implementer **must not** self-certify `VERIFIED` or `CLOSED`; independent verifier → `VERIFIED`; campaign remediation rollup → `CLOSED`.


## H. Finding IDs

Format (immutable once published):

```
AUDIT-YYYYMMDD-<LAYER_CODE>-NN
```

- `YYYYMMDD` — campaign date (folder date, no hyphens).
- `<LAYER_CODE>` — uppercase domain/layer identifier (e.g. `ORCHESTRATION`, `MEMORY`, `PLATFORM_FOUNDATION`).
- `NN` — two-digit sequence per layer per campaign (`01`, `02`, ...).

**Rules:**

- IDs are never reused or renumbered.
- If a finding is withdrawn, mark status `WITHDRAWN` with reason; do not delete.
- If duplicate root cause, cross-reference IDs; keep one primary.

---

## I. Required evidence (per finding)

Every finding MUST include:

1. **Claim falsified** — quote or paraphrase the architecture, plan, or implied invariant.
2. **Observation** — what the implementation actually does (behavior, not intent).
3. **Location** — file path(s) and line range(s) at audited SHA, or runtime trace/log snippet.
4. **Reproduction or inspection steps** — commands, call sequence, or static analysis path another auditor can repeat.
5. **Impact** — who/what is affected under section F stresses.
6. **Confidence** — `CONFIRMED` (reproduced or directly read) | `PROBABLE` (strong static evidence) | `HYPOTHESIS` (needs operator follow-up; not for CRITICAL without escalation).

Findings without item 3 and 4 are **insufficient** — downgrade to notes or block verdict (section J).

---

## J. Verdicts

Per-layer and campaign rollup:

| Verdict | Criteria |
|---------|----------|
| **PASS** | No CRITICAL/HIGH findings; MEDIUM findings acknowledged or accepted by operator; scope fully executed. |
| **PASS WITH GAPS** | No CRITICAL; HIGH items have documented mitigation or time-bound acceptance; known MEDIUM/LOW listed. |
| **FAIL** | One or more CRITICAL findings, or multiple unmitigated HIGH, or core scope claims falsified. |
| **BLOCKED / INSUFFICIENT EVIDENCE** | Cannot access code, SHA mismatch, scope ambiguity, or missing reproduction — audit cannot complete honestly. |

**Campaign verdict** is the worst layer verdict unless operator documents explicit acceptance of scoped exclusions.

---

## K. Prior-audit comparison (without anchoring)

When a prior campaign exists for the same layer:

1. List **resolved** findings (with fix SHA or evidence of removal).
2. List **regressions** (prior fixed, now present again).
3. List **still open** findings (by old ID if still valid, or new ID if changed manifestation).
4. List **new** findings.

**Anti-anchoring rules:**

- Do not assume prior PASS implies current PASS.
- Do not downgrade severity because a prior audit "already knew" — re-verify.
- Do not inherit prior verdict; compute fresh from current SHA.
- Prior audits are section B rank 5 only.

---

## L. Audit vs implementation

**During an audit:**

- **No runtime fixes** — do not patch production code, tests, or docs to "clear" findings mid-run.
- **No scope creep into implementation** — note defects; implementation is a separate change with separate review.

**Required sequence:**

1. **Inspect** — read code, config, infra as needed at pinned SHA.
2. **Falsify** — attack section E targets; run tests as evidence gathering, not as goal.
3. **Evidence** — attach section I proof to each finding.
4. **Classify** — section G severity and category.
5. **Present** — deliver findings to operator before treating them as accepted (section P).
6. **Accept** — operator confirms, disputes, or defers each finding.
7. **Persist** — write accepted artifacts to `docs/audit_results/YYYY-MM-DD/`.
8. **Sync arch/plan** — after acceptance, update docs per section M.

---

## M. Architecture / plan synchronization (after acceptance)

After operator accepts findings:

1. **Architecture** — update target docs if design must change to address ARCHITECTURAL RISK or accepted DEFECT remediation.
2. **Plan** — adjust task status, add remediation items, link finding IDs; never mark `Done` without behavior evidence.
3. **Cross-links** — layer audit file links to arch/plan sections; plan cites `AUDIT-...` IDs.
4. **No silent rewrite** — doc changes that reverse a finding require a new audit or explicit operator waiver recorded in the campaign README.

Synchronization is **post-acceptance** only — audits do not edit arch/plan to match broken code without recording the gap as a finding or accepted debt.

---

## N. Cross-layer review (rollup)

When multiple layers complete in one campaign, update the campaign `README.md` rollup section:

1. **Scope table** — layer, SHA, verdict, finding counts by severity.
2. **Cross-layer findings** — mismatches spanning tiers (e.g. orchestration assumes memory guarantee memory does not provide).
3. **Systemic themes** — repeated `dict[str, Any]`, fail-open patterns, test gaps.
4. **Campaign verdict** — per section J.
5. **Recommended remediation order** — operator-facing, not implementation.

Cross-layer issues get their own IDs using layer code `CROSS` (e.g. `AUDIT-20260818-CROSS-01`) in the campaign `README.md` rollup.

---

## O. Required per-layer result format

Each `docs/audit_results/YYYY-MM-DD/<LAYER_CODE>.md` MUST contain:

```markdown
# <LAYER_CODE> — Platform Audit

## Metadata
- Campaign date:
- Layer code:
- Tier(s):
- audited_sha:
- Status: IN_PROGRESS | COMPLETE | ABORTED
- Auditor:
- Architecture doc(s):
- Plan doc(s):
- Scope in:
- Scope out:
- Prior audit reference(s):

## Executive summary
(<=1 paragraph: verdict, headline risks)

## Verdict
(PASS | PASS WITH GAPS | FAIL | BLOCKED / INSUFFICIENT EVIDENCE)

## Findings
(enumerated AUDIT-... entries, each with section I evidence)

### AUDIT-YYYYMMDD-<LAYER_CODE>-01
- Severity:
- Category:
- Status: PROPOSED | ACCEPTED | IMPLEMENTING | IMPLEMENTED | VERIFIED | CLOSED | DISPUTED | DEFERRED | REJECTED | WITHDRAWN
- Claim falsified:
- Observation:
- Location:
- Reproduction:
- Impact:
- Confidence:

## Falsification log (negative results)
(targets examined, not falsified — brief)

## Prior-audit comparison
(section K, or N/A)

## Open questions / blocked items

## Operator acceptance
- Date:
- Accepted findings:
- Deferred findings:
- Disputed findings:
```

---

## P. Operator workflow

1. **Roadmap before layer** — operator defines campaign date, SHA (or branch to pin), layer order, and tier scope before the auditor starts.
2. **Scope confirmation** — auditor restates scope; operator confirms or corrects in writing (chat log or campaign README).
3. **Findings before persist** — present all findings and draft verdict; **no** writing to `docs/audit_results/` until operator reviews (except optional `IN_PROGRESS` stub without findings).
4. **Acceptance gates:**
   - Operator accepts, defers, or disputes each finding.
   - CRITICAL/HIGH deferrals require explicit rationale and target date in campaign README.
   - Only then persist final layer files and update register.
5. **Arch/plan sync** — separate step after acceptance (section M); may be same or follow-up change set.
6. **Campaign close** — update `docs/audit_results/README.md`, set status `COMPLETE` or `ABORTED`, publish `CAMPAIGN_SUMMARY.md` if multi-layer.

---

## Appendix: Layer codes (non-exhaustive)

Use domain names aligned with `docs/project/architecture/<DOMAIN>.md` pairs, e.g.:

`PLATFORM_FOUNDATION`, `ORCHESTRATION`, `NEXUS_EXECUTION_FLOW`, `UNIFIED_EXECUTION_RUNTIME`, `MEMORY`, `RAG`, `CONTEXT_ENGINEERING`, `INTEGRATIONS`, `MODALITY`, `RELIABILITY_FAILURE_AND_HITL`, `ADAPTIVE_HARNESS_INTELLIGENCE`, `PLATFORM_PLUGINS`

Capability-scoped audits may use capability hub names from `docs/project/capabilities/`.

---

*End of Intergrax Platform Audit Protocol v2*
