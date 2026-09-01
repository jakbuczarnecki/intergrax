# Local Knowledge Workspace (LKW) - harness hardening addendum

**Status:** Architecture addendum for LKW product proof  
**Parent architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Derived implementation plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
**Scope:** LKW-driven harness hardening only - not a separate platform refactor backlog

---

## 1. Purpose

LKW is the first product environment that validates Intergrax through a real local workload. This addendum records the platform-hardening work that is allowed because it directly supports LKW safety, diagnosability, evidence clarity, or adoption.

The goal is to avoid two failure modes:

1. building LKW on top of avoidable runtime safety gaps;
2. returning to broad abstract harness refactoring before a real product proof exists.

---

## 2. Governing decision

LKW is now the primary value-proof track. Harness work continues only when it is required by LKW or directly improves the reliability of the LKW proof path.

### Rule

A harness change is allowed in the LKW track only when it supports at least one of the following:

- LKW.1 acceptance criteria;
- LKW.2 graph pipeline acceptance criteria;
- local filesystem safety boundary;
- strict/product policy behavior;
- trace/evidence clarity for a real LKW run;
- first-run or adoption usability;
- a concrete implementation or testing bottleneck discovered while implementing LKW.

Broad platform refactors are deferred unless LKW exposes a concrete pain point.

---

## 3. Hardening register

| ID | Topic | Decision | Phase |
|----|-------|----------|-------|
| LKW-H0.1 | Strict/product policy must not silently default-allow | Required safety hardening | Before/with LKW.1 |
| LKW-H0.2 | `max_steps` boundary semantics | Add regression test before relying on step limits in product proof | Before/with LKW.1 |
| LKW-H0.3 | Post-finalization hook errors are currently not visible enough | Emit diagnostic/runtime event; do not silently swallow | Before/with LKW.1 |
| LKW-H1.1 | LKW live trace/evidence inspection | Show task, run, agent, step, tool, policy, RAG, and shadow artifact chain | LKW.1 |
| LKW-H1.2 | Observability and telemetry gaps | Implement only what is needed to inspect LKW runs | LKW.1-LKW.2 |
| LKW-H2.1 | Evidence and maturity wording may be misleading | Clarify architecture maturity vs live product proof vs production claim | After LKW.1 |
| LKW-H3.1 | Packaging and adoption are heavy | Simplify minimal install/run path after useful proof exists | After LKW.1/LKW.2 |
| LKW-W1 | `NexusLoop` constructor is wide | Watchlist only; no abstract refactor now | Deferred |
| LKW-W2 | `StepKernelContext` is wide | Watchlist only; no abstract refactor now | Deferred |

---

## 4. LKW-H0 - minimal runtime hardening for product proof

LKW-H0 is intentionally small. It must not become a general runtime rewrite.

### LKW-H0.1 - strict/product policy must fail closed

**Problem:** product or strict runtime posture cannot rely on implicit allow behavior when policy wiring is missing.

**Required behavior:**

- strict/product mode must not silently continue with `policy_engine=None`;
- missing policy in strict/product mode must produce either a configuration violation or an explicit fail-closed outcome;
- dev/test permissive behavior may remain available only when it is explicitly selected and visible;
- tests must cover both strict/product and permissive/dev behavior.

**Acceptance:** a strict/product LKW run cannot proceed through policy-sensitive execution without explicit policy wiring or explicit configuration acknowledgement.

### LKW-H0.2 - `max_steps` boundary test

**Problem:** step-limit semantics must be verified before LKW relies on bounded agent sessions.

**Required behavior:**

- test the exact boundary where `max_steps` is still allowed;
- test the first step beyond the limit;
- verify the terminal outcome and diagnostic reason;
- keep the test close to `HarnessKernel` or the ACP session loop, not inside LKW agent code.

**Acceptance:** regression test proves whether `max_steps=N` permits exactly N steps and rejects step N+1.

### LKW-H0.3 - post-finalization hook diagnostics

**Problem:** finalization or lifecycle hook errors must be inspectable. A product run may finish successfully, but invisible hook failures make the trace untrustworthy.

**Required behavior:**

- lifecycle/finalization hook errors emit a diagnostic/runtime event;
- event includes hook name, task/run identifier if available, and sanitized error type/message;
- the original terminal result may still be returned when the hook is non-critical;
- tests verify the error is visible in diagnostics or event output.

**Acceptance:** a failing finalization hook is visible in trace/diagnostics and is not silently swallowed.

---

## 5. LKW.1 - first product proof

LKW.1 remains the main active implementation wave. H0 exists only to make LKW.1 safe and inspectable.

### Required product path

```text
POST /v1/local_workspace/run
  -> capability=local.workspace.index with metadata.source_paths
  -> LocalIndexerAgent validates paths and invokes rag.ingest_document
  -> capability=local.workspace.search retrieves grounded evidence
  -> LocalSearchAgent returns answer with source references
  -> capability=local.workspace.synthesize writes draft artifact to shadow workspace
  -> trace/evidence view explains what happened
```

### LKW.1 acceptance

- `POST /v1/local_workspace/run` with `metadata.source_paths` and `capability=local.workspace.index` completes;
- follow-up search returns an answer referencing ingested content;
- synthesize with `shadow_workspace=true` writes only under the shadow root;
- Slack, tray, watcher, and OS service are not required;
- agent and host smoke tests are green;
- operator can inspect a readable run trace.

---

## 6. LKW-H1 - live trace/evidence inspection

The goal is not a hosted observability product. The goal is one clear, inspectable LKW run.

For every LKW.1 proof run, the operator should be able to inspect:

- submitted task and capability;
- task id and run id;
- selected agent;
- step sequence;
- invoked tools and outcomes;
- policy decisions;
- RAG ingest/retrieve evidence;
- shadow workspace artifact path;
- terminal outcome;
- diagnostics from non-fatal lifecycle/finalization failures.

**Acceptance:** a reviewer can understand what happened in an LKW run without reading internal runtime code.

---

## 7. LKW-H2 - evidence and maturity wording

After LKW.1 produces a real proof path, documentation must distinguish between:

| Claim type | Allowed wording |
|------------|-----------------|
| Architecture maturity | Strong architecture baseline / high architectural maturity |
| Harness baseline | Core harness proof path available |
| Live product proof | In progress through LKW |
| Production-proven claim | Not claimed until live product, provider, deployment, security, and adoption evidence exist |

**Rule:** do not market deterministic core evidence as full production certification.

---

## 8. LKW-H3 - packaging and adoption

Packaging work is deferred until LKW proves enough value to run repeatedly.

Potential follow-up packaging split:

```text
intergrax-core
intergrax-lab
intergrax-lkw
intergrax-rag
intergrax-all
```

Minimum adoption goal after LKW.1/LKW.2:

- a developer can start the LKW host locally;
- a developer can index a fixture or local folder;
- a developer can search and synthesize without reading the full platform documentation;
- dependency footprint and optional extras are documented clearly enough for a first run.

---

## 9. Deferred architecture watchlist

The following issues are real architectural pressure points, but they must not block LKW.1.

| Topic | Current decision | Trigger for action |
|-------|------------------|-------------------|
| `NexusLoop` constructor width | Accept as composition-root pressure | Refactor only if LKW requires repeated custom wiring, makes tests brittle, or forces duplicated bootstrap logic |
| `StepKernelContext` width | Accept as kernel execution-context pressure | Refactor only if unrelated concerns start changing together or test setup becomes excessive |

### Refactor rule

Do not refactor these components because they look wide. Refactor only when LKW exposes a measurable implementation, testing, or maintenance cost.

---

## 10. Execution order

```text
1. LKW-H0.1 strict/product policy fail-closed behavior
2. LKW-H0.2 max_steps boundary regression test
3. LKW-H0.3 finalization hook diagnostic event
4. LKW.1 index/search/synthesize stub product proof
5. LKW-H1.1 readable live LKW trace/evidence inspection
6. LKW.2 graph pipeline + local.workspace.* skills
7. LKW-H2.1 evidence/maturity wording cleanup
8. LKW-H3.1 packaging/adoption simplification
9. Deferred watchlist only if LKW exposes concrete pain
```

---

## 11. Stop conditions

Stop broad harness work when:

- the change cannot be tied to an LKW acceptance criterion;
- the change only improves conceptual elegance;
- the change starts a platform-wide refactor before LKW.1 is demonstrable;
- the change makes LKW harder to run locally;
- the change introduces new abstractions without a product proof requirement.

Continue only when the next change makes LKW safer, more observable, easier to run, or closer to a real user-facing proof.
