# Intergrax Audit Remediation Protocol

**Version:** 1.0
**Audience:** Human operators and model executors (Cursor agents, CI remediation jobs)
**Scope:** Post-audit remediation of findings recorded in Intergrax audit campaigns under `docs/audit_results/`.

This protocol is **canonical**. Executors MUST follow it in order. Deviations require explicit operator approval recorded in the campaign register.

---

## A. Select Audit

### A.1 Read the audit index

1. Read `docs/audit_results/README.md` before selecting any campaign.
2. Identify campaigns by **name**, **date**, and **status** (`IN_PROGRESS`, `COMPLETE`, `ABORTED`).
3. Note the campaign's **scope statement**, **severity model**, and **register location**.

### A.2 Default campaign selection

| Condition | Action |
|-----------|--------|
| Operator names a campaign | Use that campaign if it exists and is not `ABORTED` without operator override. |
| Operator does not name a campaign | Select the **latest `COMPLETE`** campaign whose scope matches the remediation request. |
| No `COMPLETE` campaign matches | **STOP.** Report gap; do not invent findings or remediate from partial notes. |

**Latest** means most recent `completed_at` (or equivalent) in the campaign metadata. If dates tie, prefer the campaign with higher declared coverage or the one referenced in the operator request.

### A.3 `IN_PROGRESS` campaigns — hard stop

- **Never** begin silent remediation against an `IN_PROGRESS` audit.
- An `IN_PROGRESS` campaign may be used only when the operator **explicitly** authorizes work against named, already-recorded findings in that campaign.
- If findings are still being added or severities are unsettled, **STOP** and request campaign completion or scoped authorization.

### A.4 Preconditions before queue build

Confirm:

- Campaign register file is readable and internally consistent (finding IDs unique).
- Accepted findings are distinguishable from draft or rejected entries.
- Architecture and plan references cited by findings exist or are flagged as missing (see Section C).

---

## B. Build Remediation Queue

### B.1 Inputs

Construct the queue from:

1. **Campaign register** — authoritative list of findings, severities, statuses, and cross-references.
2. **Accepted unresolved findings** — status `ACCEPTED` (and not `DEFERRED` / `REJECTED` / `CLOSED`). When resuming in-flight remediation, include `IMPLEMENTING` explicitly. Do **not** remediate `PROPOSED` findings before operator acceptance.
3. **Dependencies** — explicit `depends_on` fields, implied ordering (e.g., schema before consumer), and plan-block prerequisites.
4. **Plan blocks** — remediation work units from `docs/project/maintainers/plans/` or capability plan docs paired with architecture targets.

### B.2 Ordering rules

Sort candidate blocks by:

1. **Severity** (critical → high → medium → low) using the campaign's defined scale.
2. **Dependency** — unblockers before dependents; no implementation of blocked items.
3. **Leverage** — prefer changes that close multiple findings or reduce entire failure classes.

Tie-break: earlier finding ID, then alphabetically by plan block reference.

### B.3 Coherent blocks, not microtasks

- **Group** related findings into a single remediation block when they share architecture target, module boundary, and verification story.
- **Do not** create 1:1 finding-to-commit microtasks unless the finding is truly isolated.
- Each queue entry MUST name: `block_id`, `finding_ids[]`, `arch_ref`, `plan_ref`, `severity`, `dependencies[]`, `expected_proof`.

### B.4 Queue artifact

Maintain a working queue (markdown table, checklist, or campaign annex) updated as statuses change. The queue is the executor's roadmap for Sections D–L.

---

## C. Source Hierarchy

When sources disagree, resolve in this order:

| Priority | Source | Role |
|----------|--------|------|
| 1 | **Audit finding** | Describes the **problem** — observed gap, risk, or non-conformance. |
| 2 | **Architecture doc** (`docs/project/architecture/`, capability architecture) | Describes the **target** — intended design and invariants. |
| 3 | **Plan doc** (`docs/project/maintainers/plans/`, capability plan) | Describes the **work unit** — phased delivery and acceptance criteria. |
| 4 | **Codebase** | **Baseline** — what is implemented today; not a substitute for arch/plan when they define the target. |

### C.1 Conflict handling — STOP

**STOP** and escalate to the operator when:

- A finding's recommended fix contradicts the cited architecture invariant.
- Plan blocks referenced by the campaign are missing, superseded, or marked obsolete without a replacement pointer.
- Code "as-is" is treated as correct while audit and arch agree the behavior is wrong (code is baseline, not veto).
- Two architecture sources conflict with no documented precedence.

Record the conflict in the campaign register comment field; do not implement until hierarchy is reconciled.

---

## D0. Stale-finding revalidation (before every remediation block)

Before implementing **every** remediation block:

1. Resolve current `development` HEAD (`git rev-parse HEAD` on branch `development`).
2. Compare it with the `audited_sha` of the finding.
3. Revalidate that the finding still exists on current code.
4. Verify that target architecture/plan has not materially changed.

| Outcome | Action |
|---------|--------|
| Finding still exists | Implement normally. |
| Finding already fixed | **Do not** implement a duplicate fix. Identify existing commit/change, run independent verification, attach evidence, progress toward `VERIFIED`/`CLOSED` when justified. |
| Finding changed manifestation | Update trace/reconciliation; do not silently implement the stale prescription. |
| Current architecture contradicts old remediation target | **STOP** and report architecture conflict. |

An audit finding defines the **historical observed problem**. It is **not** permission to blindly overwrite current source.


## D. Task Selection

1. Select the **highest-priority unblocked** item from the remediation queue (Section B).
2. Restate the block's scope in one paragraph: findings addressed, files likely touched, out-of-scope boundaries.
3. **Scoped reads only** — open files required for this block (architecture section, plan section, implicated modules, existing tests). Do not load unrelated hubs, full domain guides, or repo-wide search unless the block requires it.
4. If the block expands beyond the stated scope during investigation, **STOP** (see Section G).

---

## E. Operator Workflow

### E.1 Interactive sessions (default)

Before implementation:

1. Present a **roadmap** — ordered blocks with severity and dependencies.
2. Explain in **plain language** what will change and why (problem → target → approach).
3. State **expected proof** — tests, CI jobs, manual checks, or artifacts that will demonstrate resolution.
4. Obtain **confirmation** before editing code or docs, unless the operator has pre-authorized autonomous execution for the session.

### E.2 Non-interactive / batch execution

When the operator pre-authorizes a batch:

- Still document the roadmap and expected proof in the campaign trace (Section K).
- Halt at the same **STOP** gates (conflicts, scope expansion, missing arch/plan).

### E.3 Status transitions at start

When work begins on a block, set implicated findings to `IMPLEMENTING` in the campaign register (Section I).

---

## F. Production Quality Requirements

All remediation MUST meet production quality bar for Intergrax:

- **Correctness** — behavior matches architecture target; edge cases from the finding are covered.
- **Clarity** — code and docs readable; public surfaces documented at the same level as surrounding code.
- **Safety** — no broadening of trust boundaries; failures fail closed where arch requires it.
- **Observability** — meaningful errors/logs where the change introduces new failure modes.
- **Performance** — no gratuitous regression; intentional trade-offs documented in plan or commit message.
- **Consistency** — naming, patterns, and tier boundaries match existing conventions (`AGENTS.md` import rules).
- **Minimal diff** — smallest change that fully resolves the block; no drive-by refactors.

---

## G. Intergrax Engineering Restrictions

Executors MUST obey these constraints during remediation:

| Rule | Requirement |
|------|-------------|
| **Branch** | Work only on the existing shared branch named exactly `development`. Do **not** create branches or worktrees unless explicitly authorized by the operator. |
| **History** | **No** `reset`, `rebase`, `stash`, `clean`, `amend`, or **force-push** unless operator explicitly requests a specific command. |
| **Concurrent work** | **Preserve** other in-progress work on the branch; do not discard unrelated changes. |
| **Staging** | Stage **task-owned files only**; never `git add -A` or blanket staging. |
| **Reflection** | **No loose reflection** — do not introduce ad-hoc introspection, dynamic attribute hacks, or stringly-typed dispatch to "make tests pass." |
| **Typing** | **Avoid** `dict[str, Any]` (and equivalent untyped bags) on **critical surfaces** — public APIs, security boundaries, orchestration contracts, persistence schemas. Use typed models or narrow mappings. |
| **Legacy** | **Remove dead legacy** touched by the change; do not leave orphaned code paths that contradict the new behavior. |
| **Scope** | On **scope expansion** (new findings, new modules, arch/plan drift), **STOP** and re-queue with operator approval. |

---

## H. Verification

Remediation is not complete at implementation time. Each block requires verification evidence.

### H.1 Required verification types

1. **Automated tests** — unit/integration tests exercising the fixed behavior.
2. **Negative / bypass tests** — cases proving the original failure mode no longer succeeds (regression guards).
3. **CI gates** — relevant pipeline jobs green (or explicitly waived by operator with reason recorded).
4. **Diff inspection** — confirm only intended files changed and no unrelated damage.

### H.2 Insufficient verification

**Old tests passing alone is insufficient** if they did not cover the finding. Add or extend tests that would have failed before the fix.

### H.3 Verification artifact

Record commands run, job URLs or IDs, and test names in the campaign trace (Section K). Set finding status to `IMPLEMENTED` when code/docs merge locally; do not set `VERIFIED` without Section J.

---

## I. Status Model

### I.1 Finding statuses

| Status | Meaning |
|--------|---------|
| `PROPOSED` | Produced by audit; not yet operator-accepted. |
| `ACCEPTED` | Agreed valid; queued for remediation. |
| `IMPLEMENTING` | Active work in progress on this finding. |
| `IMPLEMENTED` | Fix applied; awaiting independent verification. |
| `VERIFIED` | Independent verification passed (Section J). |
| `CLOSED` | Remediation complete and accepted in campaign rollup. |
| `DISPUTED` | Operator disputes; finding and evidence preserved without acceptance. |
| `DEFERRED` | Explicitly postponed with reason and revisit trigger. |
| `REJECTED` | Invalid or out of scope; will not fix; requires rationale. |
| `WITHDRAWN` | Withdrawn; ID is not reused. |

### I.2 Campaign statuses

| Status | Meaning |
|--------|---------|
| `IN_PROGRESS` | Campaign active; layers may be incomplete. |
| `COMPLETE` | Scoped layers finished; rollup published in campaign `README.md`. |
| `ABORTED` | Campaign halted with documented reason. |

Legacy material is identified by location under `legacy/`, not by a competing active campaign status.

### I.3 Transition rules

- Audit produces `PROPOSED` findings; operator acceptance → `ACCEPTED`.
- Implementer may set: `ACCEPTED`→`IMPLEMENTING`, `IMPLEMENTING`→`IMPLEMENTED`.
- Implementer **cannot** self-certify `VERIFIED` or `CLOSED`.
- Independent verifier sets `VERIFIED` after Section J.
- `CLOSED` follows `VERIFIED` and completion rollup (Section L).
- `DEFERRED`, `REJECTED`, and `DISPUTED` require operator acknowledgment and rationale in the register.
- `WITHDRAWN` does not delete or reuse the finding ID.

## J. Independent Verification Pass

After `IMPLEMENTED`:

1. A **different executor** (another agent session, reviewer, or CI bot step) performs verification per Section H without authoring the original fix.
2. The verifier runs the stated proof, inspects the diff, and confirms alignment with architecture target.
3. On success: set `VERIFIED` and link evidence in the trace chain.
4. On failure: revert status to `ACCEPTED` or `IMPLEMENTING` with a defect note; do not mark `CLOSED`.

If independent verification is impossible in the environment, **STOP** and request operator-orchestrated review.

---

## K. Traceability Chain

Every remediated finding MUST maintain an auditable chain:

```text
finding_id → arch_ref → plan_block → commit_hash → verification_evidence → finding_status
```

### K.1 Required fields (per finding or block)

| Field | Description |
|-------|-------------|
| `finding_id` | Campaign register identifier |
| `arch_ref` | Doc path + section anchor |
| `plan_block` | Plan doc path + block/phase id |
| `commit_hash` | One or more commits (full SHA) |
| `verification_evidence` | Test names, CI run, reviewer id, date |
| `finding_status` | Per-finding lifecycle status from Section I.1 (`PROPOSED` / `ACCEPTED` / `IMPLEMENTING` / `IMPLEMENTED` / `VERIFIED` / `CLOSED` / `DISPUTED` / `DEFERRED` / `REJECTED` / `WITHDRAWN`) |

Broken links in the chain block `CLOSED` for that finding.

---

## L. Completion Rollup

### L.1 Campaign closure criteria

Before recommending audit closure:

- All `ACCEPTED` findings are `CLOSED`, or explicitly `DEFERRED` / `REJECTED` with rationale.
- No finding remains in `IMPLEMENTING` or `IMPLEMENTED` without an active verifier.
- Traceability chain (Section K) is complete for every `CLOSED` finding.
- Queue is empty or only contains deferred items with revisit triggers.

### L.2 Rollup report

Produce a short rollup:

- Counts by final status and severity.
- List of `DEFERRED` and `REJECTED` with reasons.
- Commits and verification summary.
- Residual risks explicitly acknowledged.

### L.3 Follow-on audit

Recommend a **later independent audit** (new campaign) when:

- Scope was large or cross-cutting.
- Multiple items were deferred.
- Architecture changed materially during remediation.

Do not claim "fully audited" solely because remediation closed — only that **this campaign's accepted findings were addressed per this protocol**.

---

## Executor Checklist (Quick Reference)

1. [ ] Read `docs/audit_results/README.md`
2. [ ] Select latest `COMPLETE` campaign (or operator-named); never silent `IN_PROGRESS` remediation
3. [ ] Build ordered queue from register + arch + plan; group coherent blocks
4. [ ] Resolve sources per hierarchy; STOP on conflict
5. [ ] Pick top unblocked block; scoped reads only
6. [ ] Roadmap + plain language + expected proof; confirm before implement (if interactive)
7. [ ] Implement with production quality and Section G restrictions
8. [ ] Verify with new/negative tests + CI + diff review
9. [ ] Set `IMPLEMENTED`; independent pass → `VERIFIED` → `CLOSED`
10. [ ] Maintain traceability chain; rollup; recommend follow-on audit if warranted

---

*This protocol does not depend on deleted orchestrator or `progress.json` machinery. Campaign registers and this document are the sole coordination sources.*
