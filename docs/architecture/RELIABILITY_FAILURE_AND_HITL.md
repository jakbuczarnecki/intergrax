# Reliability, Failure Model, and Human-in-the-Loop

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 22  
**Audit instruction:** [`audit/RELIABILITY_FAILURE_AND_HITL.md`](../audit/RELIABILITY_FAILURE_AND_HITL.md)  
**Last updated:** 2026-06-20 — **P2-ARCH-09** Attempt Ledger + retry ownership; REL + HITL **Done**

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (RELIABILITY_FAILURE_AND_HITL canon).

- **Implement / audit default:** §30–§32 failure + retry + HITL core. Extended §33+: [`satellites/RELIABILITY_FAILURE_AND_HITL_extended_depth.md`](satellites/RELIABILITY_FAILURE_AND_HITL_extended_depth.md). §35+: [`satellites/RELIABILITY_FAILURE_AND_HITL_production_gates.md`](satellites/RELIABILITY_FAILURE_AND_HITL_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md`](../guides/audit_slices/RELIABILITY_FAILURE_AND_HITL.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/RELIABILITY_FAILURE_AND_HITL_extended_depth.md`](satellites/RELIABILITY_FAILURE_AND_HITL_extended_depth.md) | extended depth |
| [`satellites/RELIABILITY_FAILURE_AND_HITL_production_gates.md`](satellites/RELIABILITY_FAILURE_AND_HITL_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

# 30. Failure Model

Failures are expected.

The system must treat failure as normal.

Failure types:

- agent failure
- tool failure
- adapter failure
- timeout
- invalid output
- missing data
- low confidence
- unsafe action
- guardrail denial (`LlmGuardrailMiddleware` BLOCK at input/output/tool hooks — composes with HITL when policy escalates; see [`INTEGRATIONS.md`](INTEGRATIONS.md) §47)
- human rejection
- incomplete result

Failure handling options:

- retry same step
- retry with different agent
- ask human
- degrade gracefully
- return partial result
- stop execution
- mark as failed

---


---

# 31. Retry Policy

Retries must be controlled.

Every retry should have:

- reason
- retry count
- changed strategy if possible
- stop condition

Do not retry endlessly.

Retries should be visible in traces.

### 31.1 Two retry layers (do not double-retry)

Intergrax has **two independent retry layers**. Configure each explicitly; avoid aggressive values on both for the same step without trace events.

| Layer | Location | Scope | Policy |
|-------|----------|-------|--------|
| **Graph / validation** | `RetryEngine` (`runtime/nexus/retry/retry_engine.py`) | Nexus execution graph after agent step validation fails | `RetryPolicy` (`max_retries`, `retry_alternate_agent`); may switch agent via `AgentRegistry`; `RetryRecord` on task result; hooks `BEFORE_RETRY` / `AFTER_RETRY` when middleware wired |
| **Run-level** | `AgentEngine` / `HarnessKernel` | Transient LLM or tool failures inside one agent run (`RuntimeErrorCode.LLM_ERROR`, `TOOL_ERROR`, …) | `RuntimeConfig.max_run_retries`, `retry_run_on`; re-runs `on_next_step` iteration; does not change Nexus agent selection |

A future `RetryCoordinator` may delegate to both with explicit `RETRY_SCHEDULED` / `RETRY_STARTED` events (§42.34). Until then, agents emit **intent** (`AgentDecision.RETRY`); runtime executes policy — no agent-internal `for attempt in range(n)` against adapters.

**Full retry-layer taxonomy (R0–R4) and attempt reconstruction:** [Attempt Ledger](#attempt-ledger) below. **As-built mapping:** graph/validation ≈ **R3**; run-level ≈ **R2**; whole-run graph retry ≈ **R3** (coordinator scope).

---

## Attempt Ledger

**Attempt Ledger** is the logical runtime record of execution attempts, retries, failures, escalations, degradations, HITL pauses and terminal stop reasons.

It does not have to be a single physical class in the current implementation.
It is an **architectural invariant**: every meaningful retry/failure decision must be reconstructable from runtime events and retry metadata.

Sources include (non-exhaustive): `RetryRecord`, `PauseRecord`, `RuntimeCheckpoint`, `RETRY_*` / `TOOL_*` / HITL `RuntimeEvent`s, `ToolCallTrace`, validation and critic verdict payloads, and correlation fields on the observability spine ([`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine)).

**Cross-refs:** [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) §8 · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §14 · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.34 · [`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine) · [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md#verification-safety-boundaries) · [`TOOLS.md`](TOOLS.md) · [`INTEGRATIONS.md`](INTEGRATIONS.md) · [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) · [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary) · [`CODE_CRAFT.md`](CODE_CRAFT.md#codecraft-safety-boundary)

---

## Attempt Ledger responsibilities

Attempt Ledger **SHOULD** allow an operator to reconstruct:

- `task_id` / `run_id` / `node_id` / `agent_id` / `step_id` involved,
- attempt number,
- original trigger,
- failure type,
- failure source,
- retry policy used,
- retry layer,
- whether retry was allowed or denied,
- backoff / delay decision if applicable,
- idempotency key if applicable,
- side effects already performed,
- validation result,
- critic / verification result if applicable,
- HITL request / approval / rejection if applicable,
- degradation decision,
- final stop reason,
- terminal outcome.

---

## Retry ownership rules

- Agents **MAY** request recovery intent, but **MUST NOT** own global retry policy.
- Agents **MUST NOT** implement unbounded retry loops.
- Tools **MAY** expose retryable failure metadata, but **MUST NOT** silently retry high-risk side effects beyond backend/protocol-safe retry rules.
- Integrations **MAY** perform protocol-level retries only when safe and compatible with runtime retry policy.
- Nexus / runtime owns orchestration-level retry, escalation, HITL and terminal stop decisions.
- Validation / critic failures must be recorded as retry inputs, not hidden inside final narrative.
- Retry decisions must be traceable through `RuntimeEvent` / observability spine.
- High-risk side effects require idempotency or explicit policy exception before retry.
- A retry layer **MUST** be identifiable for every retry attempt.

---

## Retry layers

Normative retry layers — every retry attempt **MUST** map to exactly one layer. Layers compose; they do not duplicate uncontrolled retries at the same semantic step.

### R0 — Backend/protocol retry

**Owner:** Integration or low-level client.

**Use:** Transient transport failure, rate limit retry-after, safe idempotent backend call.

**Limits:** Must not hide semantic failure. Must not retry unsafe side effects unless idempotency and policy allow it.

### R1 — ToolRuntime retry

**Owner:** ToolRuntime / policy.

**Use:** Agent-requested tool side effect failed in a retryable way.

**Limits:** Must preserve `tool_call_id` / idempotency / attempt metadata. Must not bypass policy or observability.

### R2 — Agent step retry

**Owner:** AgentEngine / runtime policy.

**Use:** Agent step failed validation, malformed output, recoverable local step issue.

**Limits:** Agent may produce a new decision, but runtime owns retry count and stop conditions. Maps to §31.1 **run-level** retry and NEXUS §14.1 **Layer B**.

### R3 — Graph / Nexus retry

**Owner:** Nexus / graph runtime.

**Use:** Node failure, alternate agent, graph-level degradation, partial result, replan.

**Limits:** Must not duplicate side effects without idempotency / policy clearance. Maps to §31.1 **graph/validation** retry, NEXUS §14.1 **Layer A**, and whole-run **Layer C** when enabled.

### R4 — HITL / human-mediated retry

**Owner:** Nexus / HITL runtime.

**Use:** Human corrects input, approves continuation, rejects output, requests re-run.

**Limits:** Human decision must be traceable.

---

## Stop reasons

Terminal outcomes **SHOULD** include a clear **stop reason** (architectural vocabulary — not a code enum). Use the most specific reason that explains why execution ended.

| Stop reason | Meaning |
|-------------|---------|
| `completed` | Task/run reached successful terminal state |
| `validation_failed` | Deterministic validation rejected output; retries exhausted or denied |
| `policy_denied` | PolicyEngine or guardrail blocked continuation |
| `budget_exceeded` | Cost, token, tool-call or time budget exhausted |
| `timeout` | Step, tool, or run timeout reached |
| `max_attempts_exceeded` | Retry budget for the applicable layer exhausted |
| `human_rejected` | Operator rejected output or action via HITL |
| `human_timeout` | HITL queue item expired without decision |
| `unsafe_side_effect_risk` | Retry or continuation blocked due to irreversible side-effect risk |
| `missing_required_context` | Required context or dependency unavailable |
| `tool_unavailable` | Tool registry miss, deny, or hard tool failure |
| `integration_unavailable` | Backend/provider unavailable after safe retries |
| `partial_result_returned` | Run stopped with explicit partial completion policy |
| `degraded_result_returned` | Run returned degraded output per resilience policy |
| `cancelled` | Operator or system cancellation |

**Related taxonomy:** failure classes §33.4 · abandonment triggers [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §14.2 · terminal `RuntimeEvent` / `ops:completion` filters [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.1.

---

## Governed Continuation (composition — GEC-4)

**Platform reference:** [`governed_external_execution.md`](../platform/governed_external_execution.md).

**Capability:** reusable pause-for-governance → decision → continuation evidence → resume.

**Not** a new runtime. Composes existing Nexus `ExecutionInterrupt` / `ExecutionInterruptHandler`, HITL `HumanDecisionRecord`, and ACP/UAEP resume. Contract helpers: `intergrax.contracts.governed_continuation` ([ADR-GOVERNED-CONTINUATION-001](../adr/entries/2026-07-20/ADR-GOVERNED-CONTINUATION-001.md)).

```text
execution → surface GovernedContinuationRequest → ExecutionInterrupt
         → governance decision (policy / HITL) → continuation evidence → Nexus resume
```

| Concept | Owner |
|---------|--------|
| `ContinuationReason` (generic: quote, security, legal, …) | Platform contracts |
| Interrupt / pause / resume | Nexus + HITL |
| Approval / deny | Policy + human decision |
| Surface blocker + forward evidence | Tier-2 adapter (mapping only) |
| First consumer | External Work (`ContinuationReason.QUOTE` + `QuoteAcceptanceEvidence`) |

**Reuse audit (GEC-4):** platform already supported continuation via interrupt + HITL + resume; only a generic reason discriminator and composition helpers were added. Forbidden: `ContinuationRuntime`, `QuoteLifecycleEngine`, quote-specific interrupt types.

**Deferred:** quote UX, receipt persistence, provider transport.

---

## Meaningful side-effect policy (composition — GEC-5)

**Capability:** authorize proposed external actions that may create commitments, mutations, disclosures, or irreversible consequences **before** provider-bound execution.

**Not** a quote-approval engine, payment policy, or second policy runtime. Reuses `PolicyDecision` / `PolicyAction` and `PolicyEngine` / `RuntimePolicyEngine.evaluate_meaningful_side_effect`. Request contract: `intergrax.contracts.meaningful_side_effect` ([ADR-POLICY-SIDE-EFFECT-001](../adr/entries/2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md)).

```text
proposed external action → MeaningfulSideEffectRequest → policy evaluate
  → ALLOW → execute  |  DENY → stop  |  REQUIRE_HUMAN → GovernedContinuationRequest
```

| Rule | Detail |
|------|--------|
| Fail closed | Missing evaluator, principal, run identity, or indeterminate → DENY (no silent allow) |
| Quote receipt | Observational — not a side-effect gate |
| Quote acceptance | Meaningful — policy before `submit_quote_acceptance` |
| Evidence ≠ allow | Continuation evidence still requires policy ALLOW unless architecture defines a trusted final authorization artifact (not assumed here) |
| First consumer | External Work (`CREATE_EXTERNAL_WORK` / `ACCEPT_QUOTE` / `CANCEL_EXTERNAL_WORK`) |

**Composition with GEC-4:** REQUIRE_HUMAN maps to existing Governed Continuation / Nexus interrupt — policy does not resume Nexus.

**Composition with GEC-6:** after ALLOW + successful side effect, consumers **must** compose a descriptive `GovernedProofProfile` that **references** this policy outcome (does not recompute it). Platform consolidation: [`governed_external_execution.md`](../platform/governed_external_execution.md).

**Deferred:** product policy packs, spend thresholds, payment/wallet, ProofReceipt persistence.

---

## Governed proof profile (composition — GEC-6)

**Capability:** describe the minimum facts required to prove that a governed external side effect occurred under proper platform governance.

> A proof profile is a description of governed execution, not a receipt, not an audit log, and not an authorization mechanism.

**Not** persistence, signatures, cryptography, receipts, audit storage, or a verification engine. Contract: `intergrax.contracts.governed_proof` ([ADR-GOVERNED-PROOF-001](../adr/entries/2026-07-20/ADR-GOVERNED-PROOF-001.md)).

```text
ALLOW + side effect succeeds → compose GovernedProofProfile
  (principal, task/run, action, resource, provider, PolicyAction refs,
   governance evidence refs, correlation/idempotency, optional ContinuationReason)
```

| Rule | Detail |
|------|--------|
| Descriptive only | Never authorizes, resumes, evaluates policy, signs, stores, or publishes |
| Policy | Records `PolicyAction` / rule refs — does not recompute |
| Evidence | References artifacts (e.g. `QuoteAcceptanceEvidence` by id) — does not embed payloads |
| Identity | Preserves existing `task_id` / `run_id` / `correlation_id` / `idempotency_key` |
| Provider neutrality | No SDK objects, HTTP/REST/JSON-RPC payloads, or transport headers |
| First consumer | External Work (Tier-2 composes; does not own receipt product) |

**Deferred:** ProofReceipt persistence, signing, audit databases, verification/replay engines.

---

## Cursor review checklist

Before adding or modifying retry/failure/HITL behavior, Cursor must verify:

- [ ] Which retry layer is this?
- [ ] Is the attempt recorded or reconstructable?
- [ ] Is there a max attempt / stop condition?
- [ ] Are side effects idempotent or protected by policy?
- [ ] Is the failure type explicit?
- [ ] Is the failure source explicit?
- [ ] Are validation and critic failures visible to the runtime?
- [ ] Is HITL handled by Nexus / HITL runtime, not an agent-local flow?
- [ ] Are `RuntimeEvent` / observability identifiers preserved?
- [ ] Is the terminal stop reason explicit?
- [ ] Does this create duplicate retries across layers?

---

# 32. Human In The Loop

Human approval may be required for:

- sending external messages
- modifying external systems
- deleting data
- financial actions
- legal conclusions
- risky automation
- uncertain results

Nexus manages human approval.

Agents may request approval via `AgentDecision.REQUEST_HUMAN`, but Nexus controls the approval flow (§42.10).

Agents MUST NOT implement ad-hoc human gates or send approval messages directly.

---

---
