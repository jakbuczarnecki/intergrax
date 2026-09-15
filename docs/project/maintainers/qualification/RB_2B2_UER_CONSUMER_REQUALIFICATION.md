# RB-2B2 — UER Consumer Re-qualification

**Status:** COMPLETE (qualification-only — no frozen Execution Engine mutation)

**Branch:** `development`

| Gate | Value |
|------|-------|
| **RB2B2_BASELINE_HEAD** | `6ec2345cbf4db6091160965aada3965bde454dbd` |
| **RB2B2_EVIDENCE_HEAD** | `92c6fdef4a1b6810192a05c5d27f59b96177b4b9` (origin/development at qualification commit) |
| **origin/development @ qualification** | `92c6fdef4a1b6810192a05c5d27f59b96177b4b9` |
| **HEAD == origin/development** | **YES** |
| **Delta since baseline pin** | `ca5c26be8` (OBS), `0e2050236` (memory), `92c6fdef4` (delegated provider reattachment) — UER-FIX conclusions unchanged |
| **Frozen Execution Engine** | **NOT modified** |

**Parent:** [`EXECUTION_ADOPTION_RESIDUAL_AUDIT_RB2A.md`](../audits/EXECUTION_ADOPTION_RESIDUAL_AUDIT_RB2A.md) · [`CROSS_LAYER_ARCHITECTURE_REBASE_RB0_LEDGER.md`](../audits/CROSS_LAYER_ARCHITECTURE_REBASE_RB0_LEDGER.md)

**Historical source:** [`docs/audit_results/2026-08-18/EXECUTION_RUNTIME.md`](../../../audit_results/2026-08-18/EXECUTION_RUNTIME.md)

---

## Executive answer

On `development` @ `92c6fdef4`, **production execution ingress still shows zero canonical bypass** (U5 P0 inventory + RB-2A re-confirmed). **All six historical UER-FIX defects remain observable** on the direct ACP / HarnessKernel / cancellation surfaces audited in 2026-08-18; none are closed with class **A**.

RB-2B2 separates:

1. **Execution authority convergence** (identity / lifecycle / strategy / terminal via `HostTaskExecution` → `ExecutionRuntime`) — **PASS**.
2. **Historical UER-FIX remediation** (ACP session reliability, kernel step atomicity, cancel/checkpoint) — **OPEN** (classes **D** or **F** below).

---

## Phase 1 — Historical defect map

| Finding | Historical defect | Historical expected fix |
|---------|-------------------|------------------------|
| **01** | Direct ACP builds fresh `PolicyEngine()`; host carries no canonical policy engine | **UER-FIX-A** — propagate host/Nexus policy engine into `StepKernelContext` |
| **02** | `HarnessKernel` merges state before actions; failed step can leave merged state with `outcome_applied=False` | **UER-FIX-B** — atomic step commit / rollback |
| **03** | ACP resume mints new `AttemptId`; checkpoint lacks `AttemptId` | **UER-FIX-C** — resume-without-retry preserves attempt; retry mints new |
| **04** | Unexpected agent exceptions escape without typed `AgentRunResult` | **UER-FIX-D** — terminal FAILED boundary around session loop |
| **05** | Task cancellation does not reach active ACP iteration loop | **UER-FIX-E** — cooperative cancel at step/LLM/tool boundaries |
| **06** | Cancel clears task pointers only; checkpoint store has no invalidate/tombstone | **UER-FIX-E** — invalidate resumable checkpoint authority on cancel |

---

## Phase 2 — Current production mapping (representative)

| Finding | Current consumer / surface | Canonical contract | Implementation | Execution entry | Identity owner | Lifecycle owner | Retry / resume owner | Terminal owner |
|---------|---------------------------|-------------------|----------------|-----------------|----------------|-----------------|----------------------|----------------|
| 01–06 (shared ingress) | Tier-3 / harness → `IntergraxAgent.run` → `run_acp_session` | Agent run + host metadata (`ACPSessionHostContext`) | `acp_run.py` inside AGENTIC strategy work | `HostTaskExecution` → `ExecutionRuntime` → agent engine | **EE** (`identity_authority` at host); ACP re-mints at session ingress | **EE** root lifecycle | **EE** recovery plane for host resume; ACP checkpoint resume local | **EE** host terminal; ACP maps step records |
| 02 | Same agentic path → `AgentRuntime.advance_step` → `HarnessKernel` | Step execution record contract | `step_kernel.py` | Inside active execution | EE | EE | EE | Kernel + ACP result mapping |
| 05–06 | Task control + scheduler + `CancellationCoordinator` | Task metadata + checkpoint port | `cancellation/coordinator.py`, `AgentCheckpointStore` | Host resume / task control | EE | EE | Scheduler timing only; cancel flag on Task metadata | EE |

**Production bypass:** **0** — unchanged from RB-2A / U5.

---

## Phases 3–14 — Architecture qualification summary

| Invariant | Verdict | Evidence |
|-----------|---------|----------|
| Root identity (P1) | **CANONICAL** | Host paths use `ExecutionRuntime` / `identity_authority`; ACP ingress uses `default_execution_identity_authority` (not ad-hoc `mint_*` in agents) |
| Attempt / retry (P2) | **GAP (UER-03)** | ACP session ingress always mints before resume resolution |
| Child identity (P3) | **CANONICAL** | `ChildExecutionRunner` / delegated ports @ RB-2A inventory |
| Lifecycle ownership (P4) | **CANONICAL** | No production consumer owns canonical START/RUN/RETRY/terminal truth outside EE |
| Strategy routing (P5) | **CANONICAL** | Consumers express task/intent; `StrategyExecutionRouter` selects strategy |
| Resume (P6) | **PARTIAL** | Host resume via `HostTaskExecutionPort` canonical; **ACP direct checkpoint resume** still consumer-local semantics |
| Terminal (P7) | **PARTIAL** | Host terminal via EE; **UER-04** raw exceptions on direct ACP |
| Background worker (P11) | **CANONICAL** | Delivery/lease ≠ execution identity (RB-2A EP-07) |
| Agents / kernel (P12) | **ADAPTER + DEFECT** | Glue is thin; historical UER-FIX gaps remain on ACP/kernel/cancel |
| Delegated execution (P10) | **CANONICAL** | No second root owner @ HEAD |
| Pluginability (P8–P9) | **CANONICAL** | `HostTaskExecutionPort`, child ports, launch ports — fakes in unit tests |
| Zero bypass (P10) | **PASS** | `test_ee_final_arch_zero_execution_bypass.py` |

---

## Phase 15 — Finding verdicts

| Finding | Current evidence @ `6ec2345c` | Verdict | Final class |
|---------|------------------------------|---------|-------------|
| **01** | `acp_run.py` L329 `policy_engine=PolicyEngine()`; `ACPSessionHostContext` has no policy carrier | **DEFECT — CONSUMER FIX** | **D** |
| **02** | `step_kernel.py` assigns `state_root` before tool failure; behavioral test `test_rb2b2_uer02_*` | **DEFECT — FROZEN KERNEL (ADR)** | **F** |
| **03** | `_resolve_acp_session_identity` + `mint_execution_identity`; checkpoint schema without `attempt_id` | **DEFECT — CONSUMER + CONTRACT** | **D** |
| **04** | `step_loop.py` catches budget only; `test_rb2b2_uer04_*` | **DEFECT — CONSUMER FIX** | **D** |
| **05** | No `CancellationCoordinator` usage under `intergrax/agents/authoring/` | **DEFECT — CONSUMER/WIRING FIX** | **D** |
| **06** | `clear_checkpoint_state` pointer-only; checkpoint port save/get only | **DEFECT — PORT + CONSUMER** | **D** |

**E → A closures:** **0**

---

## Phase 16 — Implementation policy

No production code changes in RB-2B2 (qualification-only). Remediation tracks:

- **01, 03, 04, 05, 06:** consumer-side / port extensions without reopening `ExecutionRuntime` semantics.
- **02:** **STOP / ADR** — `HarnessKernel` is frozen EE surface (`NPSC-3C`).

---

## Phase 17 — Behavioral proof commands

```text
uv run pytest tests/unit/agents/authoring/test_rb2b2_uer_consumer_requalification.py -q
uv run pytest tests/unit/runtime/architecture/test_ee_final_arch_zero_execution_bypass.py -q
uv run pytest tests/unit/agents/authoring/test_acp_session_identity.py -q
```

Proof matrix: P1/P3/P5/P8/P9/P10 via existing architecture gates + RB-2A; P2/P6/P7 partial gaps recorded above; UER-01/02/03/04 behavioral tests in `test_rb2b2_uer_consumer_requalification.py`.

---

## STOP / ADR (UER-FIX-B)

| Field | Value |
|-------|-------|
| Problem | Non-atomic step state commit in `HarnessKernel.execute_step` |
| Canonical contract | Step execution record + state merge semantics (UER) |
| Missing capability | Rollback or commit-after-success without changing EE ownership |
| Affected frozen owner | `intergrax/runtime/kernel/step_kernel.py` |
| Options | ADR for post-freeze kernel amendment vs. compensating transaction pattern |
| Blast radius | All agentic step execution, resume, audit |

---

## Remaining UER consumer risk

Direct ACP (`run_acp_session`) remains a **production agentic consumer** with **reliability and policy propagation debt**. It does **not** constitute a second Execution Runtime, but it **can** diverge from host policy, attempt continuity, exception terminals, and cooperative cancellation until UER-FIX blocks land.
