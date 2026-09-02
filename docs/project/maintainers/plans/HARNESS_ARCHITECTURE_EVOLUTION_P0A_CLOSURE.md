# Harness Architecture Evolution — P0A Closure

## Verdict

**P0A STATUS: CLOSED**  
**Feature/code implementation during P0A: NONE**  
**Documentation sync: COMPLETE** (canonical architecture CURRENT sections synchronized; transitional reconciliation document removed)

P0A is closed against the verified repository state established by:

- [`HARNESS_ARCHITECTURE_EVOLUTION_P0A_AS_BUILT_AUDIT.md`](HARNESS_ARCHITECTURE_EVOLUTION_P0A_AS_BUILT_AUDIT.md)

The master program remains [`HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md`](../../overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md).

---

# 1. What P0A established

P0A separated four categories that must not be confused in later implementation:

- `DONE` — real shipped foundation; reuse it,
- `PARTIAL` — extend/converge it,
- `GAP` — real implementation work remains,
- `DOC-DRIFT` — older descriptive CURRENT prose is behind code and must not generate feature work.

The canonical implementation rule is:

> **ALREADY EXISTS => DO NOT REBUILD.**

Descriptive CURRENT state now lives in canonical architecture documents (`UNIFIED_EXECUTION_RUNTIME.md`, `OBSERVABILITY.md`, `BACKGROUND_TASKS.md`, and related domain hubs). Semantic contracts and invariants remain owned by their canonical domain documents. Historical P0A evidence remains in the as-built audit; there is no separate CURRENT override document in `architecture/`.

---

# 2. Exit gate

| P0A gate | Result |
|---|---|
| Freeze audit baseline | PASS |
| Classify P0/P1 foundations | PASS |
| Identify concrete code/test evidence paths | PASS |
| Separate DOC-DRIFT from implementation GAP | PASS |
| Reconcile UER ExecutionId / ExecutionBoundary / RuntimeEvent status | PASS |
| Reconcile Execution-tree checkpoint status | PASS |
| Reconcile ContextProvider status | PASS |
| Reconcile Platform Plugins closed-program status | PASS |
| Reconcile Background Tasks implementation-existence status | PASS |
| Reconcile Skills status and authority gap | PASS |
| Reconcile ToolRuntime status and authority gap | PASS |
| Synchronize canonical architecture CURRENT sections | PASS |
| Remove transitional reconciliation document | PASS |
| Prevent implementation sessions from rebuilding shipped foundations | PASS |

**P0A exit gate: PASS.**

---

# 3. Real open blockers carried into P0B

P0A found two immediate code-changing safety blockers.

## P0-SAFETY-1 — Tool Authority Intersection Integrity

Current problem:

`resolve_allowed_tools_from_config(config, explicit=...)` can return the explicit caller allow-list without intersecting it with a stricter `RuntimePolicyBundle.tool_access` restriction.

Required property:

```text
effective tool authority
=
host availability
∩ agent/skill requirement
∩ RuntimePolicyBundle
∩ modality/plan narrowing
∩ invoker/per-call narrowing
```

No downstream allow-list may widen upstream authority.

## P0-SAFETY-2 — Skill Authority Integrity

Current problem:

`extend_tool_profile_for_skills()` can expand `ToolProfile.enabled` from Skill requirements.

Required property:

```text
skill-required tool_ids ⊆ host ToolProfile availability
```

Skills declare requirements. They do not grant host capability availability.

---

# 4. P0B start order

P0B must begin in this order unless a fresh repository audit invalidates the dependency assumption:

1. P0-SAFETY-1 Tool Authority Intersection Integrity.
2. P0-SAFETY-2 Skill Authority Integrity.
3. Cross-path monotonic authority conformance.
4. Meaningful-side-effect fresh enforcement coverage.
5. Required-causal-evidence-before-meaningful-work conformance.
6. Credential exposure/resolution boundary audit.
7. Sandbox fail-closed/isolation boundary audit.
8. Retry/redelivery authorization and side-effect fence audit.

No dynamic capability, external-agent, live-reconfiguration, or runtime-evolution feature should be allowed to depend on unresolved authority expansion behavior.

---

# 5. Instructions for implementation sessions

Before modifying code for P0B/P0C:

1. verify current `development` HEAD,
2. read the owning domain architecture section (canonical CURRENT state),
3. read the P0A as-built audit for historical evidence classification,
4. inspect the exact current code and tests,
5. define one bounded change,
6. preserve existing semantic ownership,
7. add executable regression/conformance evidence,
8. do not opportunistically rebuild neighboring foundations.

P0A is documentation/evidence closure only. The next phase is the first code-changing phase of this program.
