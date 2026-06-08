**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix J — Phase V remediation traceability (audit gap → V-REM ID)

**Purpose:** 100% mapping from **Partial** audit findings (2026-06-05) to concrete remediation IDs. **Canonical phase narrative:** [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout).

**Status:** **12 tasks** · **12 Done** (2026-06-05).

### J.1 Audit gap → remediation matrix

| Audit source | Layer / area | Gap | Severity | Parent plan ID | V-REM ID | Status |
|--------------|--------------|-----|----------|----------------|----------|--------|
| Plan/code audit 2026-06-05 | Capability graph (AUDIT_MAP §19) | System edges agents→application incorrect per host | **Critical** | V-CG.2, V-CG.3, V-CG.4 | V-REM-CG.1, V-REM-CG.2 | **Done** |
| Plan/code audit 2026-06-05 | Agent lifecycle (AUDIT_MAP §31) | Governance contracts exist; no runtime routing cutoff for retired/deprecated | High | V-ALG.3 | V-REM-ALG.1 | **Done** |
| Plan/code audit 2026-06-05 | Agent lifecycle (AUDIT_MAP §31) | Ownership contracts exist; no production-eligible filter at selection | High | V-ALG.4 | V-REM-ALG.2 | **Done** |
| Plan/code audit 2026-06-05 | Prompt registry (AUDIT_MAP §17) | PromptMeta missing owner/risk; no YAML assets for E2E validation | High | V-PE.1 | V-REM-PE.1, V-REM-PE.2 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Tool injection defense not wired on execution path | High | V-SEC.2 | V-REM-SEC.1 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Retrieval poisoning defense not enforced per tenant/app | High | V-SEC.3 | V-REM-SEC.2 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Tenant isolation + audit trail hooks missing in main path | High | V-SEC.4 | V-REM-SEC.3 | **Done** |
| Plan/code audit 2026-06-05 | Evaluation (AUDIT_MAP §25) | NexusEvalRunner exists; missing integration tests + gate | Medium | A.4, A.4.1 | V-REM-A.1 | **Done** |
| Plan sync 2026-06-05 | Plan governance | Appendix J + §6.1z queue + status sync | — | — | V-REM.0.1, V-REM.0.2 | **Done** |

**Coverage target:** 100% **Done** when every **Planned** row is **Done** and parent Partial IDs (V-CG.2–4, V-ALG.3–4, V-PE.1, V-SEC.2–4, A.4) are **Done**.

### J.2 Paydown log

| Date | V-REM ID | Summary |
|------|----------|---------|
| 2026-06-05 | V-REM.0.1, V-REM.0.2 | Appendix J + Phase V-REM section + §6.1z/§6.2v + Appendix H sync |
| 2026-06-05 | V-REM-CG.1–A.1 | Runtime remediation: capability graph, lifecycle routing, V-SEC wiring, prompt governance, EvalRunner gate |
| 2026-06-05 | V-POST.1, V-POST.2 | Phase V closeout gate green; AgentEngine routability guard; NexusLoop tenant-security integration tests |

---
