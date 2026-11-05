# T-IGX-010-C — ARCH-FREEZE closure campaign (R0–R6)

## Scope

**Authorized:** R0 SSOT reconcile @ `950097ea`, WBS 54 + DAG fixes, VBN freeze SHA dossier shell, qualification doc updates, traceability validator WBS=54. **R1–R6 execution** only after operator **Approved** line and external gates (EVOL/ENT B6, OD-ENT-*, branch protection, FRZ promotions, FREEZE_CERTIFICATION_SHA lock, ECAP §2.C/§15). **No** FRZ checklist PASS promotion by agent; **no** ARCH-FREEZE certification claim until **T-B8-ROADMAP** after **T-B7-FRZ-FRZ**.

**SSOT:** [`docs/project/maintainers/qualification/ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md`](../../docs/project/maintainers/qualification/ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md)

## Plan digest

Plan digest: ARCH-FREEZE closure master plan §5 chain + WBS/DAG reconcile (PFP-ACTIVATE before LOCK-FREEZE-SHA; PFP-VERIFY before FRZ-FRZ) @ baseline `4a1ab152dc86e97fb017278310c553064a664b23`.

## Approved

Approved: OPERATOR APPROVED — ARCH-FREEZE R1–R6 CLOSURE CAMPAIGN, SCOPED AND GATE-CONDITIONAL.

Baseline: `4a1ab152dc86e97fb017278310c553064a664b23`.

Timestamp: 2026-11-04T16:05:00+01:00.

**Authorized in R1 (this campaign tranche):** SSOT/WBS/DAG reconcile; expand **ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER** §2.1–2.7 @ closure baseline HEAD; gap-register closure-path notes; mechanical `verify_arch_freeze_plan_traceability.py`. **Not authorized:** Protocol 03 **EVOL/ENT B6** (no agent-signed or fake B6); **FRZ-* checklist PASS/N/A promotion**; **T-B7-LOCK-FREEZE-SHA** / **FREEZE_CERTIFICATION_SHA** lock commit; **OD-ENT-*** operator packets; GitHub branch-protection changes; **ECAP §15** admission; roadmap §5 **ARCH-FREEZE — CLOSED — CERTIFIED**; global ARCH-FREEZE certification claims.

**Hard STOP gates (implementer must halt until external/operator gate clears):**

1. **T-B5-EVOL-B6** — independent EVOL Protocol 03 B6 + OD-EVOL-11 path; no self-certify.
2. **T-B6-ENT-B6** and **T-B6-OD-ENT-00/01/02/08** — ENT formal chain + operator/GitHub admin only.
3. **FRZ family promotions** (EVOL, AUD, RPL, …) — operator+auditor authorized rows only; agent must not edit checklist to PASS.
4. **T-B7-PFP-ACTIVATE** then material frozen-surface work qualified — **before** **T-B7-LOCK-FREEZE-SHA** (post-freeze policy implementation/activation precedes freeze SHA lock).
5. **T-B7-LOCK-FREEZE-SHA** — single operator+implementer lock commit; **no** lock in R1.
6. **T-B7-VBN-INDEPENDENT-ACCEPT** — certifying ENT VBN @ exact **FREEZE_CERTIFICATION_SHA** only after lock (HEAD dossier prep may precede lock).
7. **T-B7-FRZ-FRZ** — requires **T-B7-PFP-VERIFY**, **T-B7-COORD-ARCH-FRZ**, **T-B7-ECAP-2C**; ECAP §15 external admission inside FRZ-FRZ task.
8. **T-B8-ROADMAP** — publication only after **T-B7-FRZ-FRZ**; **T-B8-ECAP-16** post-closure handoff after roadmap.

## Test contract

```bash
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py -q
uv run pytest tests/qualification/ebh_5/ -q
uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --output-dir .tmp/proof/arch-freeze-full/ --evidence-bundle
uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FREEZE --output-dir .tmp/proof/arch-freeze-freeze/ --evidence-bundle
uv run python -m scripts.governance.igx_gov ticket
```

Evidence pointers: [`ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md`](../../docs/project/maintainers/qualification/ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md); wave-1 G6 [`ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md`](../../docs/project/maintainers/qualification/ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md); qual RUN_IDs in [`ARCH_FREEZE_QUALIFICATION_AND_EVIDENCE_PLAN.md`](../../docs/project/maintainers/qualification/ARCH_FREEZE_QUALIFICATION_AND_EVIDENCE_PLAN.md).

**Commit trailer:** `Ref: governance/tickets/T-IGX-010-C.md` on commits touching `docs/project/maintainers/qualification/ARCH_FREEZE_*`, `governance/tickets/T-IGX-010-C.md`, or `scripts/governance/verify_arch_freeze_plan_traceability.py`.

## Lifecycle state

**operator_approved_r1_r6_scoped** @ baseline `4a1ab152dc86e97fb017278310c553064a664b23` (2026-11-04) — R1 SSOT/dossier/DAG tranche authorized; wave **B0–B4** remains **independent_g6_accepted_scoped** under **T-IGX-010-B** @ G6 `3d96691f`. Residual BLOCKERs until external gates close: **FIND-VBN-PARTIAL** (→ **T-B7-VBN-*** @ FREEZE_SHA), **FIND-FORMAL-B6**, **FIND-ACCEPTED-BASELINE**. **FREEZE_CERTIFICATION_SHA:** TBD (not locked in R1).
