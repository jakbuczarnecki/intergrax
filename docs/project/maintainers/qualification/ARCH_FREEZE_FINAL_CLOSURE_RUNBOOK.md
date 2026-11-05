# ARCH-FREEZE final closure runbook

1. Confirm **AUDIT_BASELINE_SHA** (`a0624e28`) and **CLOSURE_DOC_HEAD** (`950097ea`); clean tree.
2. Wave 1 **ACCEPTED (SCOPED)** @ G6 `3d96691f` — do not reopen B0–B4 without material code delta.
3. Run **L3-ENT-FULL** @ wave EXECUTION_SHA when revalidating (wave-1 evidence @ `f7ce9dae`).
4. **EVOL B6** (external) → FRZ-EVOL promotion → **T-B5-EVOL-FORMAL-CLOSURE** (EVOL-X formal CLOSED publication).
5. **OD-ENT-00** (requires EVOL formal closure) → **ENT B6** → **OD-ENT-01/02** → FRZ-AUD → **OD-ENT-08**.
6. **T-B7-PFP-ACTIVATE** — implement/activate post-freeze change policy (FRZ-FRZ-10); complete material frozen-surface changes and qualification **before** freeze SHA lock.
7. **T-B7-LOCK-FREEZE-SHA** — set **FREEZE_CERTIFICATION_SHA** (= IMPLEMENTATION_SHA).
8. Run **L3-ENT-FREEZE** @ **FREEZE_CERTIFICATION_SHA**; **T-B7-VBN-HEAD-DOSSIER** may have been prepared @ closure baseline HEAD pre-lock — refresh rollup @ freeze SHA as needed.
9. **T-B7-VBN-INDEPENDENT-ACCEPT** (12/12 ENT VBN @ FREEZE_SHA) — certifying acceptance **after** lock only.
10. **ECAP §2.C** (external) @ FREEZE_SHA.
11. Promote remaining architecture-scoped FRZ (exclude Path B live-only); **T-B7-PROMOTE-RPL** only after VBN independent accept.
12. **T-B7-PFP-VERIFY** — certifying CHANGED-SINCE smoke @ **FREEZE_SHA** before **T-B7-FRZ-FRZ**.
13. **ECAP §15** admission (external) inside **T-B7-FRZ-FRZ**.
14. Publish roadmap §5 **ARCH-FREEZE — CLOSED — CERTIFIED** (**T-B8-ROADMAP**).
15. **ECAP §16** continuous enforcement handoff (**T-B8-ECAP-16**).

**Forbidden:** agent-signed B6; global PASS from scoped evidence; skip FREEZE @ cert SHA; **T-B7-PROMOTE-RPL** from **T-B2-VBN** alone; **T-B7-LOCK-FREEZE-SHA** before **T-B7-PFP-ACTIVATE**.
