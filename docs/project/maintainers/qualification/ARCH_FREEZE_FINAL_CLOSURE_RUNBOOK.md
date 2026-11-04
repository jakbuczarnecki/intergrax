# ARCH-FREEZE final closure runbook

1. Confirm **AUDIT_BASELINE_SHA** and clean tree.
2. Complete B2 remediation (BND03 reconcile, VBN, DEBT, DOC).
3. Run **L3-ENT-FULL** @ IMPLEMENTATION_SHA → store under `.tmp/proof/`.
4. **EVOL B6** (external) → FRZ-EVOL promotion.
5. **OD-ENT-00** → **ENT B6** → **OD-ENT-01/02** → FRZ-AUD → **OD-ENT-08**.
6. Lock **FREEZE_CERTIFICATION_SHA**.
7. Run **L3-ENT-FREEZE** + **ECAP §2.C** (external).
8. Promote remaining architecture-scoped FRZ (exclude Path B live-only).
9. **ECAP §15** admission (external).
10. Publish roadmap §5 **ARCH-FREEZE — CLOSED — CERTIFIED**.
11. Enable **ECAP §16** continuous enforcement.

**Forbidden:** agent-signed B6; global PASS from scoped evidence; skip FREEZE @ cert SHA.
