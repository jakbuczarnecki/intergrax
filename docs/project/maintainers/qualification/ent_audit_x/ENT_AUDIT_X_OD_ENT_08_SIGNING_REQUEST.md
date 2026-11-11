# ENT-AUDIT-X — OD-ENT-08 signing request (operator only)

**Decision ID:** OD-ENT-08  
**Stage:** ENT-AUDIT-X formal closure authorisation  
**Policy:** IIAC-2026-11-05.1 (`IN_FORCE`)  
**Gate A @ HEAD:** `bd026181583a54b408c05206ec194879d56b5734` — **PASS**  
**Prerequisite packet:** [`ENT_AUDIT_X_W5_OPERATOR_DECISION_PACKET.md`](./ENT_AUDIT_X_W5_OPERATOR_DECISION_PACKET.md)

---

## Operator decision (copy into signing session record)

By recording the line below, the operator authorises **W5/W6 formal closure publication only** under the **COMPENSATING (SCOPED)** disposition. This is **not** ARCH-FREEZE closure and **not** a waiver of OPEN-BP-01 or Path B limitations.

```
Approved OD-ENT-08: operator-delegate @ <ISO-8601-UTC>
Scope: ENT-AUDIT-X formal stage closure under IIAC PASS_SCOPED chain @ development HEAD bd026181583a54b408c05206ec194879d56b5734
Accepts: FRZ-AUD-01..09 PASS; FRZ-AUD-10 PASS — COMPENSATING (SCOPED); Matrix C 75/75; Population E PASS_SCOPED; W3/W4 independent PASS_SCOPED; OPEN-BP-01, F-W3-001, Path B FRZ-PRD OPEN, process_enforced assurance
Authorises: W5 closure phase 1/2, roadmap §5 headline, FORMAL_CLOSURE_PUBLICATION_SHA pin, Gate B verification — not before this timestamp
Target designation: ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED
Delegation: <path-to-O-1-or-OD-ENT-08-delegation-if-published>@<sha>
```

**Non-delegable:** IIAC policy activation, this OD-ENT-08 authorisation itself (unless operator personally signs), and cryptographic claims.

**Forbidden without a new OD:** upgrading scope to unscoped PASS, closing ARCH-FREEZE, or removing OPEN-BP-01 from the formal record.

---

## After approval (implementer/certifier only — not before Approved line exists)

1. Publish OD-ENT-08 decision record under `iiac_evidence/ENT-AUDIT-X/operator_decisions/`.  
2. Execute W5 closure phase 1 (roadmap, status JSON, terminal dossier) — **no** premature headline before Gate A rules satisfied on each commit.  
3. Pin `FORMAL_CLOSURE_PUBLICATION_SHA` on ladder.  
4. Run `ent_audit_x_formal_closure_preflight.py --gate B`.  
5. Prepare ARCH-FREEZE handoff (W6); do **not** mark ARCH-FREEZE CLOSED.

---

**Status:** UNSIGNED — awaiting operator.
