# EQO-C-01 — EQOA-A3-004 disposition evidence

**Pin:** `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` (DOC_PIN)  
**Finding:** HIST-85 worker dispositions; independent sample **33** rows → **30 agree**, **3 disagree**.

## Disputed rows (dossier A3 / `hist85.jsonl`)

| FRZ ID | EQO sample vs worker | EQO disposition |
|--------|----------------------|-----------------|
| **FRZ-BND-05** | Downgrade (`VALID_SCOPED_PASS_CANDIDATE` → `QUALIFICATION_GAP`) | **DISAGREE sustained** — mapped gate tests import direction only, not sanctioned composition (E3). |
| **FRZ-PRD-07** | Undetermined in sample | **OPEN** — strict-path evidence partial at A baseline; not re-adjudicated on TRAIN_PIN in this pass. |
| **FRZ-BND-01** | Possible upgrade | **DEFERRED** — requires independent gate replay, not dossier re-read alone. |

## Verdict (30/33 sample)

On **DOC_PIN**, the mechanical **30/33 agreement rate remains binding** for Campaign A dossier scope. The three disagreements are **not** closed; they require Campaign C targeted replay (HIST/C-04), not checklist promotion.

**EQO disposition:** `ACCEPTED_SCOPED_LIMITATION` — sample stands; three rows carry forward as `BLOCKED` for freeze until replay.

**Audit:** session E pending (not executed in this implementer session).