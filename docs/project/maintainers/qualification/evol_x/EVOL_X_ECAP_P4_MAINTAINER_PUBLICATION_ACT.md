# ECAP-P4-EVOL-SCOPE-AUTO-B6 — maintainer publication act

**Amendment ID:** `ECAP-P4-EVOL-SCOPE-AUTO-B6`  
**Amendment version (candidate):** `2026-11-02.1` (drop `-pending` only upon this act)  
**Normative text:** [EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md](./EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md)

---

## Publication status

| Field | Value |
|-------|--------|
| **Status** | **AWAITING_ECAC_MAINTAINER_PUBLICATION** |
| **`ecap_amendment_activation`** | **NOT IN FORCE** (implementer commit does not activate) |
| **Governance JSON** | Remains `PENDING_REQUIRED_APPROVALS` / `NOT_IN_FORCE` until this act is completed |

---

## Maintainer publication act (to be completed by ECAP maintainer)

The undersigned ECAP maintainer declares that amendment **`ECAP-P4-EVOL-SCOPE-AUTO-B6`** version **`2026-11-02.1`** is **`IN_FORCE`** for EVOL-X `AUTOMATION_FIRST_SCOPED` closure only, subject to:

1. Signed independent security review on file.
2. Independent key custody operational.
3. No implementer self-activation of this act via code commit alone.

| Field | Value |
|-------|--------|
| Maintainer identity | *TBD — not implementer agent* |
| Publication date (UTC) | *TBD* |
| Publication channel | *e.g. maintainer signed record + ECAP doc cross-reference update* |
| Revocation procedure | Maintainer signed revocation → governance `NOT_IN_FORCE`; attestor fail-closed |

**Signature:** ___________________________  **Date:** __________

---

## Post-publication repository updates (authorized after signature only)

1. Update [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) ECAP-§2.B footnote → `IN_FORCE` @ version `2026-11-02.1`.
2. Update [EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json](./EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json): `ecap_amendment_activation: "IN_FORCE"`, `normative_effect: "IN_FORCE"`, `ecap_amendment_version: "2026-11-02.1"`.
3. Record act hash / maintainer record id in [EVOL_X_BATCH1_EXECUTION_RECORD.md](./EVOL_X_BATCH1_EXECUTION_RECORD.md).

**Forbidden before signature:** setting governance `IN_FORCE` in implementer session.
