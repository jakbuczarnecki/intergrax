# ACT-N11 Session B — Operator activation dossier (template)

**Status:** **TEMPLATE** — publish after independent auditor **PASS**; **STOP** before Operator personal activation.  
**Policy:** `IIAC-2026-11-05.1`  
**Decision:** `D-IIAC-01=a` (shared full re-activation)  
**Ticket:** `governance/tickets/T-IGX-024-B.md`

---

## 1. Preconditions (auditor fills before Operator turn)

| Check | Evidence path | Status |
|---|---|---|
| Independent auditor PASS (scoped) | `IIAC_ENT_AUDIT_REPORT.md` or successor under this directory | `PENDING` |
| Session manifest valid | `iiac_audit_session_manifest.ENT-AUDIT-X.act_n11_session_b.json` | `PENDING` |
| Role separation @ manifest | `verify_iiac_role_separation` log | `PENDING` |
| Seven-path Δ reviewed | `audit_scope.json` + auditor diff replay | `PENDING` |
| Implementer did not self-PASS | No `*_PASS_RECORD.md` from implementer session | `VERIFIED` |

---

## 2. Pinned SHAs (update at activation time)

| Pin | SHA |
|---|---|
| Prior `audit_execution_sha` | `0946b11576d2ee617321fb099ed794f8b21bf5ba` |
| New `audited_implementation_sha` | `________________` (auditor publication) |
| New `audit_execution_sha` | `________________` (auditor read-only tip) |
| `development` activation tip | `________________` (post two-phase publication) |

---

## 3. Material paths — Operator / certifier only

**Forbidden for implementer:** edit of:

- `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json`
- Six `MATERIAL_GOVERNANCE_PATHS` in `scripts/governance/iiac_activation_gate.py` (without governing `Approved:` line)

Two-phase publication per [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../../../INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md) § Policy activation.

---

## 4. Operator decision file (create when authorized)

Filename pattern: `IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md` (or successor) under `iiac_evidence/` or operator decisions tree.

Required fields: decision id, policy version, scope `IIAC policy activation`, `Decision status: AUTHORIZED`, audit report digest, and **exact** activation line below.

---

## 5. Operator signature line

```
Approved IIAC policy activation IIAC-2026-11-05.1: operator @ ________________
```

**(EMPTY — Operator must fill ISO-8601 timestamp in a future turn; implementer MUST NOT substitute.)**

---

## 6. Post-activation verification (future turn — when green)

Execute only after §5 line is published in a governing decision file:

```powershell
.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force
.venv/Scripts/python.exe -m scripts.governance.igx_gov ticket
```

| Gate | Expected |
|---|---|
| `verify_iiac_policy_activation --require-in-force` | exit **0** |
| `governance-fast` on `development` tip | **SUCCESS** |
| PRE_ENT handoff `PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md` | Update **AUDIT_HEAD / SHA only** when verify exit 0 |

**Do not** start PRE-ENT Wave 3 until the above are green and Operator executes resume checklist [`PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md`](../../../pre_ent_enterprise_certification/PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md).

---

## 7. Cross-links

- Recovery SSOT: [`IIAC_ACT_N11_RECOVERY_BATCH.md`](../../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md)
- Auditor handoff: [`IIAC_SESSION_HANDOFF_RECORD_ACT_N11_SESSION_B.md`](./IIAC_SESSION_HANDOFF_RECORD_ACT_N11_SESSION_B.md)
- Shared agreement: [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](../../../historical_certification_campaign/HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md)
