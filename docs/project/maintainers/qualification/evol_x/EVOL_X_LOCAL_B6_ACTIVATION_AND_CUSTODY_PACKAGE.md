# EVOL-X — local B6 activation and custody package

**Status:** **PREPARED — ECAP-P4 NOT IN FORCE — FORMAL B6 PENDING**  
**Campaign:** `EVOL-X-LOCAL-B6-AUTOMATION` @ baseline `b669139d1…`  
**Ticket:** [`T-IGX-011-D.md`](../../../governance/tickets/T-IGX-011-D.md)

---

## 1. Effective vs pending approvals

| Approval | Effective? | Evidence |
|----------|------------|----------|
| Operator direction (OD-EVOL-14) | **YES** | Register + governance JSON |
| Path A mechanism (OD-EVOL-17) | **YES** | Active certification mechanism |
| OD-EVOL-18 technical acceptance | **YES** | Not Protocol 03 |
| ECAP maintainer publication | **NO** | [`EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md`](./EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md) unsigned |
| Security review sign-off | **NO** | [`EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md`](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md) open |
| Independent key custody | **NO** | Requires custodian provisioning (see below) |
| `independent_attestation_environment` | **NO** | Governance JSON `false` |
| OD-EVOL-11 `auditor_id` (Path A human) | **NO** | Register row |

**Forbidden in implementer session:** set `ecap_amendment_activation: IN_FORCE` or publish `B6_ACCEPTANCE_PUBLICATION_SHA`.

---

## 2. Implemented local mechanics (in-repo)

| Component | Path |
|-----------|------|
| Read-only attestor process | `scripts/qualification/evol_x_b6_local_attestor.py` |
| Custodian signer (external key path) | `scripts/qualification/evol_x_b6_custodian_sign.py` |
| Replay ledger (outside repo) | `scripts/qualification/evol_x_b6_replay_ledger.py` |
| Payload builder | `build_unsigned_attestation_payload` / `bind_and_sign_attestation` in `evol_x_b6_attestation.py` |
| Windows custody setup | `scripts/qualification/local_b6/` |
| Evaluate (existing) | `evol_x_b6_evaluate_signed_attestation.py` |

---

## 3. Administrator step (Windows custody)

Run **as Administrator** on operator workstation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& "D:\Projekty\intergrax\scripts\qualification\local_b6\Install-IntergraxEvolB6Custody.ps1"
```

Then create separate Windows accounts, grant ACLs per [`local_b6/README.md`](../../scripts/qualification/local_b6/README.md), perform key ceremony, publish trust root JSON to `public\`.

---

## 4. Operator activation record (template — post external approvals)

When maintainer + security + custody are **real**:

1. Complete maintainer act + security sign-off (signed artifacts).
2. Set governance JSON flags **only in publication commit authorized by maintainer**.
3. Run attestor @ pinned SHAs → custodian sign → evaluator `ELIGIBLE`.
4. Authorized publisher commits Protocol 03 PASS + `B6_ACCEPTANCE_PUBLICATION_SHA`.
5. W7-03..W7-06 sequence per master plan.

---

## 5. Technical evidence reuse

Pinned TES/FINAL/CI unchanged. R2 orchestrator report @ `.tmp/session/evol-x-r2-final-reconciliation/run-2026-11-04/` remains valid for Mode A+B technical replay unless verifier digest changes.

---

**Ref:** [`EVOL_X_LOCAL_B6_AUTOMATION_TERMINAL_REPORT.md`](./EVOL_X_LOCAL_B6_AUTOMATION_TERMINAL_REPORT.md)
