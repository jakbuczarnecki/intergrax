# ACT-N11 MCP — Operator activation dossier

**Status:** **F-MCP-ACT-N11-001 CORRECTIVE CANDIDATE ON PR #454** — prior audit **PARTIAL_EXACT_BLOCKERS** preserved; **new independent IIIC re-audit REQUIRED** (no self-PASS).  
**Policy:** `IIAC-2026-11-05.1`  
**Decision (required):** `D-MCP-IIIC-01=a`  
**Ticket:** `governance/tickets/T-IGX-035-E.md`  
**Operator decision file (prepared):** [`IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md`](../../../IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md)

---

## 1. Preconditions (not yet satisfied)

| Check | Status |
| --- | --- |
| Independent auditor `PASS_SCOPED` on MCP scope | **NOT MET** (`PARTIAL_EXACT_BLOCKERS`) |
| Session manifest `iiac_audit_session_manifest.MCP-SERVER-X.act_n11_mcp.json` | **PUBLISHED** (verdict FAIL / org PARTIAL) |
| Role separation @ manifest | **PASS** (`verify_iiac_role_separation --strict-product-diff`) |
| Implementer did not self-PASS | **VERIFIED** (no implementer pass record) |

---

## 2. Pinned SHAs (audit targets)

| Pin | SHA |
| --- | --- |
| Prior `audit_execution_sha` | `11ac8ad85f62a6925d25ea840c92d5c6a936cfe4` |
| `origin/development` (P2/P3 Δ only) | `7172f294fdd44fe35024ba60378fb9f4e0125963` |
| PR #454 head (P2–P4 full Δ) | `7c401fd98cf2b42b9b2fd5860485bac34a8c2df3` |
| P4 product anchor | `a21a6629a6f8aa25519d4e25ac0004b38426669b` |

Scope manifest: [`audit_scope.json`](./audit_scope.json)

---

## 3. Material paths — Operator / certifier only

**Forbidden for implementer and publisher:** edit of `IIAC_POLICY_ACTIVATION_RECORD.json` and six `MATERIAL_GOVERNANCE_PATHS` without governing `Approved:` line.

---

## 4. Operator signature line

```
Approved IIAC policy activation IIAC-2026-11-05.1: operator @ ________________
```

**(EMPTY — Operator must fill ISO-8601 timestamp; implementer MUST NOT substitute.)**

---

## 5. Post-activation verification (after §4 only)

```powershell
.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force
```

Expected: exit **0** on merge target; `governance-fast` **SUCCESS** on PR #454.
