# ACT-N11 Session B — Operator activation dossier

**Status:** **READY FOR OPERATOR** — independent auditor `PASS_SCOPED` published; **STOP** before personal IIAC activation.  
**Policy:** `IIAC-2026-11-05.1`  
**Decision:** `D-IIAC-01=a` (shared full re-activation)  
**Ticket:** `governance/tickets/T-IGX-024-B.md`  
**Operator decision file (prepared):** [`IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md`](../../../IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md)

---

## 1. Preconditions (publisher verified)

| Check | Evidence path | Status |
|---|---|---|
| Independent auditor PASS (scoped) | `IIAC_ACT_N11_SESSION_B_INDEPENDENT_AUDIT_DOSSIER.md` | **VERIFIED** (`PASS_SCOPED`) |
| Session manifest valid | `iiac_audit_session_manifest.ENT-AUDIT-X.act_n11_session_b.json` | **VERIFIED** (`verify_iiac_audit_manifest` PASS) |
| Role separation @ manifest | `auditor_manifest_and_role_separation.log` | **VERIFIED** (`verify_iiac_role_separation --strict-product-diff` PASS) |
| Seven-path Δ reviewed | `audit_scope.json` + `auditor_product_delta_seven_path_diff_stat.log` | **VERIFIED** (+60 −19) |
| Implementer did not self-PASS | No implementer `*_PASS_RECORD.md` | **VERIFIED** |

---

## 2. Pinned SHAs (evidence publication — update after merge)

| Pin | SHA |
|---|---|
| Prior `audit_execution_sha` (ACT-N11) | `0946b11576d2ee617321fb099ed794f8b21bf5ba` |
| `audited_implementation_sha` (product PR #434) | `03aa10959acf6ec3d81a7a0de8ac8224c8126973` |
| Auditor `audit_execution_sha` / `test_execution_sha` | `ebc2f8ba0f92599a1fdde80cde6ad48beab1934f` |
| `evidence_publication_sha` | `e2bb6e9c0dab0b603f98fb6ffa11ff463da19c0b` (phase-1 evidence commit) |
| `certification_publication_sha` | `0be70e1506200f2ef074c1e180997d1727709713` (phase-2 manifest pin) |
| `development` tip (pre-activation) | `ebc2f8ba0f92599a1fdde80cde6ad48beab1934f` → post-evidence merge tip |

Digest pins (manifest):

| Artifact | SHA-256 |
|---|---|
| Audit report | `c47f43c16c7a5cd8452f749ac60ce9a61e8921b7fd30321521d2f8c14d6b0ac2` |
| Audit scope | `e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f` |
| Test matrix | `d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b` |
| Findings | `f4c5b211a9d3a0f0f5f53dfcf5551d5a7310755c1953c854f48e1d07abb3f2d7` |

---

## 3. Material paths — Operator / certifier only

**Forbidden for implementer and publisher:** edit of:

- `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json`
- Six `MATERIAL_GOVERNANCE_PATHS` in `scripts/governance/iiac_activation_gate.py` (without governing `Approved:` line)

Two-phase activation record publication per [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../../../INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md) § Policy activation — **after** §5 operator line.

---

## 4. Operator decision file

Prepared: [`IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md`](../../../IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md) (`Decision status: PREPARED`).

Required after authorization: set `Decision status: AUTHORIZED` and publish the exact activation line in §5 below (Operator only).

---

## 5. Operator signature line

```
Approved IIAC policy activation IIAC-2026-11-05.1: operator @ ________________
```

**(EMPTY — Operator must fill ISO-8601 timestamp; publisher and implementer MUST NOT substitute.)**

---

## 6. Post-activation verification (Operator / certifier turn — not now)

Execute only after §5 line is published in the governing OD file:

```powershell
.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force
.venv/Scripts/python.exe -m scripts.governance.igx_gov ticket
```

| Gate | Expected |
|---|---|
| `verify_iiac_policy_activation --require-in-force` | exit **0** |
| `governance-fast` on `development` tip | **SUCCESS** |
| PRE_ENT handoff `PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md` | Update **AUDIT_HEAD / SHA only** when verify exit 0 |

**Do not** start PRE-ENT Wave 3 until the above are green.

---

## 7. Cross-links

- Recovery SSOT: [`IIAC_ACT_N11_RECOVERY_BATCH.md`](../../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md)
- Auditor handoff: [`IIAC_SESSION_HANDOFF_RECORD_ACT_N11_SESSION_B.md`](./IIAC_SESSION_HANDOFF_RECORD_ACT_N11_SESSION_B.md)
- Shared agreement: [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](../../../historical_certification_campaign/HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md)
