# ENT-AUDIT-X — OD-ENT-08 canonical signing session (W5)

| Field | Value |
|-------|-------|
| Signing session id | `ent-sign-w5-od-08-2026-11-11` |
| Decision ID | **OD-ENT-08** |
| Policy | `IIAC-2026-11-05.1` (`IN_FORCE`) |
| Ticket | `governance/tickets/T-IGX-024-B.md` |
| Status | **RECORDED** — `ENT_AUDIT_X_OD_ENT_08_DECISION_RECORD.md` @ `2026-11-11T07:20:00Z` |
| Assurance | `process_enforced` |

## 1. Session dedication

This is a **new** session dedicated solely to **OD-ENT-08** (W5 formal closure authorisation). It is **not** ARCH-FREEZE closure. It does **not** continue implementer W3/W4 audit or certification sessions.

**Operator commission:** standing auth `T-IGX-024-A` + W5 continuation chat 2026-11-11 (ENT-AUDIT-X W5; STOP before operator approval).

**Prior signing session (OD-ENT-00/02 only):** `ent-sign-w3-2026-11-09` — OD-ENT-08 deliberately paused until Gate A prerequisites satisfied.

## 2. Authoritative development HEAD pin

| Role | SHA |
|------|-----|
| `origin/development` HEAD (post PR **#447** merge) | `93f4b32788d429ff1a935e59fdcae91b54503877` |
| Prior HEAD (post #446) | `bd026181583a54b408c05206ec194879d56b5734` |
| Gate A replay @ `93f4b3278…` | **PASS** (`ent_audit_x_formal_closure_preflight.py --gate A`, 2026-11-11) |

**Operator rule:** record `Approved OD-ENT-08` only against the **current** `origin/development` HEAD after any session-prep merge; replay Gate A on that SHA immediately before signing.

## 3. Independent W3 / W4 evidence (accepted `PASS_SCOPED` only)

| Layer | Publication SHA | Session ID | Canonical tree |
|-------|-----------------|------------|----------------|
| W3 independent audit | `fd5728955605ef2ec013d69ec9290736f0970419` | `7c58500c-17c9-42a8-acbc-4b0e05487751` | `iiac_evidence/ENT-AUDIT-X/audit/w3_dc778d/` |
| W3 dossier (operational) | — | `c8e4a91f-2b6d-4f7e-a3c1-9d5e7f2b4a68` | `.tmp/session/ent-w2-audit/` (dossier; pub bytes on-tree) |
| W4 certification | `5f178174e2462b038d544b61192a666d5b0ca79f` | `4380025f-0a4e-43a2-bb90-ead2ce6d0bec` | `iiac_evidence/ENT-AUDIT-X/certification/w4_dc778d/` |

**Bound manifest:** `iiac_evidence/ENT-AUDIT-X/certification/iiac_audit_session_manifest.ENT-AUDIT-X.json` @ `93f4b3278…` (or successor HEAD at sign time).

**IIAC pass record (not closure):** `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md`

## 4. Accepted SCOPED dispositions (operator must not upgrade)

| Item | Disposition |
|------|-------------|
| W3 / W4 verdict | **`PASS_SCOPED`** (not unscoped PASS) |
| OD-ENT-02 | **`SUFFICIENT_COMPENSATING_SCOPED`** · `od_ent_02_mode=compensating` |
| `frz_aud_status` | **`PASS_SCOPED`** (W4-03 promotion) |
| FRZ-AUD-01 … 09 | **PASS** |
| FRZ-AUD-10 | **PASS — COMPENSATING (SCOPED)** |
| Matrix C | **75/75 QUALIFIED** |
| Population E | **PASS_SCOPED** |

## 5. Scoped limitations (explicit acceptance)

1. **OPEN-BP-01** — live branch protection on `development` unresolved.  
2. **F-W3-001** — `freeze_normal_at_<TES>` strict B5 directory naming mismatch.  
3. **Path B / FRZ-PRD** — live qualification slice **OPEN**.  
4. No cryptographic or unlimited enterprise closure — **`process_enforced`** only.

## 6. Delegation and companion artifacts

| Artifact | Path |
|----------|------|
| O-1 delegation (OD-ENT-08 in scope) | `iiac_evidence/ENT-AUDIT-X/operator_decisions/ENT_AUDIT_X_O1_SIGNING_DELEGATION_RECORD.md` |
| W5 operator packet | `ENT_AUDIT_X_W5_OPERATOR_DECISION_PACKET.md` |
| Signing request (operator copy/paste block) | `ENT_AUDIT_X_OD_ENT_08_SIGNING_REQUEST.md` |
| Closure status | `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json` |

## 7. Target designation (not in force until OD-ENT-08 + W5/W6 + Gate B)

`ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED`

## 8. Hard stops (this session)

- Do **not** fill `Approved OD-ENT-08` in git without operator chat line.  
- Do **not** set `FORMAL_CLOSURE_PUBLICATION_SHA`, roadmap §5 CLOSED headline, or ARCH-FREEZE CLOSED.  
- Do **not** run W5 closure phase 1/2 or Gate B publication until OD-ENT-08 is recorded per `ENT_AUDIT_X_OD_ENT_08_SIGNING_REQUEST.md` §After approval.
