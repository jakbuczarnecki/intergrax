# ENT-AUDIT-X — W5 operator decision packet (final, pre–OD-ENT-08)

**Status:** `OD-ENT-08 RECORDED — W5 closure phase 1 published`  
**ENT-AUDIT-X formal closure:** IIAC PASS_SCOPED — roadmap headline published; `FORMAL_CLOSURE_PUBLICATION_SHA` pinned in W5 phase 2.  
**Campaign:** `ENT-AUDIT-X-FORMAL-CERTIFICATION-AND-CLOSURE` · **Ticket:** `governance/tickets/T-IGX-024-B.md`  
**Prepared:** 2026-11-11 (post PR #447 merge; Gate A replay on `93f4b3278…`)  
**Signing session:** `ent-sign-w5-od-08-2026-11-11` · [`ENT_AUDIT_X_OD_ENT_08_SIGNING_SESSION.md`](./ENT_AUDIT_X_OD_ENT_08_SIGNING_SESSION.md)

## 1. Authoritative HEAD and lineage

| Role | SHA |
| --- | --- |
| **`origin/development` HEAD (post #447)** | `93f4b32788d429ff1a935e59fdcae91b54503877` |
| Merge PR #447 (W5 OD-ENT-08 packet) | `93f4b32788d429ff1a935e59fdcae91b54503877` |
| Merge PR #446 (W4-03) | `bd026181583a54b408c05206ec194879d56b5734` |
| TEST_EXECUTION_SHA | `32a9633440007eb4e8f00a825bbf2e399192b346` |
| EVIDENCE_PUBLICATION_SHA (B5 FULL) | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` |
| W2_NORMAL_FREEZE_EVIDENCE_PUBLICATION_SHA | `bc4b5f61666a8fea638064f50809fff2089b1b8b` |
| ACCEPTED_BASELINE_SHA | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` |
| W3_INDEPENDENT_AUDIT_PUBLICATION_SHA | `fd5728955605ef2ec013d69ec9290736f0970419` |
| W4_INDEPENDENT_CERTIFICATION_PUBLICATION_SHA | `5f178174e2462b038d544b61192a666d5b0ca79f` |
| W4_03_FRZ_PROMOTION_PUBLICATION_SHA | `24bc86834f25e6f5914017accf84158111ae0ac5` |
| N-09 corrective (in merge #446) | `dd204cf8df523cde64704950dd49009fb76c3523` |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* |

## 2. Independent session identities (IIAC)

| Role | Session ID |
| --- | --- |
| W3 dossier auditor (operational) | `c8e4a91f-2b6d-4f7e-a3c1-9d5e7f2b4a68` |
| W3 publication auditor | `7c58500c-17c9-42a8-acbc-4b0e05487751` |
| W4 certifier | `4380025f-0a4e-43a2-bb90-ead2ce6d0bec` |

## 3. Accepted dispositions (unchanged SCOPED scope)

| Item | Disposition |
| --- | --- |
| W3 independent audit | **`PASS_SCOPED`** (not unscoped PASS) |
| W4 certification | **`PASS_SCOPED`** |
| OD-ENT-02 | **`SUFFICIENT_COMPENSATING_SCOPED`** · `od_ent_02_mode=compensating` |
| `frz_aud_status` | **`PASS_SCOPED`** |
| FRZ-AUD-01 … FRZ-AUD-09 | **`PASS`** |
| FRZ-AUD-10 | **`PASS — COMPENSATING (SCOPED)`** |
| Matrix C | **75/75 QUALIFIED** |
| Population E | **PASS_SCOPED** (85 semantic oracles @ TES′ L3) |

## 4. Gate A replay @ `93f4b32788d429ff1a935e59fdcae91b54503877`

| Check | Result |
| --- | --- |
| `ent_audit_x_formal_closure_preflight.py --gate A` | **PASS** (2026-11-11, post #447 merge) |
| IIAC policy | `IN_FORCE` @ `IIAC-2026-11-05.1` |
| PR #447 CI (mandatory) | **PASS** — merged to `development` |
| PR #446 CI (mandatory) | **PASS** (W4-03 + N-09 corrective) |

## 5. Scoped limitations (operator must accept with OD-ENT-08)

1. **OPEN-BP-01** — live branch protection on `development` remains unresolved; compensating controls are **not** equivalent to LIVE protection.  
2. **F-W3-001** — `freeze_normal_at_<TES>` strict B5 directory naming mismatch (artifact bytes and indices OK).  
3. **Path B / FRZ-PRD** — live qualification slice remains **OPEN** (carried in L3 report `limitations`).  
4. **Assurance** — **`process_enforced`** only; no claim of cryptographic or unlimited enterprise closure.

## 6. Evidence index

- W3 audit tree: `docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/audit/w3_dc778d/`  
- W4 certifier tree: `docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/certification/w4_dc778d/`  
- Canonical manifest: `docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/certification/iiac_audit_session_manifest.ENT-AUDIT-X.json`  
- Closure status: `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json`  
- OD-ENT-02 sub-manifest: `docs/project/maintainers/qualification/ent_audit_x/od_ent_02_compensating.v1.json`

## 7. Intended formal designation (only after OD-ENT-08 + Gate B)

`ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED`  
Per IIAC COMPENSATING path — **not** in force until operator records OD-ENT-08 and W5/W6 closure publication completes with **Gate B PASS**.

## 8. OD-ENT-08 — operator signing block (template only)

See companion file: [`ENT_AUDIT_X_OD_ENT_08_SIGNING_REQUEST.md`](./ENT_AUDIT_X_OD_ENT_08_SIGNING_REQUEST.md).  
**This packet does not contain an approval line.**
