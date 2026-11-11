# ENT-AUDIT-X — W5 operator decision packet (consolidated)

**Status:** `W4-03 COMPLETE — GATE A TARGET — STOP before OD-ENT-08`  
**ENT-AUDIT-X is NOT closed.** `FORMAL_CLOSURE_PUBLICATION_SHA` unset. **`OD-ENT-08` not recorded** — do not infer operator approval.  
**Campaign:** `ENT-AUDIT-X-FORMAL-CERTIFICATION-AND-CLOSURE` · **Ticket:** `governance/tickets/T-IGX-024-B.md`  
**Prepared:** 2026-11-11 (certification publication lane, W4-03)

## 1. SHA ladder (authoritative pins)

| Role | SHA |
| --- | --- |
| Campaign baseline HEAD @ W4 | `dc778d125fd7e7205edd30fbbf144ca12320b0ee` |
| TEST_EXECUTION_SHA | `32a9633440007eb4e8f00a825bbf2e399192b346` |
| EVIDENCE_PUBLICATION_SHA (B5 FULL) | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` |
| W2_NORMAL_FREEZE_EVIDENCE_PUBLICATION_SHA | `bc4b5f61666a8fea638064f50809fff2089b1b8b` |
| ACCEPTED_BASELINE_SHA | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` |
| W3_INDEPENDENT_AUDIT_PUBLICATION_SHA | `fd5728955605ef2ec013d69ec9290736f0970419` |
| W4_INDEPENDENT_CERTIFICATION_PUBLICATION_SHA | `5f178174e2462b038d544b61192a666d5b0ca79f` |
| B6_ACCEPTANCE_PUBLICATION_SHA (phase-1) | `add6a6ce2015ebaa4bc38b7fadf3936f100a1522` |
| W4_03_FRZ_PROMOTION_PUBLICATION_SHA | `24bc86834f25e6f5914017accf84158111ae0ac5` |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* |

## 2. Independent session identities

| Role | Session ID |
| --- | --- |
| W3 dossier auditor (operational) | `c8e4a91f-2b6d-4f7e-a3c1-9d5e7f2b4a68` |
| W3 publication auditor | `7c58500c-17c9-42a8-acbc-4b0e05487751` |
| W4 certifier | `4380025f-0a4e-43a2-bb90-ead2ce6d0bec` |

## 3. Verdicts and dispositions

| Item | Disposition |
| --- | --- |
| W3 independent audit | `PASS_SCOPED` |
| W4 certification | `PASS_SCOPED` |
| OD-ENT-02 sufficiency | `SUFFICIENT_COMPENSATING_SCOPED` |
| `frz_aud_status` | `PASS_SCOPED` (W4-03 promoted) |
| FRZ-AUD-01..09 | `PASS` |
| FRZ-AUD-10 | `PASS — COMPENSATING (SCOPED)` |
| Matrix C | **75/75 QUALIFIED** |
| Population E | **PASS_SCOPED** (85/85 L3 @ TES′) |

## 4. Scoped limitations (preserved)

1. **OPEN-BP-01** — live `development` branch protection not equivalent to compensating package.  
2. **F-W3-001** — `freeze_normal_at_<TES>` strict B5 directory naming (bytes OK).  
3. **Path B** — FRZ-PRD live qualification dimension remains OPEN per L3 `limitations`.  
4. **Assurance** — `process_enforced`; not cryptographic closure.

## 5. Compensating controls (OD-ENT-02 summary)

- Ruleset `Protect development` (deletion + non-fast-forward only).  
- ENT workflows: `ent-audit-pr-gate`, `ent-audit-fast`; CI run `42869847981` @ TES′.  
- Gate A/B preflight wiring (`ent_audit_x_formal_closure_preflight.py`).  
- `verify_od_ent_02_compensating_control.py` @ published=true when compensating.

## 6. Formal closure scope (for OD-ENT-08 only — unsigned)

**Target designation (if operator authorizes):** `ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED` per IIAC migration matrix **COMPENSATING** path — **not** claimed until OD-ENT-08 recorded and Gate B green.

**Includes:** roadmap §5 headline (Gate B), `FORMAL_CLOSURE_PUBLICATION_SHA`, ARCH-FREEZE handoff prep (W6; **not** ARCH-FREEZE CLOSED).

## 7. Operator actions required

1. Confirm **Gate A** green @ W4-03 pin `24bc86834…` (`ent_audit_x_formal_closure_preflight.py --gate A` → **PASS** on certifier replay).  
2. Record **`Approved OD-ENT-08: …`** in a dedicated signing session (non-delegable closure authorisation per plan).  
3. Only then authorize W5 closure phase 1/2 (roadmap, terminal dossier, Gate B).

**STOP:** This packet is **not** an OD-ENT-08 decision and **does not** close ENT-AUDIT-X.

## 8. Evidence index

- W3: `iiac_evidence/ENT-AUDIT-X/audit/w3_dc778d/`  
- W4: `iiac_evidence/ENT-AUDIT-X/certification/w4_dc778d/`  
- Canonical manifest: `certification/iiac_audit_session_manifest.ENT-AUDIT-X.json`  
- Handoff: `certification/w4_dc778d/W5_CERTIFIER_HANDOFF_RECORD.md`
