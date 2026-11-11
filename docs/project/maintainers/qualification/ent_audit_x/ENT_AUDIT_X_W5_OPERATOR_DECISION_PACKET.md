# ENT-AUDIT-X — W5 operator decision packet (consolidated)

**Status:** `W4 COMPLETE — W5 OPERATOR GATE READY`  
**ENT-AUDIT-X is NOT closed.** `FORMAL_CLOSURE_PUBLICATION_SHA` unset. `OD-ENT-08` not recorded.  
**Campaign:** `ENT-AUDIT-X-FORMAL-CERTIFICATION-AND-CLOSURE` · **Ticket:** `governance/tickets/T-IGX-024-B.md`  
**Prepared:** 2026-11-11 (orchestrator, post W4 Batch D)

## 1. SHA ladder (authoritative pins)

| Role | SHA |
| --- | --- |
| `development` campaign HEAD | `dc778d125fd7e7205edd30fbbf144ca12320b0ee` |
| TEST_EXECUTION_SHA | `32a9633440007eb4e8f00a825bbf2e399192b346` |
| EVIDENCE_PUBLICATION_SHA (B5 FULL) | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` |
| W2_NORMAL_FREEZE_EVIDENCE_PUBLICATION_SHA | `bc4b5f61666a8fea638064f50809fff2089b1b8b` |
| ACCEPTED_BASELINE_SHA | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` |
| W3_INDEPENDENT_AUDIT_PUBLICATION_SHA | `fd5728955605ef2ec013d69ec9290736f0970419` |
| W4_INDEPENDENT_CERTIFICATION_PUBLICATION_SHA | `5f178174e2462b038d544b61192a666d5b0ca79f` |
| B6_ACCEPTANCE_PUBLICATION_SHA (phase-1) | `add6a6ce2015ebaa4bc38b7fadf3936f100a1522` |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* |

## 2. Independent session identities

| Role | Session ID |
| --- | --- |
| W3 dossier auditor (operational) | `c8e4a91f-2b6d-4f7e-a3c1-9d5e7f2b4a68` |
| W3 publication auditor | `7c58500c-17c9-42a8-acbc-4b0e05487751` |
| W4 certifier | `4380025f-0a4e-43a2-bb90-ead2ce6d0bec` |

## 3. Verdicts

| Gate | Verdict |
| --- | --- |
| W3 independent audit | `PASS_SCOPED` |
| W4 certification | `PASS_SCOPED` |
| OD-ENT-02 sufficiency | `SUFFICIENT_COMPENSATING_SCOPED` (not LIVE) |
| FRZ-AUD promotion (`frz_aud_status`) | **OPEN** |
| Population E | CONCUR PASS |
| Matrix C | CONCUR (69 + 6 operator-trust replayed); overlay promotion deferred W4-03 |

## 4. CI evidence

| Run ID | headSha | Conclusion |
| --- | --- | --- |
| `42869847981` | `32a9633440007eb4e8f00a825bbf2e399192b346` | success (merge PR #442) |

## 5. Remaining blockers before OD-ENT-08

1. **W4-03** — dedicated FRZ-AUD-01..10 promotion commits + Matrix C overlay reconcile.
2. **Gate A** — `ent_audit_x_formal_closure_preflight.py --gate A` (D-3 preflight lane C).
3. **OPEN-BP-01** — live branch protection debt under compensating route.
4. **OD-ENT-08** — operator signing session (W5).
5. **Closure phase 1/2** — roadmap §5, `FORMAL_CLOSURE_PUBLICATION_SHA`, Gate B.

## 6. Evidence paths

- W3: `iiac_evidence/ENT-AUDIT-X/audit/w3_dc778d/`
- W4: `iiac_evidence/ENT-AUDIT-X/certification/w4_dc778d/`
- Handoff: `certification/w4_dc778d/W5_CERTIFIER_HANDOFF_RECORD.md`

## 7. Operator actions (W5 only)

1. Review this packet and W4 certifier report.
2. Authorize W4-03 promotion batch or delegate implementer lane.
3. Schedule `OD-ENT-08` signing session after Gate A green.
4. **Do not** interpret this packet as `ENT-AUDIT-X CLOSED`.
