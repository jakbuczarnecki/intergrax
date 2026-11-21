# EQO-W3-WB12-AUD — session E verdict (wave AUD-1)

**Verdict (WB12): `BLOCKED`**
**Auditor session:** independent session E (not the EXE agent; no EXE/DIS artefact in this WB authored by this session).
**Date:** 2026-11-21
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469` (source read at the pin, read-only)
**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Verdict vocabulary

| Verdict | Meaning |
|---------|---------|
| `PASS` | Evidence at the pin supports the stated disposition. |
| `FAIL` | Source at the pin confirms the defect or contradicts the stated disposition. The finding stays OPEN. |
| `BLOCKED` | The pytest slice does not produce evidence to decide, or an owner/operator decision is a prerequisite. |

Aggregate rule: any `FAIL` gives `FAIL`; otherwise any `BLOCKED` gives `BLOCKED`; otherwise `PASS`.

## Packet checks

| Check | Result |
|-------|--------|
| TRAIN_PIN in manifest / DES_PREREG / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `cee7f64c…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 33 passed, 1 skipped |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** DUP-X closed-world and authority gates, ProofReceipt contract tests for `intergrax.proofs.receipts`, and the P6R trace-alias caller gate. The one skip is `test_ent_a_ent_a_dup_01.py::test_acceptance_ent_a_dup_01_semantic_gate`, skipped unless `ENT_AUDIT_X_SEMANTIC=1`. That is the semantic duplicate-mechanism gate, the test most relevant to A4-047 and A5-010. The DUP-X closed-world tests are structural (catalog, census and family counts are non-empty or complete).

**Disposition quality:** all six DIS rows carry the same boilerplate ("band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone"). `ACCEPTED_SCOPED_LIMITATION` names no scope, owner or reopen condition.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A3-019 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | The hypothesis is that row-level PASS evidence is tied to certification-time SHAs while production code changed later. The slice has no SHA-versus-delta check. The packet does not cite a replay of EXE/GOV/CTL/REC/HRN rows at a current SHA, so the claim is neither confirmed nor refuted. |
| EQOA-A4-039 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | Duplication confirmed at the pin: two `ProofReceipt` models (`intergrax/contracts/execution_evidence/receipt.py:26` and `intergrax/proofs/receipts/contracts.py:30`), plus `ProofReceiptStore` as a class (`intergrax/proofs/receipts/store.py:19`) and as a Protocol (`applications/governed_contractor_application/host/stores.py:83`). The slice tests only `intergrax.proofs.receipts`, so it cannot show the models are safely separated. Acceptance needs an owner decision (unify, rename, or sanction) with a recorded scope. LOW severity. |
| EQOA-A4-047 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | The semantic duplicate-mechanism gate (ENT-A-DUP-01) was skipped, and the four-mechanism scenario overlap and the SFX closed-world resolver mapping have no test in the slice. The master matrix rates its freeze impact NON_BLOCKING_WITH_EVIDENCE, but that evidence is absent from this packet. |
| EQOA-A5-010 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | The question is whether 70 product modules added since the acceptance pin were re-examined. The slice's closed-world checks count catalogs and registries. They do not diff modules against the acceptance pin, and the semantic gate was skipped. No evidence about the added modules is in the packet. |
| EQOA-A8-020 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | Not decidable by pytest. The Protocol-v2 remediation (LCI trusted tenant scope, Token Optimization receipt identity) appears only in the EQO dossiers and the debt register. `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` has no "receipt identity" entry at the pin, and no roadmap or FRZ tracking row was found. Closure needs an owner/operator decision to track it (roadmap or FRZ row) or to except it. |
| EQOA-A2-031 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | The NPSC-4.2 allowlist ("FAIL retained" text versus the allowlist gate refreshed under EBH-7 W3) is not exercised by any test in the slice. The packet has no recorded result for that gate. |

## Rationale for the aggregate

`BLOCKED`: none of the six hypotheses is falsified or confirmed by the slice. The only genuinely relevant test (ENT-A-DUP-01 semantic gate) was skipped. One finding (A4-039) shows confirmed duplication that needs an owner decision. No stated disposition is contradicted, so this is not `FAIL`, but no finding qualifies for `PASS`.

## Required follow-up (no product fix here)

1. Re-run the slice with `ENT_AUDIT_X_SEMANTIC=1` at the pin and attach the log (A4-047, A5-010).
2. A4-039: owner decision on the two `ProofReceipt` models and stores, recorded as a scoped limitation or a rename ticket.
3. A3-019: replay the EXE/GOV/CTL/REC/HRN rows at a current SHA, or record the SCOPED replay as the accepted basis.
4. A8-020: owner/operator decision to track the Protocol-v2 remediation or except it.
5. A2-031: run or cite the NPSC-4.2 allowlist gate result at the pin.
6. A5-010: produce the module delta since the acceptance pin and check it against the DUP-X census.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
