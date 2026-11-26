# ARCH-FREEZE SHA ladder

| Field | SHA / value | Status @ `development` `646e63953` (2026-11-26) |
| --- | --- | --- |
| AUDIT_BASELINE_SHA | a0624e28ff77b061e83f3e10aab4c2a9083ee1b4 | **LOCKED** (immutable planning lock) |
| B1_EXECUTION_SHA | 6bdf78c626bfcb347ad3ad9e3e18b65222f1b169 | **LOCKED** (wave-1 B1) |
| B1_INVENTORY_DIGEST | 0ffa2676… (full digest in B1 handoff) | reproducible @ B1 EXEC |
| B2_B4_EXECUTION_SHA | f7ce9daebad35537da974de0dda3962126c6030e | **LOCKED** (wave-1 B2–B4 + L3 FULL 85/85) |
| L3_FULL_G6_DIGEST | 0fe966e4… (G6 L3 bundle) | immutable G6 replay |
| G6_INDEPENDENT_AUDIT_SHA | 3d96691f1b02b393a4de189139a6e1347db99382 | **LOCKED** (wave-1 G6 @ AUDIT HEAD) |
| CLOSURE_DOC_HEAD | 950097ea4bcfd5bba828f663bcd993b8bd1e1d63 | current planning / R0 SSOT (mutable only via authorized R0 batches) |
| IMPLEMENTATION_SHA | TBD at **T-B7-LOCK-FREEZE-SHA** | OPEN |
| FREEZE_CERTIFICATION_SHA | TBD (= IMPLEMENTATION_SHA at admission) | OPEN |
| TEST_EXECUTION_SHA | TBD per certifying run | OPEN |
| EVIDENCE_PUBLICATION_SHA | TBD | OPEN |
| EVOL_B6_ACCEPTANCE_PUBLICATION_SHA | `1dd769331506149a1ee2f411c2bc62e85e5bbf25` | **LOCKED** (IIAC B6 acceptance — [`evol_x/EVOL_X_SHA_LADDER.md`](./evol_x/EVOL_X_SHA_LADDER.md); ODR **OD-EVOL-11** SUPERSEDED by IIAC; not FRZ promotion) |
| ENT_B6_ACCEPTANCE_PUBLICATION_SHA | `add6a6ce2015ebaa4bc38b7fadf3936f100a1522` | **LOCKED** (IIAC W2-04 phase-1 — [`ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md`](./ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md) L62; T-IGX-024-B) |
| ACCEPTED_BASELINE_SHA | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` | **LOCKED** (**OD-ENT-01** EXECUTED — [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](./ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md); ENT ladder L40) |
| FORMAL_CLOSURE_PUBLICATION_SHA | unset (distinct per stage: EVOL / ENT / ARCH-FREEZE) | OPEN |
| OD-IC-01 INTERNAL_CERTIFICATION | 5c08e5e6c44d21998c76bd27ad1efab172440f45 | immutable |
| POST_IC_CONSOLIDATION | 81829712acc19ebeadb4773467b9c5642ca975e3 | immutable |
| EVOL TEST_EXECUTION_SHA | 3c4a641b3e48a13f437e830a0562e2a80b03aabb | reusable evidence |
| ENT TEST_EXECUTION_SHA | 89c76eb7b53a0a3ce6f75ab395cf4a2802638a91 | reusable evidence |
| EXECUTION_WAVE1_SHA (legacy label) | 18a0f68fc6c8107e6344fdb4641ecec1945dae5e | superseded by B1/B2–B4 rungs above |
| AUDIT_BASELINE_2 | 7960ae7f6c82da60ca9acdc0ceebf63810002a42 | **PINNED 2026-11-16** — `origin/development` at PR #604 merge; start pin of MEP-AF2 v2.1 (T-AF2-BASELINE-PIN). Does not replace `AUDIT_BASELINE_SHA` above. |
| G2 | NOT SET | **NOT SET** - the `development` commit after the last product change; wave-audited SHA; the held-path freeze starts here (MEP-AF2 Appendix V6) |
| R | NOT SET | **NOT SET** - RB-2 phase-2 docs commit; `git diff G2 R` must be docs only (Appendix V6) |
| FCS | NOT SET | **NOT SET** - "product freeze confirmation" docs commit; held paths equal to G2 (Appendix V6) |
| M3 | NOT SET | **NOT SET** - EBH-7 final pin; product tree equal to G2; planned to equal FREEZE_CERTIFICATION_SHA and TEST_EXECUTION_SHA (Appendix V6) |
| AUDIT_EXECUTION_SHA | NOT SET | **NOT SET** - rung named by MEP-AF2 v2.1 section J and N1 |
| CERTIFICATION_PUBLICATION_SHA | NOT SET | **NOT SET** - rung named by MEP-AF2 v2.1 section J and N1 |

**Rungs added 2026-11-16** (ticket `T-IGX-050-A`; definitions in [`arch_freeze_program/MEP-AF2_v2.1.md`](./arch_freeze_program/MEP-AF2_v2.1.md) section J): `AUDIT_BASELINE_2` is pinned; `G2`, `R`, `FCS` and `M3` are placeholders and nothing here claims they exist. The `Status @ CLOSURE_DOC_HEAD` column header above still describes the older rungs.

**V4 read pin (2026-11-25, not a ladder rung):** canonical integration head is `origin/development` `7657745c38500653c57980b0b477f611a86cccdc`. It is not `G2`, `R`, `FCS`, or `M3`. PT-Q02 candidate `ea923cf03d88fc9d22637796e9002eb9d68e6e59` is stale and is not an ancestor of this pin.

**V5 read pin (2026-11-25, not a ladder rung):** canonical integration head for the V5 annex is `5935fa4f0cd3596852b3a6500f9d4e1d613a53df`. It is **not** `G2`, **not** `R`, **not** `FCS`, and **not** `M3`. Evidence merges `2411e4118` and `9a5b796d9` (parents include `e32f82890`) publish the V5 evidence onto `development` and are **not** ladder rungs. `origin/arch-freeze/rw-train` remains `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a`. The V4 pin `7657745c` is historical. PT-Q02 candidate `ea923cf03d88fc9d22637796e9002eb9d68e6e59` is stale and is **not** `G2`.

**Update 2026-11-26** (ticket `T-IGX-078-A`; not a ladder rung promotion):

- `AUDIT_EXECUTION_SHA` and `CERTIFICATION_PUBLICATION_SHA` rows added, both NOT SET.
- `G2`, `R`, `FCS` and `M3` now carry the definitions of [`arch_freeze_program/MEP-AF2_v2.1.md`](./arch_freeze_program/MEP-AF2_v2.1.md) Appendix V6 (AD-AF-DEVTRAIN). The appendix is **signed** (2026-11-26, as relayed; see its signature record). The signature sets no rung: all four stay NOT SET.
- PT-Q02 candidate `ea923cf03d88fc9d22637796e9002eb9d68e6e59` is **SUPERSEDED**: no `arch-freeze/rw-train` tip is a `G2` candidate (Appendix V6). `11787a7e196cbfea32a982bfa260f02a34fe05fc` is the historical train absorption, not `G2` and not `FCS`.
- The status column header was refreshed. The last sentence of the 2026-11-16 note above is historical, as are the train statements in the V4 and V5 notes.

**Rules:** Do not overwrite G6 / B1 / B2–B4 rungs. **FREEZE_CERTIFICATION_SHA** is the single consumer SHA for L3 FREEZE, ECAP §2.C, VBN independent accept, and authorized FRZ promotions.
