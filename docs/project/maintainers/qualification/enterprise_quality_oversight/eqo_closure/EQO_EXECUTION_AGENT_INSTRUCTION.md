# EQO execution agent instruction

**Language:** Operator chat may be PL; all EQO commits, PR titles/bodies, and `gh pr create` text must be **English only** (`agent-env/rules/intergrax-english-only-commits.md`).

1. Read `EQO_PIN_BIND.json`; if pins drift, run EQO-RV-01 not full W0.
2. Work only in `eqo_closure/` and isolated test worktrees (`eqo-w3`).
3. Pre-FCS: execute WB DES→EXE→EVD→DIS→AUD on TRAIN_PIN; do not wait for FCS to find critical gaps.
4. Do not edit roadmap, FRZ, or product code unless in an authorized train ticket session (pasmo D).
5. Update closure matrix when tasks CLS; terminal status only after W8/W9.
6. FCS ≠ ARCH-FREEZE CLOSED; Campaign D only after M3 candidate exists.
