# ARCH-FREEZE manifests: DRAFT (T-AF2-MANIFESTS, ticket T-IGX-050-G)

> **DRAFT. Nothing in this folder is frozen, accepted, certified or promoted.** It does not change the status of FRZ-FRZ-05, FRZ-FRZ-06, FRZ-FRZ-07 or FRZ-FRZ-08, which stay as the FRZ checklist and the FRZ acceptance matrix record them. The final manifests are generated from the G2 tree in W5 (ticket T-IGX-052-B), then independently audited. This draft only gives the RW product train and the W5 generation a reviewed shape to start from.

| Item | Value |
| --- | --- |
| Plan node | `T-AF2-MANIFESTS`, W2 draft (merged plan MEP-AF2 v2.1, Appendix Z) |
| Operator decision it answers | DP-40: define artefact, generator and owner for FRZ-FRZ-03 and FRZ-FRZ-05..09 (this draft covers 05..08) |
| Tree it was generated from | `origin/development` at `332c5217c390202a911ad5592e88a8f366845587` (tree `7ad728ffd26e45e7d2c401b165a35f092de9ac35`). The same values are in every JSON `meta` block |
| Generator | [`GENERATOR_REFERENCE_LISTING.py.txt`](GENERATOR_REFERENCE_LISTING.py.txt), a reference listing only (see below) |
| Generated summary | [`SUMMARY.md`](SUMMARY.md) (counts, produced by the generator) |

## Files

| File | FRZ row | What it records |
| --- | --- | --- |
| [`layer_manifest.draft.json`](layer_manifest.draft.json) | FRZ-FRZ-06 | The tier map (AGENTS.md tier line), rules R1..R4, one row per package unit, and every forbidden import edge found by an AST scan of the tracked non-test product files |
| [`contract_manifest.draft.json`](contract_manifest.draft.json) | FRZ-FRZ-05 | Every module under `intergrax/contracts/` and `intergrax/applications/contracts/` with its draft status and top-level symbols, the `PLANNED_MOVE (PT-WA)` entries, and the claim exclusions (`NOT_CLAIMED`, `OUTSIDE_FROZEN_SCOPE`, `TRANSFER_TO_LIVE_PROD_Q`, `BOUNDED_CLAIM`) |
| [`ownership_manifest.draft.json`](ownership_manifest.draft.json) | FRZ-FRZ-07 | The semantic owner and composition owner per concern, read from the CONFIG-X inventory, the EBH-5 platform concern register and the fixed-substrate decision register, with a check that every path they name exists |
| [`composition_manifest.draft.json`](composition_manifest.draft.json) | FRZ-FRZ-08 | The strict production roots, the composition, wiring and resolver modules and host entry points found by file pattern, and the `NOT_COMPOSED` entries |

## Mark vocabulary

| Mark | Source of the mark | Meaning in this draft |
| --- | --- | --- |
| `NOT_COMPOSED` | plan section E; `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md` Table A, rows whose quoted DS-EXC disposition is `ACCEPTABLE_SCOPED_WITH_EVIDENCE` or the dormant-item decision of EX-L-04 | The item is deliberately not composed. Plan section E allows this only with a fail-closed non-composition gate. **Every gate is `PENDING`**; none is built here |
| `NOT_CLAIMED` | OD-AF-18 (packet 1) for the E2B, Modal and Daytona sandbox providers (FRZ-SEC-05, FRZ-SEC-06); residual rows EX-D4-02 and EX-D5-01 | The freeze makes no claim about the subject |
| `OUTSIDE_FROZEN_SCOPE` | residual rows EX-G-02, EX-G-07, EX-D2-04, EX-D4-03, EX-D5-04 | The residual document says the exclusion needs a manifest entry |
| `TRANSFER_TO_LIVE_PROD_Q`, `BOUNDED_CLAIM` | residual rows EX-G-04 and EX-G-01 | Carried as quoted; the bounded claim points at the VBN fixed-substrate section |
| `PLANNED_MOVE (PT-WA)` | closed-world `RESULT.md` section 1.4 B (a proposal; the binding scope is T-AF2-RW-PLAN) | The neutral contract is planned to move to `intergrax/contracts/` and be re-exported from the old path |
| `DRAFT_PUBLIC_CANDIDATE`, `DRAFT_INTERNAL` | the path rule in the contract manifest (`public_surface_rule`) | Mechanical only: no `__all__` manifest exists for the contract roots, so a module is internal when a path component below the root starts with `_`. A reviewed public/internal split is a W5 task |

Dispositions quoted from the residual document are the Operator's recorded recommended options (packet 1, as that document states). They are carried here with attribution and are not re-asserted.

## What the draft surfaces (to be read, not to be taken as findings of record)

- **Layer.** The scan finds forbidden import edges under rules R1 and R4 (R2 finds none: 0 edges); the counts are in `SUMMARY.md`. R4 is the registry-bypass rule enforced by `scripts/maintenance/check_agent_registry_bypass.py`; it is not one of the three hard boundaries of AGENTS.md. Whether the app-facing packages `intergrax/cli`, `intergrax/harness`, `intergrax/scaffold`, `intergrax/debug` and `intergrax/agent_distribution` belong to Tier-1 or Tier-3 is not stated by any path rule; the draft keeps the path rule and lists their edges (open decision `LAYER-OPEN-1`).
- **`ExecutionMode` is already at Tier-0.** The closed-world result lists `ExecutionMode` among the neutral contracts to move. At the baseline commit it is defined in `intergrax/contracts/execution_mode.py` and `intergrax/applications/contracts/execution_mode.py` re-exports it, so for that symbol the planned PT-WA work is to repoint importers, not to move it. The entry keeps the `PLANNED_MOVE (PT-WA)` mark as requested and carries a `note` and `already_defined_at_tier0`.
- **Ownership drift.** Two paths named by registers do not exist in the tree: `intergrax/integrations/providers/external_work/*` (CONFIG-X row `integration.external_work`) and `intergrax/agents/persistence/*sqlite*` (fixed-substrate decision `runtime-sqlite-persistence-bundle`). They are listed with `exists_at_baseline: false`, not corrected.
- **Residual decisions without a derivable mark.** Four Operator-decision items (EX-G-03, EX-D2-06, EX-D4-05, EX-D7-03) say nothing that maps to a manifest mark; they are listed under `unmarked_operator_decisions` with the quoted text.

## How it was generated

The generator is a single stdlib-only script. It reads one checked-out tree through `git ls-files` and the AST; it imports and executes nothing from that tree, and it writes no timestamps, so two runs on the same tree give identical bytes. Its SHA-256 and the SHA-256 of every input document are recorded in `SUMMARY.md` and in each JSON `meta` block.

```bash
python -I gen_manifests.py --root <checkout of the baseline commit> --out <output folder>
```

The script is kept here only as a **reference listing** (`GENERATOR_REFERENCE_LISTING.py.txt`, a `.txt` suffix so that no tool collects or lints it as source). Per the task brief it is **not** placed under `scripts/`: adding a script there, and the drift gate that would read these files, need an Operator decision. To rerun it, copy the listing to a `.py` file outside the repository.

## Not included (needs the Operator, a later node or another session)

- A drift gate in `scripts/maintenance` that compares the manifests with the tree (it comes after the Operator decision on `scripts/`).
- The non-composition gates behind every `NOT_COMPOSED` mark (`T-AF2-RW-GATES`).
- The reviewed public/internal contract split, and any path-level ownership that no register states. The ownership manifest records only owners that the three registers state; the matrix row for FRZ-FRZ-07 names the known gaps (security, sandbox, data lifecycle).
- A named owner for each artefact. This draft does not assign one; the matrix names `T-B7-FRZ-FRZ` as accountable.
- The final generation from the G2 tree (W5, T-IGX-052-B) and every independent audit row.
