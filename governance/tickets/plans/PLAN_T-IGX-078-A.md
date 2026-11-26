# PLAN T-IGX-078-A - AF-CLOSE K0.3 ladder amendment and bookkeeping repair (docs)

Records the 2026-11-26 decisions D-1..D-4 and repairs ARCH-FREEZE bookkeeping. It sets no gate: not SCOPE, RW-PLAN-AUDIT PASS, G2, R, FCS, M3 or ENTRY. Base: `646e63953`. Accepted by the AF-CLOSE orchestrator with `GO K0.3`.

Write set (`Q/` = `docs/project/maintainers/qualification/`, `P/` = `Q/arch_freeze_program/`):

- `P/MEP-AF2_v2.1.md`: append Appendix V6 / AD-AF-DEVTRAIN, marked not signed; earlier text unchanged.
- `Q/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`: rows OD-AF-CLOSE-D1..D4 and AD-AF-DEVTRAIN; role-registry change as a proposal only.
- `Q/ARCH_FREEZE_SHA_LADDER.md`: current header, rungs AUDIT_EXECUTION and CERTIFICATION_PUBLICATION, V6 definitions pending signature, candidate `ea923cf03` superseded, undefined P1/P2 labels removed.
- `P/ARCH_FREEZE_NODE_LEDGER.md`: W0 rows, T-AF2-RED-CLASSES, one status per gate (SCOPE, RW-PLAN-AUDIT), histogram.
- `P/continuation/`: banners on `RW_PLAN_AUDIT_STATUS.md`, `RW_PLAN_AUDIT_EVIDENCE_INDEX.md`, `SCOPE_PREDECESSOR_INDEX.md`; retirement banners on `TRAIN_DEV_GAP.md`, `TRAIN_CHERRY_PICK_STATUS.md`; label removal in `CAMPAIGN_B_MISSING_CASE_MAP.md`.
- `P/WP12_DAG.md`: regenerated from an untracked copy of the generator, after a baseline run reproduces the tracked file. Adds ECAP-2C, PROMOTE-*, COORD, W8, ECAP-EXEC and the nodes required by the section C edges; DS-ADM moves to W8.
- `P/README.md`, `P/SOURCES.sha256`: pointers and digests, including the stale entries, each with its commit.
- `governance/tickets/T-IGX-053-I.md`, `T-IGX-065-A.md`, `T-IGX-073-E.md`: add an Owner section only.
- `governance/tickets/T-IGX-078-A.md`, `governance/tickets/plans/PLAN_T-IGX-078-A.md`.

The EC campaign F packet is untracked in the main checkout. This ticket does not edit it; an addendum text goes to the orchestrator's session folder.

Out of scope: `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, `governance/agent-control-plane/role_registry.yaml`, roadmap, FRZ checklist, any signature or approval line.

Test: `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` (also `--self-test`); `uv run python scripts/governance/verify_rw_plan_phase_check3.py`; `sha256sum -c SOURCES.sha256` in `P/`; DAG generator exit 0; `uv run python -m scripts.governance.igx_gov ticket --profile smoke`.
