# T-IGX-043-E — EQO-A remediation code wave, gate slice of FC-2: capacity and sandbox maintenance gate scripts

## Scope

Gate-meaning change, split out of `T-IGX-043-B` (FC-2) after the independent delegate review. It edits two maintenance gate scripts, `scripts/maintenance/check_production_capacity_adapters.py` (AUDIT-IDEAL-30.4) and `scripts/maintenance/check_sandbox_policy_wiring.py` (AUDIT-IDEAL-11.1), so that they assert the fail-closed behaviour that the FC-2 product part introduces: capacity wiring without a cluster is reported unqualified and an in-memory backend never yields a passed probe (DP-29), and a sandbox profile without an explicit root is refused for side-effect tools in PRODUCT/STRICT (DP-24, DP-25). Both scripts are run by `scripts/gates/check_audit_ideal_gates.py`. Read at `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42`; both scripts are unchanged since the baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`. By reading, the capacity script is already red at that head independently of FC-2 (it calls the wiring with a boundary-less governance on a PRODUCT profile, which raises, and an existing test pins that raise); part 1 covers repairing it. Write set: the two scripts and the ticket and plan files; no product path, no test file.

**Authority:** this is a gate-meaning change. Master-plan node `PT-G01` owns `scripts/maintenance/check_*` and states that a gate meaning change is not delegable; the approval of this ticket is therefore the Operator's personal ruling and cannot be written by a delegate or a commissioned session. If the owner of `PT-G01` absorbs these two edits, this ticket is withdrawn.

**Ruling recorded:** from the Operator's chat answers as transcribed by the orchestrating session on 2026-11-16 in `.tmp/session/eqo-a-remediation/FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a session artefact under `.tmp/`, a draft transcription that is not yet a repository record and not an Operator signature), the Operator's personal ruling accepts both parts as drafted and allows the sandbox part to be ruled out separately without affecting `T-IGX-043-B`; the merge order B, then A, then C, then F is confirmed. The approval line below stays the Operator's own and is not written; milestone `M1` is not confirmed by that transcription and stays open.

**Two parts, ruled separately if the Operator wishes:** part 1 repairs and strengthens the capacity gate, which is red at the current head independently of FC-2 (by reading; not executed); part 2 changes the sandbox gate, which the FC-2 sandbox guard makes red. **Dependency:** the product part of `T-IGX-043-B` assumes that part 2 lands in the same wave (same integration train); it has no dependency on part 1. If part 2 is ruled out, `T-IGX-043-B` stays approvable, the sandbox gate stays red in the gate runner and is owned by the owner of `PT-G01` together with the Operator (plan section 1). This ticket uses names defined by `T-IGX-043-B` (`CapacityBackendPosture`, `CapacityQualification`, `SandboxCompositionError`) and is re-aligned if they change. The Operator's personal ruling on this ticket is a prerequisite of the wave, not of the approval of `T-IGX-043-B`. Whether the two scripts are IIAC-material and whether `check_audit_ideal_gates.py` runs in CI are UNVERIFIED. Nothing in this ticket claims closure, PASS, FRZ promotion or independent verification.

## Plan digest

Plan digest: 80402c1289dd1757b98ec0867c47f483c1ef3e2ef1c8d2e92581ee520646d86b

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-E.md`](plans/PLAN_T-IGX-043-E.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line goes here after plan review. Because this ticket changes the meaning of two gates, it is the Operator's personal ruling and is not written by a delegate or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time under the shared test lock; none run when this ticket was drafted:

- `uv run python scripts/maintenance/check_production_capacity_adapters.py` (first on the unedited script for the red-first record, then edited)
- `uv run python scripts/maintenance/check_sandbox_policy_wiring.py` (same order)
- `uv run pytest tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_30_4_production_capacity tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_11_1_sandbox_product_requirement -p no:cacheprovider`
- `uv run python -m scripts.governance.igx_gov ticket` before push
- Reviewer check: `git diff --stat` lists exactly the two scripts plus this ticket and its plan.
- The full runner `scripts/gates/check_audit_ideal_gates.py` is not part of this contract (broad); it needs a separate Operator authorisation. The independent mutation check (plan section 5, E-3) belongs to the wave audit at the exact wave SHA by a session that did not implement or review it.

## Out of scope

Every product path (`intergrax/`, `agents/`, `applications/`), every test file, `scripts/gates/check_audit_ideal_gates.py`, all other `check_*` scripts, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, IIAC records, GitHub settings, production configuration, Docker, FRZ promotion, and the product guards themselves (`T-IGX-043-B`).

## Lifecycle state

`draft`
